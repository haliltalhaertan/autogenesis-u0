#!/usr/bin/env python3
import json, math, hashlib
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL_PATH=ROOT/'PHASE_B_FROZEN_PROTOCOL.json'
SIDECAR_PATH=ROOT/'PHASE_B_FROZEN_PROTOCOL.sha256'

def protocol_sha256(path=PROTOCOL_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_protocol(verify=True):
    p=json.loads(PROTOCOL_PATH.read_text())
    if verify:
        got=protocol_sha256()
        exp=SIDECAR_PATH.read_text().split()[0]
        if got!=exp: raise RuntimeError(f'protocol hash mismatch {got} != {exp}')
    return p

P=load_protocol()
RC=float(P['candidate_neighbors']['R_candidate'])
RB=float(P['bond_rule']['retention_break_distance_max'])
RREP=float(P['dimensionless_physics']['nonbonded_repulsion']['cutoff_r_rep'])
KS=float(P['dimensionless_physics']['spring_strength_k_s'])
KREP=float(P['dimensionless_physics']['nonbonded_repulsion']['strength_k_rep'])
GAMMA=float(P['dimensionless_physics']['damping_gamma'])
DT=float(P['dimensionless_physics']['timestep_dt'])
R0=float(P['dimensionless_physics']['bond_equilibrium_distance_r0'])
LO=float(P['node_rule']['birth_active_bonded_neighbor_fraction_inclusive'][0])
HI=float(P['node_rule']['birth_active_bonded_neighbor_fraction_inclusive'][1])

@dataclass
class State:
    x: np.ndarray
    u: np.ndarray
    active: np.ndarray
    bonds: frozenset
    def copy(self):
        return State(self.x.copy(),self.u.copy(),self.active.copy(),frozenset(self.bonds))

def ek(i,j):
    i=int(i);j=int(j)
    return (i,j) if i<j else (j,i)

def canonical_bonds(edges):
    out=set()
    for i,j in edges:
        if i==j: raise AssertionError('self bond')
        out.add(ek(i,j))
    return frozenset(out)

def degrees(n,bonds):
    d=np.zeros(n,dtype=np.int16)
    for i,j in bonds:
        d[i]+=1;d[j]+=1
    return d

def candidate_edges_naive(x, cutoff=RC):
    n=len(x); c2=cutoff*cutoff
    out=set()
    for i in range(n):
        for j in range(i+1,n):
            dr=x[j]-x[i]
            if float(np.dot(dr,dr)) <= c2:
                out.add((i,j))
    return frozenset(out)

def proximity_adjacency(n,edges):
    adj=[set() for _ in range(n)]
    for i,j in edges:
        adj[i].add(j);adj[j].add(i)
    return adj

def common_active_support(edges,n,active):
    adj=proximity_adjacency(n,edges)
    scores={}
    for i,j in edges:
        # integer support; no iteration-order dependence
        if len(adj[i])<=len(adj[j]): small,other=adj[i],adj[j]
        else: small,other=adj[j],adj[i]
        scores[(i,j)]=sum(1 for k in small if k in other and bool(active[k]))
    return scores

def node_update(active,bonds):
    n=len(active); adj=[[] for _ in range(n)]
    for i,j in bonds:
        adj[i].append(j);adj[j].append(i)
    na=np.zeros(n,dtype=np.bool_)
    for i,qs in enumerate(adj):
        if not qs: continue
        frac=sum(bool(active[q]) for q in qs)/len(qs)
        if LO <= frac <= HI: na[i]=True
    return na

def retained_edges(state):
    kept=set()
    a=state.active;x=state.x
    for i,j in state.bonds:
        if not (bool(a[i]) or bool(a[j])): continue  # II fails A* retention
        r=float(np.linalg.norm(x[j]-x[i]))
        if r<=RB: kept.add((i,j))
    return frozenset(kept)

def eligible_proposals(state, v, candidates, support):
    n=len(state.active); olddeg=degrees(n,state.bonds)
    out=set();a=state.active
    for e in candidates:
        i,j=e
        if e in state.bonds: continue
        if not (bool(a[i]) or bool(a[j])): continue
        if olddeg[i] >= v or olddeg[j] >= v: continue
        if support[e] < 1: continue
        out.add(e)
    return frozenset(out)

def allocate_A(elig,kept,n,v):
    kd=degrees(n,kept); inc=Counter()
    for i,j in elig: inc[i]+=1;inc[j]+=1
    accept={p:(cnt <= v-int(kd[p])) for p,cnt in inc.items()}
    return frozenset(e for e in elig if accept.get(e[0],False) and accept.get(e[1],False))

def allocate_B(elig,kept,n,v,support):
    kd=degrees(n,kept); incident=defaultdict(list)
    for e in elig:
        incident[e[0]].append(e);incident[e[1]].append(e)
    accepted={}
    for p,es in incident.items():
        rem=v-int(kd[p]); groups=defaultdict(list)
        for e in es: groups[support[e]].append(e)
        acc=set()
        for s in sorted(groups,reverse=True):
            g=groups[s]
            if len(g)<=rem:
                acc.update(g);rem-=len(g)
            else:
                break
        accepted[p]=acc
    return frozenset(e for e in elig if e in accepted.get(e[0],set()) and e in accepted.get(e[1],set()))

def allocate_C(elig,kept,n,v):
    kd=degrees(n,kept); T=set(elig)
    for _ in range(2):
        inc=Counter()
        for i,j in T: inc[i]+=1;inc[j]+=1
        overloaded={p for p,c in inc.items() if int(kd[p])+c>v}
        if not overloaded: break
        rem={e for e in T if e[0] in overloaded and e[1] in overloaded}
        if not rem: break
        T.difference_update(rem)
    inc=Counter()
    for i,j in T: inc[i]+=1;inc[j]+=1
    overloaded={p for p,c in inc.items() if int(kd[p])+c>v}
    return frozenset(e for e in T if e[0] not in overloaded and e[1] not in overloaded)

def topology_transition(state,v,allocator,candidate_edges=None,support=None):
    n=len(state.active)
    candidates=candidate_edges if candidate_edges is not None else candidate_edges_naive(state.x)
    sup=support if support is not None else common_active_support(candidates,n,state.active)
    na=node_update(state.active,state.bonds)  # synchronous OLD bonds
    kept=retained_edges(state)
    elig=eligible_proposals(state,v,candidates,sup)
    if allocator=='A_batch_all': formed=allocate_A(elig,kept,n,v)
    elif allocator=='B_support_orbit': formed=allocate_B(elig,kept,n,v,sup)
    elif allocator=='C_overload_prune': formed=allocate_C(elig,kept,n,v)
    else: raise ValueError(allocator)
    nb=canonical_bonds(set(kept)|set(formed))
    if len(nb)!=len(set(nb)): raise AssertionError('bond symmetry')
    d=degrees(n,nb)
    if len(d) and int(d.max())>v: raise AssertionError(f'strict valence violated {d.max()}>{v}')
    return na,nb

def forces_reference(x,u,bonds):
    n=len(x);F=-GAMMA*u.copy(); bset=set(bonds)
    for i in range(n):
        for j in range(i+1,n):
            dr=x[j]-x[i]; r=float(np.linalg.norm(dr))
            if r<1e-12: raise FloatingPointError('coincident pair')
            rh=dr/r
            if (i,j) in bset:
                fij=KS*(r-R0)*rh
            elif r<RREP:
                fij=-KREP*(RREP-r)*rh
            else:
                continue
            F[i]+=fij;F[j]-=fij
    if not np.isfinite(F).all(): raise FloatingPointError('nonfinite force')
    return F

def step(state,v,allocator,motion=True):
    c=candidate_edges_naive(state.x)
    sup=common_active_support(c,len(state.active),state.active)
    na,nb=topology_transition(state,v,allocator,c,sup)
    if motion:
        F=forces_reference(state.x,state.u,nb)
        nu=state.u + DT*F
        nx=state.x + DT*nu
    else:
        nu=np.zeros_like(state.u);nx=state.x.copy()
    if not np.isfinite(nx).all() or not np.isfinite(nu).all(): raise FloatingPointError('NaN/Inf')
    return State(nx,nu,na,nb)

def serialize_hash(state):
    h=hashlib.sha256()
    h.update(np.ascontiguousarray(state.active.astype(np.uint8)).tobytes())
    h.update(np.array(sorted(state.bonds),dtype=np.int32).tobytes())
    h.update(np.ascontiguousarray(state.x,dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(state.u,dtype=np.float64).tobytes())
    return h.hexdigest()

def pair_force_spring(xi,xj):
    dr=np.asarray(xj,float)-np.asarray(xi,float);r=float(np.linalg.norm(dr))
    if r<1e-12: raise FloatingPointError('coincident')
    return KS*(r-R0)*(dr/r)

def pair_force_repulsion(xi,xj):
    dr=np.asarray(xj,float)-np.asarray(xi,float);r=float(np.linalg.norm(dr))
    if r<1e-12: raise FloatingPointError('coincident')
    if r>=RREP:return np.zeros(3)
    return -KREP*(RREP-r)*(dr/r)
