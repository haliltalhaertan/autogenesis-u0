import sys, itertools
from fractions import Fraction
from collections import Counter
LO,HI=Fraction(1,6),Fraction(1,2); WLO,WHI=Fraction(1,3),Fraction(5,11)

def run(n):
    pairs=list(itertools.combinations(range(n),2)); L=len(pairs)
    perms=list(itertools.permutations(range(n)))
    recs=[]
    for mask in range(1,1<<L):
        edges=[pairs[i] for i in range(L) if mask>>i&1]
        adj=[0]*n
        for u,v in edges: adj[u]|=1<<v; adj[v]|=1<<u
        deg=[bin(a).count("1") for a in adj]; m=len(edges)
        for A in range(1,1<<n):
            P=((1<<n)-1)^A
            # P independent?
            bad=False
            for i in range(n):
                if P>>i&1 and (adj[i]&P): bad=True; break
            if bad: continue
            # fixed?
            new=0
            for i in range(n):
                if deg[i]==0: continue
                a=bin(adj[i]&A).count("1")
                f=Fraction(a,deg[i])
                if LO<=f<=HI: new|=1<<i
            if new!=A: continue
            s=Fraction(sum(deg[i] for i in range(n) if P>>i&1),2*m)
            # connected?
            seen=1; st=[0]
            while st:
                x=st.pop()
                for y in range(n):
                    if adj[x]>>y&1 and not seen>>y&1: seen|=1<<y; st.append(y)
            conn=(seen==(1<<n)-1)
            # colour-preserving canonical form
            canon=(n,)+min(
                (tuple(sorted(tuple(sorted((p[u],p[v]))) for u,v in edges)),
                 tuple(sorted(p[i] for i in range(n) if A>>i&1)))
                for p in perms)
            recs.append((s,conn,canon))
    return recs

allr=[]
per_n={}
for n in range(2,7):
    r=run(n); per_n[n]=len(r); allr+=r
    print(f"n={n}: {len(r)} labelled live cover fixed points")

tot=len(allr)
conn=[r for r in allr if r[1]]
floor=[r for r in allr if r[0]==WLO]
ceil=[r for r in allr if r[0]==WHI]
viol=[r for r in allr if not (WLO<=r[0]<=WHI)]
cls=len({r[2] for r in allr}); ccls=len({r[2] for r in conn})
fcls=len({r[2] for r in floor}); fccls=len({r[2] for r in floor if r[1]})
print(f"\nlabelled total          {tot}   violations {len(viol)}   ceiling {len(ceil)}")
print(f"connected labelled      {len(conn)}")
print(f"coloured iso classes    {cls}    connected {ccls}")
print(f"at floor: labelled      {len(floor)}   iso classes {fcls}   connected iso {fccls}")
print(f"\nfloor share  all-labelled {100*len(floor)/tot:.1f}%   "
      f"connected {100*len([r for r in floor if r[1]])/len(conn):.1f}%   "
      f"iso {100*fcls/cls:.1f}%   connected-iso {100*fccls/ccls:.1f}%")
