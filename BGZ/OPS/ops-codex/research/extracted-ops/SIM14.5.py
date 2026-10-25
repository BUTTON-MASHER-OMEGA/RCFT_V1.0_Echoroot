# SIM14.5 — AFFINE–SYMPLECTIC COMPATIBILITY CLASSIFICATION
from itertools import permutations, combinations
from collections import Counter, deque
import numpy as np

X=tuple(range(8)); ID=X
def compose(a,b): return tuple(a[b[i]] for i in X)
def inv(a):
    z=[0]*8
    for i,j in enumerate(a): z[j]=i
    return tuple(z)
def generated_group(gens):
    S={ID}; q=deque([ID])
    while q:
        a=q.popleft()
        for b in gens:
            for c in (compose(a,b),compose(b,a)):
                if c not in S: S.add(c); q.append(c)
    return frozenset(S)
def order(a):
    z=ID
    for n in range(1,100):
        z=compose(a,z)
        if z==ID:return n
    raise RuntimeError
def conjugate(a,g): return compose(compose(a,g),inv(a))
def commutator(a,b): return compose(compose(compose(a,b),inv(a)),inv(b))
def cyc(a):
    seen=set(); out=[]
    for i in X:
        if i in seen: continue
        c=[]; j=i
        while j not in seen: seen.add(j); c.append(j); j=a[j]
        if len(c)>1: out.append(tuple(c))
    return tuple(out)
def cnot(a): return "()" if not cyc(a) else "".join("("+" ".join(map(str,c))+")" for c in cyc(a))
def mapped_set(a,S): return frozenset(a[x] for x in S)
def mapped_edges(a,E): return frozenset(frozenset((a[u],a[v])) for u,v in map(tuple,E))
def Pmat(a):
    P=np.zeros((8,8),dtype=int)
    for i in X:P[a[i],i]=1
    return P
def hist(v): return dict(sorted(Counter(v).items(),key=lambda x:repr(x[0])))
def orbit(H,x): return frozenset(g[x] for g in H)

C8=frozenset(frozenset(e) for e in [(0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,7)])
p=(1,0,3,2,5,4,7,6); h=(7,6,5,4,3,2,1,0); s=(2,3,0,1,6,7,4,5)
K=generated_group([p,h]); G=generated_group([p,h,s])
OMEGA=np.zeros((8,8),dtype=int)
for a,b in [(0,2),(1,3),(4,6),(5,7)]: OMEGA[a,b]=1; OMEGA[b,a]=-1
ABS=np.abs(OMEGA)

# derive 7 V4s, 14 affine planes, 7 parallel classes
V=set()
for a,b in combinations([g for g in G if g!=ID],2):
    H=generated_group([a,b])
    if len(H)==4: V.add(H)
V=tuple(V)
planes=set(); pcs=[]
for H in V:
    unseen=set(X); oo=[]
    while unseen:
        x=min(unseen); O=frozenset(g[x] for g in H); oo.append(O); planes.add(O); unseen-=O
    pcs.append(frozenset(oo))
PLANES=tuple(sorted(planes,key=lambda z:tuple(sorted(z))))
PCS=tuple(sorted(set(pcs),key=repr))
PI={P:i for i,P in enumerate(PLANES)}

def affine(a): return frozenset(mapped_set(a,P) for P in PLANES)==frozenset(PLANES)
def oclass(a):
    T=Pmat(a).T@OMEGA@Pmat(a)
    if np.array_equal(T,OMEGA):return "+Omega"
    if np.array_equal(T,-OMEGA):return "-Omega"
    return "neither"
def absok(a): return np.array_equal(Pmat(a).T@ABS@Pmat(a),ABS)
def sok(a): return conjugate(a,s)==s
def c8ok(a): return mapped_edges(a,C8)==C8

AUT=[]; PLUS=[]; MINUS=[]; AST=[]; SST=[]; AC8=[]
for a in permutations(X):
    if c8ok(a): AC8.append(a)
    if affine(a):
        AUT.append(a)
        if oclass(a)=="+Omega":PLUS.append(a)
        elif oclass(a)=="-Omega":MINUS.append(a)
AUT=frozenset(AUT); PLUS=frozenset(PLUS); MINUS=frozenset(MINUS); N=PLUS|MINUS
AFFABS=frozenset(a for a in AUT if absok(a)); AFFS=frozenset(a for a in AUT if sok(a)); AC8=frozenset(AC8)

def center(H): return frozenset(a for a in H if all(compose(a,b)==compose(b,a) for b in H))
def derived(H): return generated_group([commutator(a,b) for a in H for b in H])
def classes(H):
    rem=set(H); out=[]
    while rem:
        x=next(iter(rem)); C=frozenset(conjugate(g,x) for g in H); out.append(C); rem-=C
    return tuple(out)
def normal(J,H): return all(conjugate(g,j) in J for g in H for j in J)
def all_subgroups(H):
    subs={frozenset([ID])}; q=deque(subs)
    while q:
        J=q.popleft()
        for g in H:
            if g not in J:
                L=generated_group(list(J)+[g])
                if L not in subs: subs.add(L); q.append(L)
    return tuple(subs)
def summary(name,H):
    Z=center(H); D=derived(H); C=classes(H); subs=all_subgroups(H); norms=[J for J in subs if normal(J,H)]
    print(name)
    print(" order =",len(H))
    print(" element orders =",hist(order(g) for g in H))
    print(" center =",len(Z),"orders",hist(order(g) for g in Z))
    print(" derived =",len(D),"orders",hist(order(g) for g in D))
    print(" abelianization order =",len(H)//len(D))
    print(" conjugacy class sizes =",sorted(map(len,C)))
    print(" subgroup orders =",hist(map(len,subs))," total =",len(subs))
    print(" normal subgroup orders =",hist(map(len,norms))," total =",len(norms))

# word labels for 7 translation directions
W={}
for A in (0,1):
 for B in (0,1):
  for C in (0,1):
   g=ID
   if A:g=compose(p,g)
   if B:g=compose(h,g)
   if C:g=compose(s,g)
   W[g]=f"{A}{B}{C}"
DIR=tuple(g for g in G if g!=ID)

def obj_orbits(H,objs,act):
    rem=set(objs); out=[]
    while rem:
        x=next(iter(rem)); O=frozenset(act(g,x) for g in H); out.append(O); rem-=O
    return sorted(out,key=len)
def report_actions(name,H):
    print(name)
    specs=[
      ("points",X,lambda a,x:a[x]),
      ("directions",DIR,lambda a,d:conjugate(a,d)),
      ("planes",PLANES,lambda a,P:mapped_set(a,P)),
      ("parallel classes",PCS,lambda a,C:frozenset(mapped_set(a,P) for P in C))]
    for label,objs,act in specs:
        O=obj_orbits(H,objs,act)
        st=hist(sum(act(g,x)==x for g in H) for x in objs)
        print(" ",label,"orbit sizes",sorted(map(len,O)),"stabilizers",st)

print("="*92); print("SIM14.5 — AFFINE–SYMPLECTIC COMPATIBILITY CLASSIFICATION"); print("="*92)
print("No external group target; no F4/24-cell/E8/H4; no J/g/U4; no ISP/LCO.")
print("\nA. SANITY")
print("|K|",len(K),"|G|",len(G),"regular",all(len(orbit(G,x))==8 for x in X))
print("V4s",len(V),"planes",len(PLANES),"parallel classes",len(PCS))
print("Omega antisymmetric",np.array_equal(OMEGA.T,-OMEGA),"rank",np.linalg.matrix_rank(OMEGA),"det",round(np.linalg.det(OMEGA)))

print("\nB. DISCOVERY")
print("|Aut affine|",len(AUT))
print("Omega classes",hist(oclass(a) for a in AUT))
print("|N+|",len(PLUS),"|N-|",len(MINUS),"|N +/-|",len(N))

print("\nC. ABSTRACT FINGERPRINTS")
summary("N+",PLUS); print(); summary("N +/-",N)

print("\nD. INDEX-TWO EXTENSION")
chi=lambda a: 1 if a in PLUS else -1
print("N+ normal =",normal(PLUS,N))
print("sign homomorphism =",all(chi(compose(a,b))==chi(a)*chi(b) for a in N for b in N))
print("N- order histogram =",hist(order(a) for a in MINUS))
split=[a for a in MINUS if order(a)==2]
print("order-2 reversing witnesses =",len(split)," split extension =",bool(split))
central=[r for r in MINUS if all(compose(r,g)==compose(g,r) for g in PLUS)]
print("reversers centralizing all N+ =",len(central))
print("reverser conjugation fixed-count histogram =",hist(sum(conjugate(r,g)==g for g in PLUS) for r in MINUS))

print("\nE. ACTIONS")
report_actions("N+",PLUS); report_actions("N +/-",N)

print("\nF. INFORMATION-LOSS ABLATION (AFFINE-RESTRICTED)")
for label,H in [("+Omega",PLUS),("+/-Omega",N),("|S|",AFFABS),("partner s",AFFS)]:
    print(label,"order",len(H),"element orders",hist(order(g) for g in H))

print("\nG. C8 / FROZEN INTERSECTIONS")
SPH=frozenset(a for a in AC8 if conjugate(a,p)==p and conjugate(a,h)==h)
SPHS=frozenset(a for a in SPH if conjugate(a,s)==s)
for label,H in [("Aut(C8)",AC8),("Aut(C8) cap N",AC8&N),("Aut(C8) cap N+",AC8&PLUS),("Stab_C8(p,h)",SPH),("Stab_C8(p,h,s)",SPHS)]:
    print(label,"order",len(H),"Omega",hist(oclass(a) for a in H),"elements",sorted(cnot(a) for a in H))

print("\nH. EXPLICIT N +/- DIRECTION ORBITS")
for i,O in enumerate(obj_orbits(N,DIR,lambda a,d:conjugate(a,d))):
    print(i,sorted(W[d] for d in O))

print("\nI. MACHINE TRUTH PACKET")
print("|Aut AG(3,2)| =",len(AUT))
print("|N+| =",len(PLUS),"|N-| =",len(MINUS),"|N| =",len(N))
print("|Z(N+)| =",len(center(PLUS)),"|Z(N)| =",len(center(N)))
print("|[N+,N+]| =",len(derived(PLUS)),"|[N,N]| =",len(derived(N)))
print("No external identification made.")
print("SIM14.5 COMPLETE — classification sanity checks PASS")

assert len(K)==4 and len(G)==8 and len(V)==7 and len(PLANES)==14 and len(PCS)==7
assert len(AUT)==1344 and len(PLUS)==24 and len(MINUS)==24 and len(N)==48
assert np.array_equal(OMEGA.T,-OMEGA) and np.linalg.matrix_rank(OMEGA)==8
assert all(compose(a,b) in N for a in N for b in N)
assert all(compose(a,b) in PLUS for a in PLUS for b in PLUS)






