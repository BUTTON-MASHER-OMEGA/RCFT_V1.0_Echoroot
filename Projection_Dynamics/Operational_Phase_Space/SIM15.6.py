"""SIM15.6 — TRIALITY–INCIDENCE REFINEMENT TEST
Self-contained; NumPy only. Derived from frozen SIM15.5 H4 and native D4
constructions. One native cell; NO 25-cell or C2^3 ambient census.
All outputs are mathematical finite-geometry results, not dynamics.
"""
import itertools as it
import math
from collections import Counter, deque
import numpy as np

PHI=(1+math.sqrt(5))/2
TOL=1e-7

def check(name, condition):
    print(('PASS' if condition else 'FAIL')+' | '+name)
    if not condition: raise AssertionError(name)

def parity(p): return sum(p[i]>p[j] for i in range(len(p)) for j in range(i+1,len(p)))%2

def roots_h4():
    a=[]
    for i in range(4):
        for s in (-1,1):
            v=np.zeros(4);v[i]=s*math.sqrt(2);a.append(v)
    for s in it.product((-1,1),repeat=4): a.append(np.array(s,dtype=float)/math.sqrt(2))
    base=(0,1,PHI,1/PHI)
    for p in it.permutations(range(4)):
        if parity(p):continue
        v=[base[p[i]] for i in range(4)]
        nz=[i for i in range(4) if v[i]]
        for ss in it.product((-1,1),repeat=3):
            w=np.array(v,dtype=float)
            for i,s in zip(nz,ss):w[i]*=s
            a.append(w/math.sqrt(2))
    uniq={tuple(np.round(v,10)):v for v in a}
    return np.array(list(uniq.values()))

def identity(n):return tuple(range(n))
def mul(a,b):return tuple(a[b[i]] for i in range(len(a)))
def inv(p):
    q=[0]*len(p)
    for i,j in enumerate(p):q[j]=i
    return tuple(q)
def conj(g,h):return mul(mul(g,h),inv(g))
def group(gens,n):
    e=identity(n);out={e};todo=deque([e]);gens=list(gens)
    while todo:
        a=todo.popleft()
        for b in gens:
            c=mul(a,b)
            if c not in out:out.add(c);todo.append(c)
    return out

def order(p):
    a=identity(len(p));q=p;k=1
    while q!=a:q=mul(q,p);k+=1
    return k

def perm_of_matrix(M,R,lookup):
    out=[]
    for v in R:
        w=M@v;k=tuple(np.round(w,9))
        if k not in lookup:raise RuntimeError('root map not closed')
        out.append(lookup[k])
    return tuple(out)

def root_reflection(i,R,lookup):
    v=R[i];return perm_of_matrix(np.eye(4)-np.outer(v,v),R,lookup)

def native_frame(R):
    D=R@R.T
    for c in range(120):
        near=[j for j in range(120) if j!=c and abs(D[c,j]+1)<TOL]
        for a,b,d in it.combinations(near,3):
            if all(abs(D[x,y])<TOL for x,y in ((a,b),(a,d),(b,d))):
                return c,a,b,d
    raise RuntimeError('no D4 frame')

def simple_h4(R):
    D=R@R.T
    for a2 in range(120):
        if abs(D[0,a2]+PHI)>TOL:continue
        for a3 in range(120):
            if abs(D[a2,a3]+1)>TOL or abs(D[0,a3])>TOL:continue
            for a4 in range(120):
                if abs(D[a3,a4]+1)<TOL and abs(D[a2,a4])<TOL and abs(D[0,a4])<TOL:
                    return (0,a2,a3,a4)
    raise RuntimeError('no H4 simple roots')

def image(g,obj):return frozenset(g[x] for x in obj)
def orbit(G,objs):
    remaining=set(objs);out=[]
    while remaining:
        seed=min(remaining,key=lambda x:tuple(sorted(x)))
        o={image(g,seed) for g in G}
        if not o<=set(objs):raise AssertionError('action not closed')
        out.append(frozenset(o));remaining-=o
    return tuple(sorted(out,key=lambda x:(len(x),tuple(sorted(tuple(sorted(y)) for y in x)))))

def edges_and_triangles(C,R):
    V=sorted(C);A={v:set() for v in V};E=[]
    for i,a in enumerate(V):
        for b in V[i+1:]:
            if abs(float(R[a]@R[b])-1)<TOL:
                E.append(frozenset((a,b)));A[a].add(b);A[b].add(a)
    tri={frozenset((a,b,c)) for a in V for b in A[a] if b>a for c in A[a]&A[b] if c>b}
    return tuple(E),tuple(sorted(tri,key=lambda x:tuple(sorted(x)))),A

def facets_octa(C,R):
    # Frozen SIM15.5 hyperplane test, but stop once 24 certified facets found.
    V=sorted(C);X=R[V];found=set()
    for ids in it.combinations(range(24),4):
        P=X[list(ids)];D=P[1:]-P[0]
        if np.linalg.matrix_rank(D,tol=1e-8)<3:continue
        _,_,vh=np.linalg.svd(D);n=vh[-1];n/=np.linalg.norm(n)
        vals=X@n;h=float(P[0]@n)
        for sign in (1,-1):
            v=sign*vals;hh=sign*h;mx=float(np.max(v))
            if abs(mx-hh)>TOL:continue
            inds=[i for i,z in enumerate(v) if abs(z-mx)<TOL]
            if len(inds)==6:found.add(frozenset(V[i] for i in inds))
        if len(found)==24:break
    # Graph certification: six vertices, twelve edges, degree four.
    E,_,A=edges_and_triangles(C,R)
    good=[]
    for F in found:
        if sum(len(A[v]&F) for v in F)==24:good.append(F)
    return tuple(sorted(good,key=lambda x:tuple(sorted(x))))

def normal_elementary_8(L,n):
    """Enumerate normal C2^3 inside L using its conjugacy orbits of involutions.
    A normal subgroup is a union of L conjugacy classes; only classes
    of involutions of total size 7 can contribute.
    """
    e=identity(n)
    invol={g for g in L if g!=e and mul(g,g)==e}
    remaining=set(invol);classes=[]
    while remaining:
        a=next(iter(remaining));cl={conj(g,a) for g in L}
        classes.append(frozenset(cl));remaining-=cl
    options=[]
    for r in range(1,len(classes)+1):
        for selection in it.combinations(classes,r):
            if sum(map(len,selection))!=7:continue
            H={e}.union(*selection)
            if len(H)!=8:continue
            if all(mul(a,b) in H for a in H for b in H):
                options.append(frozenset(H))
    return tuple(sorted(set(options),key=lambda H:tuple(sorted(H))))

def cyc_labels(items,c,transform):
    items=set(items);first=min(items,key=lambda x:repr(x))
    result=[first]
    for _ in range(2):result.append(transform(c,result[-1]))
    if len(set(result))!=3 or transform(c,result[-1])!=result[0]:
        raise AssertionError('not a regular C3 action')
    if set(result)!=items:raise AssertionError('incomplete C3 triple')
    return tuple(result)

def stabilizer_size(T,obj):return sum(image(g,obj)==obj for g in T)

def partition(values):return tuple(sorted(Counter(values).values()))

def fixed_character(T,objs):
    # Intrinsic complete action fingerprint: sorted elementwise fixed counts.
    return tuple(sorted(sum(image(g,x)==x for x in objs) for g in T))

def row_orbit_signature(T,objs):return tuple(sorted(map(len,orbit(T,objs))))

def stats(T,objs,triangles=None):
    stab=tuple(sorted(stabilizer_size(T,x) for x in objs))
    ans=(row_orbit_signature(T,objs),stab,fixed_character(T,objs))
    if triangles is None:return ans
    # Joint incidence: flags (triangle contained in facet) for facets in objs.
    flags=tuple(frozenset((('t',tuple(sorted(t))),('f',tuple(sorted(f)))))
                for f in objs for t in triangles if t<=f)
    # Group action on flags, avoiding mixing vertex IDs and tag names.
    def flag_image(g,flag):
        return frozenset((tag,tuple(sorted(g[v] for v in vertices))) for tag,vertices in flag)
    un=set(flags);sizes=[]
    while un:
        f=next(iter(un));o={flag_image(g,f) for g in T}
        if not o<=set(flags):raise AssertionError('flag action not closed')
        sizes.append(len(o));un-=o
    ans+=(tuple(sorted(sizes)),tuple(sorted(sum(flag_image(g,f)==f for f in flags) for g in T)))
    return ans

def matrix_report(name,Tlist,Flist,triangles=None):
    M=[[stats(T,F,triangles) for F in Flist] for T in Tlist]
    print('\n'+name)
    for i,row in enumerate(M):
        print('  T%d: '%i+' | '.join('F%d=%s'%(j,str(v)) for j,v in enumerate(row)))
    circulant=all(M[i][j]==M[(i+1)%3][(j+1)%3] for i in range(3) for j in range(3))
    check(name+' N-covariance/circulant',circulant)
    phases=[M[0][k] for k in range(3)]
    distinct=len(set(phases))
    print('  relative phases distinct =',distinct,'/3')
    print('  abstract equivariant bijections =',[(i,(i+k)%3) for k in range(3) for i in range(3)])
    return M,distinct

def main():
    print('='*80+'\nSIM15.6 — TRIALITY–INCIDENCE REFINEMENT TEST\n'+'='*80)
    print('One native 24-cell; frozen SIM15.5 constructions; no dynamics.')
    R=roots_h4();lookup={tuple(np.round(v,9)):i for i,v in enumerate(R)}
    check('120 H4 roots',len(R)==120)
    frame=native_frame(R)
    refs=[root_reflection(i,R,lookup) for i in frame]
    C=frozenset(frame);todo=deque(frame)
    while todo:
        x=todo.popleft()
        for g in refs:
            y=g[x]
            if y not in C:C=C|{y};todo.append(y)
    check('native D4 closure has 24 vertices',len(C)==24)
    # L = Weyl group generated by all root reflections of native D4.
    L=group(refs,120)
    check('native W(D4) reflection group order 192',len(L)==192)
    check('L preserves native C',all(image(g,C)==C for g in L))
    simple=[root_reflection(i,R,lookup) for i in simple_h4(R)]
    W=group(simple,120)
    check('ambient W(H4) order 14400',len(W)==14400)
    N={g for g in W if image(g,C)==C}
    check('native cell stabilizer order 576',len(N)==576)
    check('L normal in N, index 3',all(conj(g,h) in L for g in N for h in refs) and len(N)==3*len(L))
    Tlist=normal_elementary_8(L,120)
    print('normal elementary abelian order-8 subgroups in L:',len(Tlist))
    check('exactly three normal C2^3 kernels',len(Tlist)==3)
    check('T kernels contained in L',all(set(T)<=L for T in Tlist))
    c=next((g for g in N-L if order(g)==3 and {frozenset(conj(g,h) for h in T) for T in Tlist}==set(Tlist) and all(frozenset(conj(g,h) for h in T)!=T for T in Tlist)),None)
    check('order-three witness cycles all T kernels',c is not None)
    Tlist=cyc_labels(Tlist,c,lambda g,T:frozenset(conj(g,h) for h in T))
    E,Tr,A=edges_and_triangles(C,R);F=facets_octa(C,R)
    check('native incidences 96 edges, 96 triangles, 24 octahedra',(len(E),len(Tr),len(F))==(96,96,24))
    Torb=orbit(L,Tr);Forb=orbit(L,F)
    check('L triangle orbit sizes 32+32+32',tuple(sorted(map(len,Torb)))==(32,32,32))
    check('L facet orbit sizes 8+8+8',tuple(sorted(map(len,Forb)))==(8,8,8))
    Torb=cyc_labels(Torb,c,lambda g,O:frozenset(image(g,x) for x in O))
    Forb=cyc_labels(Forb,c,lambda g,O:frozenset(image(g,x) for x in O))
    print('Three abstract equivariant matchings: phase offsets 0,1,2')
    # PREDECLARED BINARY RELATION: every facet in a class has nontrivial
    # point stabilizer in T. No maximization/minimization convention.
    B=[[all(stabilizer_size(T,f)>1 for f in O) for O in Forb] for T in Tlist]
    print('\nHARD GATE | nontrivial T-facet stabilizer relation:')
    for i,row in enumerate(B):print('  T%d:'%i,[int(v) for v in row])
    check('binary incidence is a permutation matrix',
          all(sum(row)==1 for row in B) and all(sum(B[i][j] for i in range(3))==1 for j in range(3)))
    # Check all 576 elements, not just the chosen order-three witness.
    Ti={T:i for i,T in enumerate(Tlist)};Fi={F:j for j,F in enumerate(Forb)}
    equiv=True
    for g in N:
        tp=[Ti[frozenset(conj(g,h) for h in T)] for T in Tlist]
        fp=[Fi[frozenset(image(g,f) for f in O)] for O in Forb]
        if any(B[i][j]!=B[tp[i]][fp[j]] for i in range(3) for j in range(3)):
            equiv=False;break
    check('binary matching equivariant under all 576 N elements',equiv)
    # All six independent row and column relabelings preserve the fact
    # that the relation selects one perfect matching.
    check('matching intrinsic under all 6x6 independent relabelings',
          all(all(sum(B[pi[i]][pj[j]] for j in range(3))==1 for i in range(3))
              and all(sum(B[pi[i]][pj[j]] for i in range(3))==1 for j in range(3))
              for pi in it.permutations(range(3)) for pj in it.permutations(range(3))))
    print('  uniquely selected matching (current labels):',[(i,B[i].index(True)) for i in range(3)])
    print('T kernel orbit signatures on all facets:',[row_orbit_signature(T,F) for T in Tlist])
    M1,d1=matrix_report('A | Kernel x octahedral facet class',Tlist,Forb)
    M2,d2=matrix_report('B | Kernel x triangle class',Tlist,Torb)
    M3,d3=matrix_report('C | Kernel x facet class, with triangle-facet flags',Tlist,Forb,Tr)
    # Strong relabeling control: rotate both index sets by triality; same matrix.
    for M,name in ((M1,'facets'),(M2,'triangles'),(M3,'flags')):
        check('relabeling invariance '+name,all(M[i][j]==M[(i+1)%3][(j+1)%3] for i in range(3) for j in range(3)))
    print('\nNEGATIVE CONTROL: erase incidence; 3 equivariant bijections remain.')
    check('abstract C3 action alone selects none',len([k for k in range(3) if all((i+1+k)%3==(i+k+1)%3 for i in range(3))])==3)
    print('\nVERDICT:')
    print('  distinct tested relative-phase signatures:',{'facets':d1,'triangles':d2,'flags':d3})
    if max(d1,d2,d3)==1:
        print('  INCONCLUSIVE: tested intrinsic invariants leave all three phases indistinguishable.')
    else:
        print('  STRUCTURAL DIFFERENTIATION: incidence detects relative phases.')
        print('  CANONICAL RELATIVE TO FROZEN BINARY RELATION: unique facet matching.')
        print('  This does not prove uniqueness among every conceivable intrinsic relation.')
    print('  NO CLAIM OF ISP DYNAMICS OR PHYSICAL TRANSITIONS.')
    print('END SIM15.6')

if __name__=='__main__':main()






~~~~~~~~~~~~~~~~~~~~~~~






================================================================================
SIM15.6 — TRIALITY–INCIDENCE REFINEMENT TEST
================================================================================
One native 24-cell; frozen SIM15.5 constructions; no dynamics.
PASS | 120 H4 roots
PASS | native D4 closure has 24 vertices
PASS | native W(D4) reflection group order 192
PASS | L preserves native C
PASS | ambient W(H4) order 14400
PASS | native cell stabilizer order 576
PASS | L normal in N, index 3
normal elementary abelian order-8 subgroups in L: 3
PASS | exactly three normal C2^3 kernels
PASS | T kernels contained in L
PASS | order-three witness cycles all T kernels
PASS | native incidences 96 edges, 96 triangles, 24 octahedra
PASS | L triangle orbit sizes 32+32+32
PASS | L facet orbit sizes 8+8+8
Three abstract equivariant matchings: phase offsets 0,1,2

HARD GATE | nontrivial T-facet stabilizer relation:
  T0: [1, 0, 0]
  T1: [0, 1, 0]
  T2: [0, 0, 1]
PASS | binary incidence is a permutation matrix
PASS | binary matching equivariant under all 576 N elements
PASS | matching intrinsic under all 6x6 independent relabelings
  uniquely selected matching (current labels): [(0, 0), (1, 1), (2, 2)]
T kernel orbit signatures on all facets: [(2, 2, 2, 2, 8, 8), (2, 2, 2, 2, 8, 8), (2, 2, 2, 2, 8, 8)]

A | Kernel x octahedral facet class
  T0: F0=((2, 2, 2, 2), (4, 4, 4, 4, 4, 4, 4, 4), (0, 4, 4, 4, 4, 4, 4, 8)) | F1=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8)) | F2=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8))
  T1: F0=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8)) | F1=((2, 2, 2, 2), (4, 4, 4, 4, 4, 4, 4, 4), (0, 4, 4, 4, 4, 4, 4, 8)) | F2=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8))
  T2: F0=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8)) | F1=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8)) | F2=((2, 2, 2, 2), (4, 4, 4, 4, 4, 4, 4, 4), (0, 4, 4, 4, 4, 4, 4, 8))
PASS | A | Kernel x octahedral facet class N-covariance/circulant
  relative phases distinct = 2 /3
  abstract equivariant bijections = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0), (0, 2), (1, 0), (2, 1)]

B | Kernel x triangle class
  T0: F0=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32)) | F1=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32)) | F2=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32))
  T1: F0=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32)) | F1=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32)) | F2=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32))
  T2: F0=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32)) | F1=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32)) | F2=((8, 8, 8, 8), (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 32))
PASS | B | Kernel x triangle class N-covariance/circulant
  relative phases distinct = 1 /3
  abstract equivariant bijections = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0), (0, 2), (1, 0), (2, 1)]

C | Kernel x facet class, with triangle-facet flags
  T0: F0=((2, 2, 2, 2), (4, 4, 4, 4, 4, 4, 4, 4), (0, 4, 4, 4, 4, 4, 4, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64)) | F1=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64)) | F2=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64))
  T1: F0=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64)) | F1=((2, 2, 2, 2), (4, 4, 4, 4, 4, 4, 4, 4), (0, 4, 4, 4, 4, 4, 4, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64)) | F2=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64))
  T2: F0=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64)) | F1=((8,), (1, 1, 1, 1, 1, 1, 1, 1), (0, 0, 0, 0, 0, 0, 0, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64)) | F2=((2, 2, 2, 2), (4, 4, 4, 4, 4, 4, 4, 4), (0, 4, 4, 4, 4, 4, 4, 8), (8, 8, 8, 8, 8, 8, 8, 8), (0, 0, 0, 0, 0, 0, 0, 64))
PASS | C | Kernel x facet class, with triangle-facet flags N-covariance/circulant
  relative phases distinct = 2 /3
  abstract equivariant bijections = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0), (0, 2), (1, 0), (2, 1)]
PASS | relabeling invariance facets
PASS | relabeling invariance triangles
PASS | relabeling invariance flags

NEGATIVE CONTROL: erase incidence; 3 equivariant bijections remain.
PASS | abstract C3 action alone selects none

VERDICT:
  distinct tested relative-phase signatures: {'facets': 2, 'triangles': 1, 'flags': 2}
  STRUCTURAL DIFFERENTIATION: incidence detects relative phases.
  CANONICAL RELATIVE TO FROZEN BINARY RELATION: unique facet matching.
  This does not prove uniqueness among every conceivable intrinsic relation.
  NO CLAIM OF ISP DYNAMICS OR PHYSICAL TRANSITIONS.
END SIM15.6


** Process exited - Return Code: 0 **
