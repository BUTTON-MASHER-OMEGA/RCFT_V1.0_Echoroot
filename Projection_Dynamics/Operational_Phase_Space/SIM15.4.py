import itertools, math
from collections import Counter, defaultdict, deque
import numpy as np

# ============================================================
# SIM15.4 — NATIVE 24-CELL REALIZATION OF L_25
#
# FROZEN QUESTION:
# Does the independently selected operational architecture
# coincide, object-for-object and relation-for-relation,
# with native 24-cell structure in H4?
#
# A  frozen SIM15 regression
# B  blind native 24-cell census
# C  geometric stabilizers
# D  literal S_25 == L_25                <-- killer gate
# E  W(H4)-equivariance
# F0 blind pair orbitals
# F  native geometric pair fingerprints
# G  geometry <-> |Li cap Lj| = 8/12
# H  classify I8/I12
# I  reconstruct rank-3 graph
# J  exploratory triality
# K  C3/C5 controls
# ============================================================


# ============================================================
# 0. FROZEN PERMUTATION / GROUP MACHINERY FROM SIM15.3
# ============================================================

def pid(n):
    return tuple(range(n))

def pmul(a,b):
    return tuple(a[b[i]] for i in range(len(a)))

def pinv(a):
    q=[0]*len(a)
    for i,j in enumerate(a):
        q[j]=i
    return tuple(q)

def pconj(g,x):
    return pmul(pmul(g,x),pinv(g))

def porder(p):
    seen=[False]*len(p)
    ans=1
    for i in range(len(p)):
        if seen[i]:
            continue
        j=i
        L=0
        while not seen[j]:
            seen[j]=True
            j=p[j]
            L+=1
        if L:
            ans=math.lcm(ans,L)
    return ans

def pcycles(p,include_fixed=False):
    seen=[False]*len(p)
    out=[]
    for i in range(len(p)):
        if seen[i]:
            continue
        c=[]
        j=i
        while not seen[j]:
            seen[j]=True
            c.append(j)
            j=p[j]
        if include_fixed or len(c)>1:
            out.append(tuple(c))
    return tuple(out)

def cycle_signature(p):
    return dict(sorted(Counter(len(c) for c in pcycles(p,True)).items()))

def generated_group(gens,n=None):
    if not gens:
        return {pid(n)}
    n=len(gens[0])
    e=pid(n)
    G={e}
    q=deque([e])
    while q:
        x=q.popleft()
        for g in gens:
            y=pmul(g,x)
            if y not in G:
                G.add(y)
                q.append(y)
    return G

def subgroup_generated(gens,n):
    return generated_group(list(gens),n)

def greedy_generators(G):
    G=set(G)
    n=len(next(iter(G)))
    e=pid(n)
    gens=[]
    H={e}
    while H!=G:
        x=next(g for g in G if g not in H)
        gens.append(x)
        H=subgroup_generated(gens,n)
    return gens

def commute(a,b):
    return pmul(a,b)==pmul(b,a)

def center(G):
    G=set(G)
    return {
        z for z in G
        if all(pmul(z,g)==pmul(g,z) for g in G)
    }

def centralizer_in(W,H):
    H=set(H)
    return {
        g for g in W
        if all(pmul(g,h)==pmul(h,g) for h in H)
    }

def normalizer_in(W,H):
    H=set(H)
    return {
        g for g in W
        if {pconj(g,h) for h in H}==H
    }

def order_hist(G):
    return dict(sorted(Counter(porder(g) for g in G).items()))

def conjugate_subgroup(g,H):
    return frozenset(pconj(g,h) for h in H)

def conjugacy_orbit_subgroup(W,H):
    return {conjugate_subgroup(g,H) for g in W}


# ============================================================
# 1. FROZEN H4 CONSTRUCTION
# ============================================================

PHI=(1+math.sqrt(5))/2
IPHI=1/PHI
SQRT2=math.sqrt(2)
TOL=1e-7

def permutation_parity(p):
    inv=0
    for i in range(len(p)):
        for j in range(i+1,len(p)):
            if p[i]>p[j]:
                inv+=1
    return inv&1

def unique_vectors(vectors,decimals=10):
    d={}
    for v in vectors:
        d[tuple(np.round(v,decimals))]=np.asarray(v,float)
    return np.array(list(d.values()),float)

def build_h4_roots():
    roots=[]

    for i in range(4):
        for s in (-1.,1.):
            v=np.zeros(4)
            v[i]=2*s
            roots.append(v/SQRT2)

    for signs in itertools.product((-1.,1.),repeat=4):
        roots.append(np.array(signs,float)/SQRT2)

    base=(0.,1.,PHI,IPHI)

    even=[
        p for p in itertools.permutations(range(4))
        if permutation_parity(p)==0
    ]

    for p in even:
        vals=np.array([base[p[i]] for i in range(4)],float)
        nz=[i for i,x in enumerate(vals) if abs(x)>1e-8]

        for signs in itertools.product((-1.,1.),repeat=3):
            v=vals.copy()
            for j,s in zip(nz,signs):
                v[j]*=s
            roots.append(v/SQRT2)

    R=unique_vectors(roots)

    if len(R)!=120:
        raise RuntimeError(f"H4 roots={len(R)}, expected 120")

    if not np.allclose(np.sum(R*R,axis=1),2,atol=1e-8):
        raise RuntimeError("H4 root norms failed")

    return R

def vec_key(v):
    return tuple(np.round(v,9))

def build_root_lookup(R):
    return {vec_key(v):i for i,v in enumerate(R)}

def reflection_matrix(alpha):
    return np.eye(4)-np.outer(alpha,alpha)

def linear_map_to_perm(M,R,lookup):
    p=[]
    for v in R:
        w=M@v
        k=vec_key(w)
        if k in lookup:
            p.append(lookup[k])
        else:
            d=np.linalg.norm(R-w[None,:],axis=1)
            j=int(np.argmin(d))
            if d[j]>1e-6:
                raise RuntimeError("map does not permute roots")
            p.append(j)
    return tuple(p)

def find_simple_h4_roots(R):
    dots=R@R.T

    def inds(i,target):
        return [
            j for j in range(len(R))
            if j!=i and abs(dots[i,j]-target)<1e-7
        ]

    a1=0

    for a2 in inds(a1,-PHI):
        for a3 in [
            x for x in inds(a2,-1)
            if abs(dots[a1,x])<1e-7
        ]:
            cand=[
                a4 for a4 in inds(a3,-1)
                if abs(dots[a2,a4])<1e-7
                and abs(dots[a1,a4])<1e-7
            ]
            if cand:
                return [a1,a2,a3,cand[0]]

    raise RuntimeError("H4 simple roots not found")

def build_W_H4(R):
    lookup=build_root_lookup(R)
    idx=find_simple_h4_roots(R)

    gens=[
        linear_map_to_perm(reflection_matrix(R[i]),R,lookup)
        for i in idx
    ]

    W=generated_group(gens)

    if len(W)!=14400:
        raise RuntimeError(f"|W(H4)|={len(W)}")

    return W,gens,idx


# ============================================================
# 2. FROZEN SUCCESSFUL C2^3 FINGERPRINT
# ============================================================

def enumerate_C2_3_subgroups(W):
    Wlist=list(W)
    e=pid(len(Wlist[0]))

    invol=[
        g for g in Wlist
        if g!=e and porder(g)==2
    ]

    print("involutions in W(H4):",len(invol))

    groups=set()

    for ia,a in enumerate(invol):
        for ib in range(ia+1,len(invol)):
            b=invol[ib]

            if not commute(a,b):
                continue

            H4=subgroup_generated([a,b],len(a))

            if len(H4)!=4:
                continue

            for c in invol:
                if c in H4:
                    continue

                if not commute(a,c) or not commute(b,c):
                    continue

                H8=subgroup_generated([a,b,c],len(a))

                if (
                    len(H8)==8
                    and all(porder(x) in (1,2) for x in H8)
                ):
                    groups.add(frozenset(H8))

    return [set(H) for H in groups]

def subgroup_conjugacy_classes(W,subs):
    unseen={frozenset(H) for H in subs}
    classes=[]

    while unseen:
        H=next(iter(unseen))
        orb={conjugate_subgroup(g,H) for g in W}
        cls=orb & unseen
        classes.append(cls)
        unseen-=orb

    return classes

F2V=[
    (0,0,0),(0,0,1),(0,1,0),(0,1,1),
    (1,0,0),(1,0,1),(1,1,0),(1,1,1)
]

I3=((1,0,0),(0,1,0),(0,0,1))

def choose_T_basis(T):
    e=pid(len(next(iter(T))))
    nz=[x for x in T if x!=e]

    for a,b,c in itertools.combinations(nz,3):
        if subgroup_generated([a,b,c],len(a))==set(T):
            return a,b,c

    raise RuntimeError("T basis failed")

def T_coordinate_map(T):
    e=pid(len(next(iter(T))))
    a,b,c=choose_T_basis(T)

    c2e={}
    e2c={}

    for v in F2V:
        x=e
        if v[0]: x=pmul(a,x)
        if v[1]: x=pmul(b,x)
        if v[2]: x=pmul(c,x)

        c2e[v]=x
        e2c[x]=v

    return e2c,c2e

def mat_apply(M,v):
    return tuple(
        sum(M[i][j]*v[j] for j in range(3))%2
        for i in range(3)
    )

def mat_mul(A,B):
    return tuple(
        tuple(
            sum(A[i][k]*B[k][j] for k in range(3))%2
            for j in range(3)
        )
        for i in range(3)
    )

def mat_order(M):
    x=I3
    for k in range(1,100):
        x=mat_mul(M,x)
        if x==I3:
            return k
    raise RuntimeError("matrix order failure")

def mat_from_action_on_T(g,e2c,c2e):
    basis=[(1,0,0),(0,1,0),(0,0,1)]
    cols=[]

    for v in basis:
        x=c2e[v]
        y=pconj(g,x)
        cols.append(e2c[y])

    return tuple(
        tuple(cols[j][i] for j in range(3))
        for i in range(3)
    )

TARGET_GL_HIST={1:1,2:9,3:8,4:6}

def full_fingerprint(W,T):
    N=normalizer_in(W,T)
    C=centralizer_in(W,T)

    e2c,c2e=T_coordinate_map(T)

    image={
        mat_from_action_on_T(g,e2c,c2e)
        for g in N
    }

    fixed=[
        v for v in F2V
        if all(mat_apply(M,v)==v for M in image)
    ]

    gh=dict(sorted(Counter(mat_order(M) for M in image).items()))

    zN=center(N)
    e=pid(len(next(iter(T))))

    z=None
    if len(zN)==2:
        z=next(x for x in zN if x!=e)

    nf=[v for v in fixed if v!=(0,0,0)]

    center_fixed=(
        z is not None
        and len(nf)==1
        and z==c2e[nf[0]]
    )

    checks={
        "T8":len(T)==8,
        "N192":len(N)==192,
        "centralizer_T":C==set(T),
        "GL24":len(image)==24,
        "GL_S4_hist":gh==TARGET_GL_HIST,
        "one_nonzero_fixed":len(nf)==1,
        "center2":len(zN)==2,
        "center_fixed":center_fixed,
    }

    return {
        "full":all(checks.values()),
        "N":N,
        "C":C,
        "z":z,
        "checks":checks,
    }

def find_successful_structure(W):
    Tsubs=enumerate_C2_3_subgroups(W)

    print("C2^3 subgroups:",len(Tsubs))

    classes=subgroup_conjugacy_classes(W,Tsubs)

    print("C2^3 conjugacy classes:",len(classes))
    print("class sizes:",sorted(len(C) for C in classes))

    good=[]

    for ci,C in enumerate(classes):
        T=set(next(iter(C)))
        L=normalizer_in(W,T)

        print(
            f"class {ci}: orbit={len(C)} normalizer={len(L)}"
        )

        if len(L)!=192:
            continue

        fp=full_fingerprint(W,T)

        print(
            "  fingerprint:",
            fp["checks"],
            "FULL=",
            fp["full"]
        )

        if fp["full"]:
            good.append((ci,C,T,fp))

    if len(good)!=1:
        raise RuntimeError(
            f"Expected one successful class, found {len(good)}"
        )

    ci,C,T,fp=good[0]

    L=fp["N"]
    N=normalizer_in(W,L)

    return {
        "class_id":ci,
        "successful_class":C,
        "T":T,
        "L":L,
        "N":N,
        "z":fp["z"],
    }

def find_C3_complement(N,L):
    e=pid(len(next(iter(N))))
    Lgens=greedy_generators(L)
    out=[]

    for c in N:
        if c in L or porder(c)!=3:
            continue

        C3=subgroup_generated([c],len(c))

        if C3 & set(L)!={e}:
            continue

        if subgroup_generated(Lgens+[c],len(c))==set(N):
            out.append(c)

    return out


# ============================================================
# 3. SIM15.4 UTILITIES
# ============================================================

def header(label,title):
    print("\n"+label+") "+title)
    print("-"*92)

def frozen_family_with_first(S,first):
    first=frozenset(first)
    rest=sorted(
        [H for H in S if H!=first],
        key=lambda H:tuple(sorted(H))
    )
    return [first]+rest

def subset_image(g,C):
    return frozenset(g[v] for v in C)

def action_on_subset_family(g,F,index):
    return tuple(index[subset_image(g,C)] for C in F)

def intersection_size(A,B):
    return len(set(A)&set(B))

def point_orbits_group(G,C):
    C=set(C)
    unseen=set(C)
    out=[]

    while unseen:
        x=min(unseen)
        O={g[x] for g in G} & C
        out.append(tuple(sorted(O)))
        unseen-=O

    return tuple(sorted(out,key=lambda x:(len(x),x)))


# ============================================================
# 4. BLIND NATIVE 24-CELL CONSTRUCTION
# ============================================================

def root_reflection_perm(i,R,lookup):
    a=R[i]
    p=[]

    for v in R:
        w=v-np.dot(v,a)*a
        k=vec_key(w)

        if k in lookup:
            p.append(lookup[k])
        else:
            d=np.linalg.norm(R-w[None,:],axis=1)
            j=int(np.argmin(d))

            if d[j]>1e-6:
                raise RuntimeError("root reflection failure")

            p.append(j)

    return tuple(p)

def d4_frames(R):
    D=R@R.T
    n=len(R)

    minus1=[
        [
            j for j in range(n)
            if j!=i and abs(D[i,j]+1)<TOL
        ]
        for i in range(n)
    ]

    for c in range(n):
        for a,b,d in itertools.combinations(minus1[c],3):
            if (
                abs(D[a,b])<TOL
                and abs(D[a,d])<TOL
                and abs(D[b,d])<TOL
            ):
                yield c,a,b,d

def close_under_reflections(seed,refl):
    C=set(seed)
    q=deque(seed)

    while q:
        x=q.popleft()

        for r in refl:
            y=r[x]

            if y not in C:
                C.add(y)
                q.append(y)

    return frozenset(C)

def enumerate_native_24cells(R):
    lookup=build_root_lookup(R)
    cache={}
    cells=set()
    hist=Counter()
    nf=0

    for frame in d4_frames(R):
        nf+=1
        rr=[]

        for i in frame:
            if i not in cache:
                cache[i]=root_reflection_perm(i,R,lookup)

            rr.append(cache[i])

        C=close_under_reflections(frame,rr)
        hist[len(C)]+=1

        if len(C)==24:
            cells.add(C)

    return (
        tuple(sorted(cells,key=lambda C:tuple(sorted(C)))),
        nf,
        dict(sorted(hist.items()))
    )


# ============================================================
# 5. 24-CELL CERTIFICATION
# ============================================================

def induced_edges(C,R,target=1.0):
    V=sorted(C)
    E=[]
    A={v:set() for v in V}

    for ia,a in enumerate(V):
        for b in V[ia+1:]:
            if abs(float(np.dot(R[a],R[b]))-target)<TOL:
                E.append((a,b))
                A[a].add(b)
                A[b].add(a)

    return tuple(E),A

def triangles(A):
    T=set()

    for a in sorted(A):
        for b in [x for x in A[a] if x>a]:
            for c in A[a]&A[b]:
                if c>b:
                    T.add((a,b,c))

    return tuple(sorted(T))

def octahedral_facets(C,R):
    V=sorted(C)
    X=R[V]
    facets=set()

    for ids in itertools.combinations(range(24),4):
        P=X[list(ids)]
        D=P[1:]-P[0]

        if np.linalg.matrix_rank(D,tol=1e-8)<3:
            continue

        _,_,vh=np.linalg.svd(D)
        n=vh[-1]

        if np.linalg.norm(n)<1e-10:
            continue

        n=n/np.linalg.norm(n)
        vals=X@n
        h=float(P[0]@n)

        for s in (1.,-1.):
            vv=s*vals
            hh=s*h
            mx=float(np.max(vv))

            if abs(mx-hh)>1e-7:
                continue

            inds=[
                k for k,x in enumerate(vv)
                if abs(x-mx)<1e-7
            ]

            if len(inds)==6:
                facets.add(
                    frozenset(V[k] for k in inds)
                )

    good=set()

    for F in facets:
        E,A=induced_edges(F,R)

        if (
            len(E)==12
            and Counter(len(A[v]) for v in F)=={4:6}
        ):
            good.add(F)

    return tuple(sorted(good,key=lambda F:tuple(sorted(F))))

def ip_spectrum(C,R):
    V=sorted(C)
    H=Counter()

    for ia,a in enumerate(V):
        for b in V[ia+1:]:
            x=float(np.dot(R[a],R[b]))

            for t in (-2.,-1.,0.,1.,2.):
                if abs(x-t)<1e-7:
                    x=t
                    break

            H[round(x,8)]+=1

    return dict(sorted(H.items()))

def certify_24cell(C,R):
    V=sorted(C)
    X=R[V]

    E,A=induced_edges(C,R)
    T=triangles(A)
    F=octahedral_facets(C,R)

    lookup={vec_key(R[v]) for v in V}
    antipodal=all(vec_key(-R[v]) in lookup for v in V)

    cert={
        "vertices":len(V),
        "rank":int(np.linalg.matrix_rank(X,tol=1e-8)),
        "antipodal":antipodal,
        "degree_hist":dict(sorted(Counter(len(A[v]) for v in V).items())),
        "edges":len(E),
        "triangles":len(T),
        "octahedral_facets":len(F),
        "ip_spectrum":ip_spectrum(C,R),
    }

    cert["passes"]=(
        cert["vertices"]==24
        and cert["rank"]==4
        and cert["antipodal"]
        and cert["degree_hist"]=={8:24}
        and cert["edges"]==96
        and cert["triangles"]==96
        and cert["octahedral_facets"]==24
    )

    return (
        cert,
        frozenset(tuple(sorted(e)) for e in E),
        frozenset(frozenset(t) for t in T),
        frozenset(F)
    )


# ============================================================
# 6. STABILIZERS / ACTIONS / PAIR ORBITS
# ============================================================

def setwise_stabilizer(W,C):
    return frozenset(
        g for g in W
        if subset_image(g,C)==C
    )

def build_cell_action(W,F):
    index={C:i for i,C in enumerate(F)}
    action={}

    for g in W:
        p=[]

        for C in F:
            D=subset_image(g,C)

            if D not in index:
                return index,None,False,(g,C,D)

            p.append(index[D])

        action[g]=tuple(p)

    return index,action,True,None

def action_kernel(action):
    n=len(next(iter(action.values())))
    e=pid(n)

    return {g for g,p in action.items() if p==e}

def pair_orbits(action,n):
    unseen={
        (i,j)
        for i in range(n)
        for j in range(i+1,n)
    }

    image=set(action.values())
    out=[]

    while unseen:
        a,b=min(unseen)

        O={
            tuple(sorted((p[a],p[b])))
            for p in image
        }

        out.append(frozenset(O))
        unseen-=O

    return tuple(sorted(out,key=lambda O:(len(O),tuple(sorted(O)))))


# ============================================================
# 7. NATIVE 600-CELL GEOMETRY
# ============================================================

def build_600_graph(R):
    A=[set() for _ in R]

    for i in range(len(R)):
        for j in range(i+1,len(R)):
            if abs(float(R[i]@R[j])-PHI)<1e-7:
                A[i].add(j)
                A[j].add(i)

    return A

def graph_distances(A):
    D={}

    for s in range(len(A)):
        dist={s:0}
        q=deque([s])

        while q:
            x=q.popleft()

            for y in A[x]:
                if y not in dist:
                    dist[y]=dist[x]+1
                    q.append(y)

        for t in range(len(A)):
            D[s,t]=dist[t]

    return D

def pair_fingerprint(C,D,R,Dist,ES,TS,FS):
    cross_dist=Counter()
    cross_ip=Counter()

    vals=[
        -2.,-PHI,-1.,-IPHI,0.,
        IPHI,1.,PHI,2.
    ]

    for a in C:
        for b in D:
            cross_dist[Dist[a,b]]+=1

            x=float(R[a]@R[b])
            z=None

            for t in vals:
                if abs(x-t)<1e-7:
                    z=round(t,8)
                    break

            if z is None:
                z=round(x,8)

            cross_ip[z]+=1

    return (
        len(C&D),
        len(ES[C]&ES[D]),
        len(TS[C]&TS[D]),
        len(FS[C]&FS[D]),
        tuple(sorted(cross_dist.items())),
        tuple(sorted(cross_ip.items()))
    )

def pretty_fp(fp):
    return {
        "shared_vertices":fp[0],
        "shared_edges":fp[1],
        "shared_triangles":fp[2],
        "shared_facets":fp[3],
        "cross_distance_hist":dict(fp[4]),
        "cross_ip_hist":dict(fp[5]),
    }


# ============================================================
# 8. I8 / I12 INVARIANTS
# ============================================================

def commutator(a,b):
    return pmul(pmul(pmul(a,b),pinv(a)),pinv(b))

def derived_subgroup(G):
    G=set(G)

    comms={
        commutator(a,b)
        for a in G
        for b in G
    }

    return frozenset(
        subgroup_generated(
            list(comms),
            len(next(iter(G)))
        )
    )

def subgroup_invariants(I,W):
    I=frozenset(I)
    Z=frozenset(center(I))
    D=derived_subgroup(I)

    return {
        "order":len(I),
        "order_hist":order_hist(I),
        "center_order":len(Z),
        "center_hist":order_hist(Z),
        "derived_order":len(D),
        "derived_hist":order_hist(D),
        "abelianization_order":len(I)//len(D),
        "normalizer_order":len(normalizer_in(W,I)),
        "centralizer_order":len(centralizer_in(W,I)),
    }

def group_label(inv):
    n=inv["order"]
    h=inv["order_hist"]
    z=inv["center_order"]
    d=inv["derived_order"]

    if n==8:
        if h=={1:1,2:7}:
            return "C2^3"

        if h=={1:1,2:3,4:4} and z==2 and d==2:
            return "D8"

        if h=={1:1,2:1,4:6} and z==2 and d==2:
            return "Q8"

        if h=={1:1,2:1,4:2,8:4}:
            return "C8"

    if n==12:
        if h=={1:1,2:3,3:8}:
            return "A4"

        if h=={1:1,2:7,3:2,6:2}:
            return "D12"

    return "unresolved"


# ============================================================
# 9. GRAPH SIGNATURE
# ============================================================

def graph_from_pairs(n,O):
    A=np.zeros((n,n),int)

    for i,j in O:
        A[i,j]=A[j,i]=1

    return A

def graph_signature(A):
    n=len(A)

    deg=Counter(int(x) for x in A.sum(axis=1))

    adj=Counter()
    non=Counter()

    for i in range(n):
        for j in range(i+1,n):
            c=int(A[i]@A[j])

            if A[i,j]:
                adj[c]+=1
            else:
                non[c]+=1

    vals=np.round(np.linalg.eigvalsh(A.astype(float)),9)

    seen={0}
    q=deque([0])

    while q:
        x=q.popleft()

        for y in np.flatnonzero(A[x]):
            y=int(y)

            if y not in seen:
                seen.add(y)
                q.append(y)

    return {
        "v":n,
        "edges":int(A.sum()//2),
        "degree_hist":dict(sorted(deg.items())),
        "connected":len(seen)==n,
        "adjacent_common_neighbor_hist":dict(sorted(adj.items())),
        "nonadjacent_common_neighbor_hist":dict(sorted(non.items())),
        "spectrum":[
            (float(x),int(c))
            for x,c in sorted(Counter(vals).items())
        ],
    }


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print("="*92)
    print("SIM15.4 — NATIVE 24-CELL REALIZATION OF THE L_25 ARCHITECTURE")
    print("="*92)

    print("""
FROZEN QUESTION:
Does the independently selected operational architecture coincide,
object-for-object and relation-for-relation, with native 24-cell
structure in H4?
""")

    # --------------------------------------------------------
    # A
    # --------------------------------------------------------

    header("A","FROZEN SIM15 REGRESSION")

    R=build_h4_roots()
    W,Wgens,simple_idx=build_W_H4(R)

    print("H4 roots =",len(R))
    print("|W(H4)| =",len(W))
    print("simple-root indices =",simple_idx)

    S=find_successful_structure(W)

    T0=S["T"]
    L0=S["L"]
    N0=S["N"]
    z=S["z"]

    Tfamily=frozen_family_with_first(
        set(S["successful_class"]),
        T0
    )

    Lfamily=frozen_family_with_first(
        conjugacy_orbit_subgroup(W,L0),
        L0
    )

    print("|T0| =",len(T0))
    print("|L0| =",len(L0))
    print("|N0| =",len(N0))
    print("|T75| =",len(Tfamily))
    print("|L25| =",len(Lfamily))

    GATE_A=(
        len(R)==120
        and len(W)==14400
        and len(T0)==8
        and len(L0)==192
        and len(N0)==576
        and len(Tfamily)==75
        and len(Lfamily)==25
    )

    print("GATE A =",GATE_A)

    # --------------------------------------------------------
    # B
    # --------------------------------------------------------

    header("B","BLIND NATIVE 24-CELL CENSUS")

    CRAW,nframes,closure_hist=enumerate_native_24cells(R)

    print("D4 frame count =",nframes)
    print("closure-size histogram =",closure_hist)
    print("distinct 24-root closures =",len(CRAW))

    Cfamily=[]
    ES={}
    TS={}
    FS={}
    cert_hist=Counter()

    for C in CRAW:
        cert,E,T,F=certify_24cell(C,R)

        key=(
            cert["vertices"],
            cert["rank"],
            cert["antipodal"],
            tuple(cert["degree_hist"].items()),
            cert["edges"],
            cert["triangles"],
            cert["octahedral_facets"],
            tuple(cert["ip_spectrum"].items()),
            cert["passes"]
        )

        cert_hist[key]+=1

        if cert["passes"]:
            Cfamily.append(C)
            ES[C]=E
            TS[C]=T
            FS[C]=F

    Cfamily=tuple(sorted(Cfamily,key=lambda C:tuple(sorted(C))))

    print("certificate types =",len(cert_hist))

    for key,n in cert_hist.items():
        print(" count =",n)
        print(" certificate =",{
            "vertices":key[0],
            "rank":key[1],
            "antipodal":key[2],
            "degree_hist":dict(key[3]),
            "edges":key[4],
            "triangles":key[5],
            "octahedral_facets":key[6],
            "ip_spectrum":dict(key[7]),
            "passes":key[8],
        })

    print("certified native 24-cells =",len(Cfamily))

    GATE_B=(
        len(CRAW)==25
        and len(Cfamily)==25
    )

    print("GATE B =",GATE_B)

    # --------------------------------------------------------
    # C
    # --------------------------------------------------------

    header("C","GEOMETRIC 24-CELL STABILIZERS")

    Cstabs=tuple(
        setwise_stabilizer(W,C)
        for C in Cfamily
    )

    stab_orders=dict(sorted(Counter(len(H) for H in Cstabs).items()))

    print("stabilizer-order histogram =",stab_orders)

    histtypes=Counter(
        tuple(order_hist(H).items())
        for H in Cstabs
    )

    print("stabilizer order-histogram types =",len(histtypes))

    for h,n in histtypes.items():
        print(" count",n,"hist =",dict(h))

    GATE_C=(
        len(Cstabs)==25
        and stab_orders=={192:25}
        and all(order_hist(H)==order_hist(L0) for H in Cstabs)
    )

    print("GATE C =",GATE_C)

    # --------------------------------------------------------
    # D — KILLER GATE
    # --------------------------------------------------------

    header("D","LITERAL STABILIZER-FAMILY IDENTITY")

    Sset={frozenset(H) for H in Cstabs}
    Lset={frozenset(H) for H in Lfamily}

    print("|S| =",len(Sset))
    print("|L| =",len(Lset))
    print("|S cap L| =",len(Sset&Lset))
    print("|S \\ L| =",len(Sset-Lset))
    print("|L \\ S| =",len(Lset-Sset))

    GATE_D=(Sset==Lset)

    print("GATE D — LITERAL S_25 == L_25 =",GATE_D)

    C_to_L={}
    L_to_C={}

    if GATE_D:
        LI={H:i for i,H in enumerate(Lfamily)}

        for ci,H in enumerate(Cstabs):
            li=LI[frozenset(H)]
            C_to_L[ci]=li
            L_to_C[li]=ci

        print("bijection size =",len(C_to_L))

    # --------------------------------------------------------
    # E
    # --------------------------------------------------------

    header("E","W(H4)-EQUIVARIANCE")

    Cindex,actionC,closed,failure=build_cell_action(W,Cfamily)

    print("C24 family W-closed =",closed)

    if not closed:
        print("failure witness =",failure)

    equiv=False

    if GATE_D and closed:
        LI={H:i for i,H in enumerate(Lfamily)}
        equiv=True

        for g in W:
            p=actionC[g]

            for ci in range(25):
                li=C_to_L[ci]

                li2=LI[
                    conjugate_subgroup(g,Lfamily[li])
                ]

                ci2=p[ci]

                if C_to_L[ci2]!=li2:
                    equiv=False
                    print("equivariance failure =",ci,li,ci2,li2)
                    break

            if not equiv:
                break

    GATE_E=GATE_D and closed and equiv

    print("GATE E =",GATE_E)

    if actionC:
        image=set(actionC.values())
        kernel=action_kernel(actionC)

        print("|C24 action image| =",len(image))
        print("|C24 action kernel| =",len(kernel))
        print("z in kernel =",z in kernel)

    # --------------------------------------------------------
    # F0
    # --------------------------------------------------------

    header("F0","BLIND W(H4) PAIR ORBITALS ON C24")

    PORB=pair_orbits(actionC,len(Cfamily)) if actionC else ()

    print("pair orbital count =",len(PORB))
    print("pair orbital sizes =",[len(O) for O in PORB])

    GATE_F0=(
        len(Cfamily)==25
        and len(PORB)==2
        and sorted(len(O) for O in PORB)==[100,200]
    )

    print("GATE F0 =",GATE_F0)

    # --------------------------------------------------------
    # F
    # --------------------------------------------------------

    header("F","NATIVE GEOMETRIC PAIR FINGERPRINTS")

    A600=build_600_graph(R)

    print(
        "600-cell degree histogram =",
        dict(sorted(Counter(len(x) for x in A600).items()))
    )

    print(
        "600-cell edges =",
        sum(len(x) for x in A600)//2
    )

    Dist=graph_distances(A600)

    FP={}
    global_hist=Counter()
    orbital_hist=[]

    for oi,O in enumerate(PORB):
        H=Counter()

        for i,j in O:
            fp=pair_fingerprint(
                Cfamily[i],Cfamily[j],
                R,Dist,ES,TS,FS
            )

            FP[i,j]=fp
            H[fp]+=1
            global_hist[fp]+=1

        orbital_hist.append(H)

        print(
            f"orbital {oi}: size={len(O)}, "
            f"geometric fingerprint types={len(H)}"
        )

        for fp,n in H.items():
            print(" count",n,"fingerprint =",pretty_fp(fp))

    print("global fingerprint types =",len(global_hist))

    GATE_F=(
        GATE_F0
        and len(global_hist)==2
        and sorted(global_hist.values())==[100,200]
        and all(len(H)==1 for H in orbital_hist)
    )

    print("GATE F =",GATE_F)

    # --------------------------------------------------------
    # G
    # --------------------------------------------------------

    header("G","GEOMETRY <-> L-INTERSECTION 8/12")

    orb_lint=defaultdict(Counter)
    fp_lint=defaultdict(Counter)
    records=[]

    if GATE_D:
        for oi,O in enumerate(PORB):
            for i,j in O:
                Li=Lfamily[C_to_L[i]]
                Lj=Lfamily[C_to_L[j]]

                k=intersection_size(Li,Lj)
                fp=FP[i,j]

                orb_lint[oi][k]+=1
                fp_lint[fp][k]+=1

                records.append((i,j,oi,fp,k))

        for oi,H in sorted(orb_lint.items()):
            print(
                "orbital",
                oi,
                "-> intersection histogram =",
                dict(sorted(H.items()))
            )

        print("fingerprint -> intersection:")

        for fp,H in fp_lint.items():
            print(
                pretty_fp(fp),
                "->",
                dict(sorted(H.items()))
            )

    observed=sorted({x[4] for x in records})

    print("observed intersection orders =",observed)

    GATE_G=(
        GATE_D
        and GATE_F0
        and observed==[8,12]
        and all(len(H)==1 for H in orb_lint.values())
        and sorted(next(iter(H)) for H in orb_lint.values())==[8,12]
    )

    print("GATE G =",GATE_G)

    # --------------------------------------------------------
    # H
    # --------------------------------------------------------

    header("H","CLASSIFY I8 AND I12")

    invtypes=defaultdict(Counter)
    labels=defaultdict(Counter)

    for i,j,oi,fp,k in records:
        I=frozenset(
            set(Lfamily[C_to_L[i]])
            &
            set(Lfamily[C_to_L[j]])
        )

        inv=subgroup_invariants(I,W)

        key=(
            tuple(inv["order_hist"].items()),
            inv["center_order"],
            tuple(inv["center_hist"].items()),
            inv["derived_order"],
            tuple(inv["derived_hist"].items()),
            inv["abelianization_order"],
            inv["normalizer_order"],
            inv["centralizer_order"],
        )

        invtypes[k][key]+=1
        labels[k][group_label(inv)]+=1

    for k in sorted(invtypes):
        print(f"|I|={k}: invariant types =",len(invtypes[k]))

        for key,n in invtypes[k].items():
            print(" count =",n)
            print("  order_hist =",dict(key[0]))
            print("  center_order =",key[1])
            print("  center_hist =",dict(key[2]))
            print("  derived_order =",key[3])
            print("  derived_hist =",dict(key[4]))
            print("  abelianization_order =",key[5])
            print("  normalizer_order =",key[6])
            print("  centralizer_order =",key[7])

        print(" labels =",dict(labels[k]))

    GATE_H=(
        set(invtypes)=={8,12}
        and all(len(invtypes[k])==1 for k in (8,12))
        and all(len(labels[k])==1 for k in (8,12))
    )

    print("GATE H =",GATE_H)

    # --------------------------------------------------------
    # I
    # --------------------------------------------------------

    header("I","GEOMETRY-ONLY RANK-3 GRAPH")

    graphdata=None
    GATE_I=False

    if GATE_F0:
        small=min(PORB,key=len)

        G=graph_from_pairs(25,small)
        graphdata=graph_signature(G)

        for k,v in graphdata.items():
            print(f"{k:38s}= {v}")

        GATE_I=(
            graphdata["v"]==25
            and graphdata["edges"]==100
            and graphdata["degree_hist"]=={8:25}
            and graphdata["connected"]
            and set(
                graphdata["adjacent_common_neighbor_hist"]
            )=={3}
            and set(
                graphdata["nonadjacent_common_neighbor_hist"]
            )=={2}
        )

    print(
        "GATE I — geometry reconstructs srg(25,8,3,2) =",
        GATE_I
    )

    # --------------------------------------------------------
    # J — exploratory triality
    # --------------------------------------------------------

    header("J","EXPLORATORY TRIALITY ON EACH 24-CELL")

    if GATE_D:
        fibers=[]

        for L in Lfamily:
            fibers.append(
                tuple(
                    ti
                    for ti,T in enumerate(Tfamily)
                    if set(T)<=set(L)
                )
            )

        print(
            "L fiber-size histogram =",
            dict(sorted(Counter(len(F) for F in fibers).items()))
        )

        triple_sigs=Counter()

        for ci,C in enumerate(Cfamily):
            li=C_to_L[ci]

            local=[]

            for ti in fibers[li]:
                O=point_orbits_group(Tfamily[ti],C)

                local.append(
                    tuple(sorted(len(x) for x in O))
                )

            triple_sigs[
                tuple(sorted(local))
            ]+=1

        print(
            "triality restricted-action signature types =",
            len(triple_sigs)
        )

        for sig,n in triple_sigs.items():
            print(" count",n,"signature =",sig)

    # --------------------------------------------------------
    # K — C3/C5 controls
    # --------------------------------------------------------

    header("K","C3 / C5 NATIVE 24-CELL CONTROLS")

    C5_ok=False
    C3_ok=False

    if GATE_D and closed:

        order5=[g for g in W if porder(g)==5]

        print("order-5 elements =",len(order5))

        if order5:
            f=order5[0]
            pf=action_on_subset_family(f,Cfamily,Cindex)

            print(
                "representative C5 action on C24 =",
                cycle_signature(pf)
            )

            C5_ok=cycle_signature(pf)=={5:5}

        cands=find_C3_complement(N0,L0)

        print("C3 complement witnesses =",len(cands))

        if cands:
            c=cands[0]
            pc=action_on_subset_family(c,Cfamily,Cindex)

            print(
                "representative C3 action on C24 =",
                cycle_signature(pc)
            )

            c0=next(
                ci for ci,li in C_to_L.items()
                if li==0
            )

            C3_ok=(pc[c0]==c0)

            print("C3 fixes cell corresponding to L0 =",C3_ok)

    # --------------------------------------------------------
    # HARD GATES
    # --------------------------------------------------------

    print("\n"+"="*92)
    print("SIM15.4 HARD GATES")
    print("="*92)

    gates={
        "A frozen SIM15 regression":GATE_A,
        "B blind native 24-cell census":GATE_B,
        "C W(D4)-type geometric stabilizers":GATE_C,
        "D LITERAL S_25 == L_25":GATE_D,
        "E W(H4)-equivariant identification":GATE_E,
        "F0 blind pair orbitals 100+200":GATE_F0,
        "F two native geometric pair types":GATE_F,
        "G geometry <-> 8/12":GATE_G,
        "H uniform I8/I12 types":GATE_H,
        "I native srg(25,8,3,2)":GATE_I,
    }

    for k,v in gates.items():
        print(f"{k:48s}: {v}")

    # --------------------------------------------------------
    # MACHINE TRUTH PACKET
    # --------------------------------------------------------

    print("\n"+"="*92)
    print("MACHINE TRUTH PACKET")
    print("="*92)

    truth={
        "H4_roots":len(R),
        "H4_order":len(W),
        "successful_T_count":len(Tfamily),
        "compatible_L_count":len(Lfamily),
        "native_D4_frame_count":nframes,
        "native_24root_closure_count":len(CRAW),
        "certified_24cell_count":len(Cfamily),
        "24cell_stabilizer_order_hist":stab_orders,
        "literal_stabilizer_matches":len(Sset&Lset),
        "literal_family_equality":GATE_D,
        "identification_equivariant":GATE_E,
        "C24_pair_orbit_sizes":[len(O) for O in PORB],
        "geometric_pair_fingerprint_types":len(global_hist),
        "L_intersection_values":observed,
        "I8_labels":dict(labels.get(8,{})),
        "I12_labels":dict(labels.get(12,{})),
        "geometry_graph_signature":graphdata,
        "C5_on_C24_is_5x5cycles":C5_ok,
        "C3_fixes_cell_over_L0":C3_ok,
        "ALL_HARD_GATES_A_TO_I":all(gates.values()),
    }

    for k,v in truth.items():
        print(f"{k:44s}: {v}")

    print("\n"+"="*92)
    print("INTERPRETATION GUARDRAILS")
    print("="*92)

    print("""
1. 24-cell existence != W(D4) stabilizer type != literal L25 identity.

2. Gate B is generated from native H4 root geometry. L25 does not generate
   or filter the 24-cell candidates.

3. A 24-root set is separately certified by the regular 24-cell fingerprint:
       (f0,f1,f2,f3) = (24,96,96,24),
   degree 8, octahedral facets, rank 4, antipodality, and metric spectrum.

4. |C24|=25 is NOT a pass.

5. Stabilizer order 192 is NOT a pass.

6. W(D4)-type stabilizers are NOT a pass.

7. The central result is:
       {Stab_W(C) : C in C24} == L25
   literally inside the same independently reconstructed W(H4).

8. Pair orbitals are computed before geometric fingerprint engineering.

9. A simple geometric fingerprint failing to distinguish pair classes does
   not erase a native W(H4) pair-orbital result.

10. The 8/12 split is called geometric only if the independently obtained
    geometric relation agrees pair-by-pair with subgroup intersection type.

11. I8 and I12 are classified from their actual group elements, not from
    their orders alone.

12. Triality is exploratory: no three geometric objects are named in advance.

13. C3/C5 are controls, not physical interpretations.

14. No E8, F4 finite-field geometry, rook graph, or RCFT interpretation is
    used to construct or rescue this experiment.

15. No downstream success rescues an upstream failed gate.

END SIM15.4
""")


main()




~~~~~~~~~~~~~~~~~~~~~~





RESULTS:





============================================================================================
SIM15.4 — NATIVE 24-CELL REALIZATION OF THE L_25 ARCHITECTURE
============================================================================================

FROZEN QUESTION:
Does the independently selected operational architecture coincide,
object-for-object and relation-for-relation, with native 24-cell
structure in H4?

A) FROZEN SIM15 REGRESSION--------------------------------------------------------------------------------------------
H4 roots = 120
|W(H4)| = 14400
simple-root indices = [0, 76, 5, 41]
involutions in W(H4): 571
C2^3 subgroups: 1200
C2^3 conjugacy classes: 5
class sizes: [75, 75, 300, 300, 450]
class 0: orbit=450 normalizer=32
class 1: orbit=300 normalizer=48
class 2: orbit=75 normalizer=192
  fingerprint: {'T8': True, 'N192': True, 'centralizer_T': False, 'GL24': False, 'GL_S4_hist': False, 'one_nonzero_fixed': True, 'center2': True, 'center_fixed': True} FULL= False
class 3: orbit=300 normalizer=48
class 4: orbit=75 normalizer=192
  fingerprint: {'T8': True, 'N192': True, 'centralizer_T': True, 'GL24': True, 'GL_S4_hist': True, 'one_nonzero_fixed': True, 'center2': True, 'center_fixed': True} FULL= True
|T0| = 8
|L0| = 192
|N0| = 576
|T75| = 75
|L25| = 25
GATE A = True

B) BLIND NATIVE 24-CELL CENSUS--------------------------------------------------------------------------------------------
D4 frame count = 4800
closure-size histogram = {24: 4800}
distinct 24-root closures = 25
certificate types = 1
 count = 25
 certificate = {'vertices': 24, 'rank': 4, 'antipodal': True, 'degree_hist': {8: 24}, 'edges': 96, 'triangles': 96, 'octahedral_facets': 24, 'ip_spectrum': {-2.0: 12, -1.0: 96, 0.0: 72, 1.0: 96}, 'passes': True}
certified native 24-cells = 25
GATE B = True

C) GEOMETRIC 24-CELL STABILIZERS--------------------------------------------------------------------------------------------
stabilizer-order histogram = {576: 25}
stabilizer order-histogram types = 1
 count 25 hist = {1: 1, 2: 43, 3: 80, 4: 84, 6: 272, 12: 96}
GATE C = False

D) LITERAL STABILIZER-FAMILY IDENTITY--------------------------------------------------------------------------------------------
|S| = 25
|L| = 25
|S cap L| = 0
|S \ L| = 25
|L \ S| = 25
GATE D — LITERAL S_25 == L_25 = False

E) W(H4)-EQUIVARIANCE--------------------------------------------------------------------------------------------
C24 family W-closed = True
GATE E = False
|C24 action image| = 7200
|C24 action kernel| = 2
z in kernel = True

F0) BLIND W(H4) PAIR ORBITALS ON C24--------------------------------------------------------------------------------------------
pair orbital count = 2
pair orbital sizes = [100, 200]
GATE F0 = True

F) NATIVE GEOMETRIC PAIR FINGERPRINTS--------------------------------------------------------------------------------------------
600-cell degree histogram = {12: 120}
600-cell edges = 720
orbital 0: size=100, geometric fingerprint types=1
 count 100 fingerprint = {'shared_vertices': 0, 'shared_edges': 0, 'shared_triangles': 0, 'shared_facets': 0, 'cross_distance_hist': {1: 72, 2: 144, 3: 216, 4: 144}, 'cross_ip_hist': {-1.61803399: 72, -1.0: 72, -0.61803399: 72, 0.0: 144, 0.61803399: 72, 1.0: 72, 1.61803399: 72}}
orbital 1: size=200, geometric fingerprint types=1
 count 200 fingerprint = {'shared_vertices': 6, 'shared_edges': 6, 'shared_triangles': 0, 'shared_facets': 0, 'cross_distance_hist': {0: 6, 1: 54, 2: 156, 3: 198, 4: 156, 5: 6}, 'cross_ip_hist': {-2.0: 6, -1.61803399: 54, -1.0: 102, -0.61803399: 54, 0.0: 144, 0.61803399: 54, 1.0: 102, 1.61803399: 54, 2.0: 6}}
global fingerprint types = 2
GATE F = True

G) GEOMETRY <-> L-INTERSECTION 8/12--------------------------------------------------------------------------------------------
observed intersection orders = []
GATE G = False

H) CLASSIFY I8 AND I12--------------------------------------------------------------------------------------------
GATE H = False

I) GEOMETRY-ONLY RANK-3 GRAPH--------------------------------------------------------------------------------------------
v                                     = 25
edges                                 = 100
degree_hist                           = {8: 25}
connected                             = True
adjacent_common_neighbor_hist         = {3: 100}
nonadjacent_common_neighbor_hist      = {2: 200}
spectrum                              = [(-2.0, 16), (3.0, 8), (8.0, 1)]
GATE I — geometry reconstructs srg(25,8,3,2) = True

J) EXPLORATORY TRIALITY ON EACH 24-CELL--------------------------------------------------------------------------------------------

K) C3 / C5 NATIVE 24-CELL CONTROLS--------------------------------------------------------------------------------------------

============================================================================================SIM15.4 HARD GATES
============================================================================================
A frozen SIM15 regression                       : True
B blind native 24-cell census                   : True
C W(D4)-type geometric stabilizers              : False
D LITERAL S_25 == L_25                          : False
E W(H4)-equivariant identification              : False
F0 blind pair orbitals 100+200                  : True
F two native geometric pair types               : True
G geometry <-> 8/12                             : False
H uniform I8/I12 types                          : False
I native srg(25,8,3,2)                          : True

============================================================================================MACHINE TRUTH PACKET
============================================================================================
H4_roots                                    : 120
H4_order                                    : 14400
successful_T_count                          : 75
compatible_L_count                          : 25
native_D4_frame_count                       : 4800
native_24root_closure_count                 : 25
certified_24cell_count                      : 25
24cell_stabilizer_order_hist                : {576: 25}
literal_stabilizer_matches                  : 0
literal_family_equality                     : False
identification_equivariant                  : False
C24_pair_orbit_sizes                        : [100, 200]
geometric_pair_fingerprint_types            : 2
L_intersection_values                       : []
I8_labels                                   : {}
I12_labels                                  : {}
geometry_graph_signature                    : {'v': 25, 'edges': 100, 'degree_hist': {8: 25}, 'connected': True, 'adjacent_common_neighbor_hist': {3: 100}, 'nonadjacent_common_neighbor_hist': {2: 200}, 'spectrum': [(-2.0, 16), (3.0, 8), (8.0, 1)]}
C5_on_C24_is_5x5cycles                      : False
C3_fixes_cell_over_L0                       : False
ALL_HARD_GATES_A_TO_I                       : False

============================================================================================INTERPRETATION GUARDRAILS
============================================================================================

1. 24-cell existence != W(D4) stabilizer type != literal L25 identity.

2. Gate B is generated from native H4 root geometry. L25 does not generate
   or filter the 24-cell candidates.

3. A 24-root set is separately certified by the regular 24-cell fingerprint:
       (f0,f1,f2,f3) = (24,96,96,24),
   degree 8, octahedral facets, rank 4, antipodality, and metric spectrum.

4. |C24|=25 is NOT a pass.

5. Stabilizer order 192 is NOT a pass.

6. W(D4)-type stabilizers are NOT a pass.

7. The central result is:
       {Stab_W(C) : C in C24} == L25
   literally inside the same independently reconstructed W(H4).

8. Pair orbitals are computed before geometric fingerprint engineering.

9. A simple geometric fingerprint failing to distinguish pair classes does
   not erase a native W(H4) pair-orbital result.

10. The 8/12 split is called geometric only if the independently obtained
    geometric relation agrees pair-by-pair with subgroup intersection type.

11. I8 and I12 are classified from their actual group elements, not from
    their orders alone.

12. Triality is exploratory: no three geometric objects are named in advance.

13. C3/C5 are controls, not physical interpretations.

14. No E8, F4 finite-field geometry, rook graph, or RCFT interpretation is
    used to construct or rescue this experiment.

15. No downstream success rescues an upstream failed gate.

END SIM15.4
