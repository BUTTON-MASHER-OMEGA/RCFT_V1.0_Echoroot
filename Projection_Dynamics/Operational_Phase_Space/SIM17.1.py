#!/usr/bin/env python3
"""SIM17.1 — Octavian 240-unit Moufang loop / signed-subloop census.

Exact integer arithmetic using doubled real octonion coordinates.
The signed Cayley-Dickson product and vertex labels are frozen to SIM17.0.
A MAXIMAL OCTAVIAN ORDER is selected by explicitly recorded half-unit support
code. Its 240 norm-one units are exhaustively checked for multiplication
closure, inverse closure, Moufang identity, and E8 root inner products.

Scope: mathematical certification of ONE chosen octavian order.
No intrinsic selection of an order, a subloop, or C8 is claimed.
"""
import itertools
import json
from collections import Counter
import numpy as np

N=8
SIM14_COORD={0:0,1:4,2:1,3:5,4:7,5:3,6:6,7:2}
SIM14_VERTEX={v:k for k,v in SIM14_COORD.items()}
ETA=(1,1,1,1,1,1,1,-1)

def check(label, predicate):
    if not bool(predicate):
        raise AssertionError('FAIL | '+label)
    print('PASS | '+label, flush=True)

def cd(i,j,n=8):
    if n==1: return (1,0)
    k=n//2
    if i<k and j<k: return cd(i,j,k)
    if i<k and j>=k:
        s,v=cd(j-k,i,k);return (s,v+k)
    if i>=k and j<k:
        s,v=cd(i-k,j,k);return (s*(1 if j==0 else -1),v+k)
    s,v=cd(j-k,i-k,k)
    return (-s*(1 if j-k==0 else -1),v)

SIGN=np.zeros((8,8),dtype=np.int64)
INDEX=np.zeros((8,8),dtype=np.int64)
for i in range(8):
    for j in range(8):
        z,k=cd(SIM14_COORD[i],SIM14_COORD[j]);v=SIM14_VERTEX[k]
        SIGN[i,j]=z*ETA[i]*ETA[j]*ETA[v]
        INDEX[i,j]=v

def mul2(a,b):
    """Multiply doubled coordinates: inputs x=2a,y=2b; output 4ab."""
    out=[0]*8
    for i,ai in enumerate(a):
        if ai:
            for j,bj in enumerate(b):
                if bj:
                    out[int(INDEX[i,j])]+=int(SIGN[i,j])*int(ai)*int(bj)
    return tuple(out)

# GATE A0: reproduce exact SIM17.0 signed multiplication and L_s=J0
E=np.eye(8,dtype=np.int64)
Omega=np.zeros((8,8),dtype=np.int64)
for i,j in ((0,2),(1,3),(4,6),(5,7)):
    Omega[i,j]=1;Omega[j,i]=-1
left_s=np.zeros((8,8),dtype=np.int64)
s=SIM14_VERTEX[1]
for j in range(8):left_s[INDEX[s,j],j]=SIGN[s,j]
check('A0 frozen SIM17.0 Cayley-Dickson signs and J0',np.array_equal(left_s,-Omega))

# GATE A1: specify the actual integral order's 14 half-support planes.
# IMPORTANT: These 14 supports differ from the 14 affine planes of SIM14.
# They are a particular extended Hamming 8,4,4 code, explicitly pinned
# in SIM17.0's vertex coordinates and multiplication convention.
HALF_SUPPORTS=(
    (0,1,2,3),(0,1,4,5),(0,1,6,7),
    (0,2,4,7),(0,2,5,6),(0,3,4,6),(0,3,5,7),
    (1,2,4,6),(1,2,5,7),(1,3,4,7),(1,3,5,6),
    (2,3,4,5),(2,3,6,7),(4,5,6,7)
)

def make_roots(half_supports):
    r=set()
    for i in range(8):
        for sign in (-2,2):
            a=[0]*8;a[i]=sign;r.add(tuple(a))
    for support in half_supports:
        for signs in itertools.product((-1,1),repeat=4):
            a=[0]*8
            for i,v in zip(support,signs):a[i]=v
            r.add(tuple(a))
    return sorted(r)

ROOTS=make_roots(HALF_SUPPORTS)
root_to_id={r:i for i,r in enumerate(ROOTS)}
check('A1 exactly 240 distinct norm-one vectors',len(ROOTS)==240 and all(sum(v*v for v in r)==4 for r in ROOTS))
check('A1 16 signed basis + 224 half-integral units',sum(max(abs(v) for v in r)==2 for r in ROOTS)==16)
# These half-root supports form the length-eight extended binary Hamming code.
# Since the eight coordinate units are integral, this code describes the
# index-16 overlattice Z^8 + 1/2*C, with covolume 1/16.
support_masks={sum(1<<i for i in u) for u in HALF_SUPPORTS}
code={0,255}|support_masks
check('A1 extended Hamming support code (16 codewords, doubly even)',
      len(code)==16 and all((x^y) in code for x in code for y in code)
      and Counter(x.bit_count() for x in code)==Counter({0:1,4:14,8:1}))
print('A1 | octavian lattice covolume relative to Z^8: 1/16',flush=True)
R=np.array(ROOTS,dtype=np.int64)

# E8 root-system Gram profile at normalized squared length 1.
gram=R@R.T
profile=Counter(int(v) for v in gram[0])
check('A2 E8 root inner-product distribution',profile==Counter({-4:1,-2:56,0:126,2:56,4:1}) and all(Counter(int(v) for v in row)==profile for row in gram))
# Every root reflection r -> r - 2<r,a>/<a,a> a preserves roots.
for k in range(240):
    img=R-(gram[:,k,None]//2)*R[k]
    if any(tuple(v) not in root_to_id for v in img):
        raise AssertionError('FAIL | A2 E8 root reflection closure')
check('A2 all 240 root reflections preserve the 240-set',True)

# GATE B: full multiplication table, exact 240 x 240 products.
M=np.zeros((240,240),dtype=np.int64)
for i,ra in enumerate(ROOTS):
    for j,rb in enumerate(ROOTS):
        doubled=mul2(ra,rb)
        if any(v%2 for v in doubled):
            raise AssertionError('FAIL | B nonintegral doubled product')
        r=tuple(v//2 for v in doubled)
        if r not in root_to_id:
            raise AssertionError('FAIL | B unit multiplication nonclosure')
        M[i,j]=root_to_id[r]
check('B exhaustive 57,600-product octavian unit closure',True)
identity=root_to_id[(2,0,0,0,0,0,0,0)]
negative=np.array([root_to_id[tuple(-v for v in r)] for r in ROOTS],dtype=np.int64)
inverse=np.array([root_to_id[(r[0],)+tuple(-v for v in r[1:])] for r in ROOTS],dtype=np.int64)
check('B identity / norm-conjugation inverses',all(M[identity,i]==i and M[i,identity]==i and M[i,inverse[i]]==identity and M[inverse[i],i]==identity for i in range(240)))
check('B signed negative central',all(M[negative[i],j]==negative[M[i,j]] and M[i,negative[j]]==negative[M[i,j]] for i in range(240) for j in range(240)))
check('B multiplication has row and column cancellation',all(len(set(int(v) for v in M[i,:]))==240 and len(set(int(v) for v in M[:,i]))==240 for i in range(240)))

# Moufang identity (xy)(zx)=x((yz)x), exhaustively for 240^3 triples.
for x in range(240):
    lhs=M[M[x,:][:,None],M[:,x][None,:]]
    rhs=M[x,M[M[:,:],x]]
    if not np.array_equal(lhs,rhs):
        raise AssertionError('FAIL | B Moufang identity for x='+str(x))
check('B exhaustive Moufang identity for all 13,824,000 triples',True)
# Explicit nonassociative witness using SIM17 basis indices.
def id_basis(v,negative=False):
    a=[0]*8;a[v]=-2 if negative else 2;return root_to_id[tuple(a)]
nonassoc=None
for i,j,k in itertools.product(range(1,8),repeat=3):
    x,y,z=map(id_basis,(i,j,k))
    if M[M[x,y],z]!=M[x,M[y,z]]:
        nonassoc=(i,j,k);break
check('B nonassociative witness exists inside SIM17 signed basis',nonassoc is not None)

# GATE C: enumerate ALL U16 subloops with quotient C2^3.
# Every noncentral element must square to -1; these are precisely
# the 126 purely imaginary units of the 240-loop.
minus_one=negative[identity]
imag=[i for i in range(240) if M[i,i]==minus_one]
check('C exactly 126 imaginary square-minus-one octavians',len(imag)==126)
canon=sorted(min(i,int(negative[i])) for i in imag if i<int(negative[i]))
check('C 63 opposite-sign imaginary directions',len(canon)==63)

# Pick an orthogonal imaginary pair u,v, then w orthogonal to
# span_R{1,u,v,uv}. It generates U16 by alternativity + closure.
# Quotient representatives are canonical opposite-sign pairs.
def signed_pair(i):return min(i,int(negative[i]))
subloops=set()
for aa,u in enumerate(canon):
    for v in canon[aa+1:]:
        if int(R[u]@R[v])!=0:continue
        uv=int(M[u,v]);uvpair=signed_pair(uv)
        four={identity,minus_one,u,int(negative[u]),v,int(negative[v]),uv,int(negative[uv])}
        for w in canon:
            if signed_pair(w) in (u,v,uvpair):continue
            if any(int(R[w]@R[t])!=0 for t in (u,v,uv)):continue
            uw=int(M[u,w]);vw=int(M[v,w]);uvw=int(M[uv,w])
            seven={u,v,uvpair,signed_pair(w),signed_pair(uw),signed_pair(vw),signed_pair(uvw)}
            if len(seven)!=7:continue
            subloops.add(tuple(sorted(seven)))
check('C nonempty census of candidate signed U16 subloops',len(subloops)>0)
# Verify every one under all 16x16 products, no reliance solely on generators.
for sub in subloops:
    all_ids={identity,minus_one}
    for t in sub:all_ids.add(t);all_ids.add(int(negative[t]))
    if len(all_ids)!=16 or any(int(M[x,y]) not in all_ids for x in all_ids for y in all_ids):
        raise AssertionError('FAIL | C candidate 16-unit closure')
check('C every enumerated 16-unit candidate closed (all 256 products)',True)
incidences=Counter(t for sub in subloops for t in sub)
check('C uniform signed-subloop incidence over all 63 imaginary ± pairs',
      set(incidences)==set(canon) and len(set(incidences.values()))==1)
print('C | subloops containing each imaginary ± pair:',next(iter(incidences.values())),flush=True)
# Exhaustiveness: Any U16 with elementary abelian sign quotient has seven
# noncentral sign pairs of square -1; select u,v and w external to their
# associative quaternionic subloop, recovering it in the census above.
print('C | signed U16 subloop census:',len(subloops),flush=True)

# GATE D: identify frozen SIM17.0 +/- basis and its Fano/associator geometry.
frozen=tuple(sorted(signed_pair(id_basis(i)) for i in range(1,8)))
check('D frozen SIM17 signed basis subloop appears in octavian census',frozen in subloops)
F=set([identity,minus_one])
for i in range(1,8):F|={id_basis(i),id_basis(i,True)}
check('D multiplication on all frozen 16 signed basis units agrees with SIM17.0',all(M[id_basis(i,si),id_basis(j,sj)]==root_to_id[tuple(((-1 if si else 1)*(-1 if sj else 1)*int(SIGN[i,j])*2 if k==INDEX[i,j] else 0) for k in range(8))] for i in range(8) for j in range(8) for si in (False,True) for sj in (False,True)))
# For any U16 candidate: quotient by ±, every nonzero class has order 2,
# the product table is associative and commutative, hence F2^3.
for sub in subloops:
    elems=[identity]+list(sub)
    classes={identity:0,minus_one:0}
    for k,q in enumerate(sub,1):classes[q]=k;classes[int(negative[q])]=k
    for x,y in itertools.product(elems,repeat=2):
        if classes[int(M[x,y])]!=classes[int(M[y,x])]:
            raise AssertionError('FAIL | D quotient noncommutative')
    if any(classes[int(M[M[x,y],z])]!=classes[int(M[x,M[y,z]])] for x,y,z in itertools.product(elems,repeat=3)):
        raise AssertionError('FAIL | D quotient nonassociative')
check('D every U16 / central signs is elementary abelian order 8',True)

# Reproduce the complete SIM17.0 associator/projector formula in this
# 240-unit parent algebra, restricted to its embedded eight-real-dimensional
# octonionic coordinate carrier. This is NOT merely a quotient check.
LEFT=[]
for i in range(8):
    L=np.zeros((8,8),dtype=np.int64)
    for j in range(8):L[INDEX[i,j],j]=SIGN[i,j]
    LEFT.append(L)
for i,j in itertools.combinations(range(1,8),2):
    z=int(INDEX[i,j]);sgn=int(SIGN[i,j])
    defect=sgn*LEFT[z]-LEFT[i]@LEFT[j]
    direction={0,SIM14_COORD[i],SIM14_COORD[j],SIM14_COORD[i]^SIM14_COORD[j]}
    q=np.zeros((8,8),dtype=np.int64)
    for d in direction:q[SIM14_VERTEX[d],SIM14_VERTEX[d]]=1
    if not np.array_equal(defect.T@defect,4*(E-q)):
        raise AssertionError('FAIL | D embedded associator-projector identity')
check('D all 21 SIM17.0 associator/projector identities reproduced',True)

# GATE E: grade the exact relationship to SIM14/SIM17 Fano lines.
# Three nonzero quotient elements whose product is the fourth describe
# seven rank-2 subgroups, corresponding to seven quaternionic Q8 subloops.
for sub in subloops:
    reps=list(sub);groups=set()
    for u,v in itertools.combinations(reps,2):
        z=signed_pair(int(M[u,v]))
        groups.add(tuple(sorted((u,v,z))))
    if len(groups)!=7 or any(len(set(g))!=3 for g in groups):
        raise AssertionError('FAIL | E Fano/Q8 incidence')
check('E all signed U16 subloops have seven quaternionic Q8 incidence lines',True)
# For frozen basis, quotient under index XOR is exactly SIM14 F2^3.
check('E frozen signed-unit quotient indexes multiply by SIM14 XOR',all(SIM14_COORD[int(INDEX[i,j])]==(SIM14_COORD[i]^SIM14_COORD[j]) for i in range(8) for j in range(8)))

# GATE F: Noncanonical selection and geometry-only controls.
# Ordinary SIM14 affine planes ARE NOT the octavian half-support planes.
lines={frozenset((0,a,b,a^b)) for a,b in itertools.combinations(range(1,8),2)}
sim14planes={frozenset(SIM14_VERTEX[x] for x in q) for q in lines}
sim14planes|={frozenset(range(8))-q for q in tuple(sim14planes)}
check('F octavian half-root support code differs from SIM14 Fano affine planes',set(map(frozenset,HALF_SUPPORTS))!=sim14planes)
print('F | no canonical selection of the frozen 16-subloop established')

packet={
 'sim':'17.1', 'scope':'one multiplication-compatible maximal octavian order, exact finite audit',
 'unit_count':len(ROOTS),'root_gram_profile_doubled':dict(sorted(profile.items())),
 'moufang_triples_tested':240**3,'nonassoc_basis_witness':nonassoc,
 'imaginary_units':len(imag),'signed_subloop_count':len(subloops),
 'subloops_per_imaginary_pair':next(iter(incidences.values())),
 'half_support_binary_code_size':len(code),
 'sim17_frozen_U16_included':frozen in subloops,
 'all_subloop_quotients_C2_cubed':True,
 'octavian_half_supports':[list(x) for x in HALF_SUPPORTS],
 'octavian_automorphism_orbits':'NOT_CLASSIFIED',
 'canonical_subloop_selection':False,
 'canonical_C8_selection':False,
 'physical_projection_derived':False,
 'status':'PASS_WITH_DECLARED_SELECTION_LIMITS'
}
print('TRUTH_PACKET_JSON='+json.dumps(packet,sort_keys=True),flush=True)
print('END SIM17.1',flush=True)







"""




RESULTS:



PASS | A0 frozen SIM17.0 Cayley-Dickson signs and J0
PASS | A1 exactly 240 distinct norm-one vectors
PASS | A1 16 signed basis + 224 half-integral units
PASS | A1 extended Hamming support code (16 codewords, doubly even)
A1 | octavian lattice covolume relative to Z^8: 1/16
PASS | A2 E8 root inner-product distribution
PASS | A2 all 240 root reflections preserve the 240-set
PASS | B exhaustive 57,600-product octavian unit closure
PASS | B identity / norm-conjugation inverses
PASS | B signed negative central
PASS | B multiplication has row and column cancellation
PASS | B exhaustive Moufang identity for all 13,824,000 triples
PASS | B nonassociative witness exists inside SIM17 signed basis
PASS | C exactly 126 imaginary square-minus-one octavians
PASS | C 63 opposite-sign imaginary directions
PASS | C nonempty census of candidate signed U16 subloops
PASS | C every enumerated 16-unit candidate closed (all 256 products)
PASS | C uniform signed-subloop incidence over all 63 imaginary ± pairs
C | subloops containing each imaginary ± pair: 7
C | signed U16 subloop census: 63
PASS | D frozen SIM17 signed basis subloop appears in octavian census
PASS | D multiplication on all frozen 16 signed basis units agrees with SIM17.0
PASS | D every U16 / central signs is elementary abelian order 8
PASS | D all 21 SIM17.0 associator/projector identities reproduced
PASS | E all signed U16 subloops have seven quaternionic Q8 incidence lines
PASS | E frozen signed-unit quotient indexes multiply by SIM14 XOR
PASS | F octavian half-root support code differs from SIM14 Fano affine planes
F | no canonical selection of the frozen 16-subloop established
TRUTH_PACKET_JSON={"all_subloop_quotients_C2_cubed": true, "canonical_C8_selection": false, "canonical_subloop_selection": false, "half_support_binary_code_size": 16, "imaginary_units": 126, "moufang_triples_tested": 13824000, "nonassoc_basis_witness": [1, 2, 4], "octavian_automorphism_orbits": "NOT_CLASSIFIED", "octavian_half_supports": [[0, 1, 2, 3], [0, 1, 4, 5], [0, 1, 6, 7], [0, 2, 4, 7], [0, 2, 5, 6], [0, 3, 4, 6], [0, 3, 5, 7], [1, 2, 4, 6], [1, 2, 5, 7], [1, 3, 4, 7], [1, 3, 5, 6], [2, 3, 4, 5], [2, 3, 6, 7], [4, 5, 6, 7]], "physical_projection_derived": false, "root_gram_profile_doubled": {"-4": 1, "-2": 56, "0": 126, "2": 56, "4": 1}, "scope": "one multiplication-compatible maximal octavian order, exact finite audit", "signed_subloop_count": 63, "sim": "17.1", "sim17_frozen_U16_included": true, "status": "PASS_WITH_DECLARED_SELECTION_LIMITS", "subloops_per_imaginary_pair": 7, "unit_count": 240}
END SIM17.1


** Process exited - Return Code: 0 **
"""
