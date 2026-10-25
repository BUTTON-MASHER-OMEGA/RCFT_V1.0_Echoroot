"""Bounded graph-certified reconstruction from the existing native H4 root function.

No group census, optimizer, dynamics, or heavy simulation is run.
The original coordinate table remains immutable.
"""
from pathlib import Path
from itertools import combinations
from collections import defaultdict
import ast,hashlib,json,math
import numpy as np
import networkx as nx

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
project=ROOT/'projects/lisi-smg-test';geo=project/'reference_geometry'
# Use the exact code portion already preserved/extracted during onboarding.
root_source=ROOT/'projects/ops-codex/research/extracted-ops/SIM15.6.py'
tree=ast.parse(root_source.read_text(encoding='utf-8'))
module=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'parity','roots_h4'}],type_ignores=[])
env={'np':np,'it':__import__('itertools'),'math':math,'PHI':(1+math.sqrt(5))/2}
exec(compile(module,str(root_source),'exec'),env)
R=env['roots_h4']()
d=((R[:,None,:]-R[None,:,:])**2).sum(2);minimum=d[d>1e-10].min()
primal=nx.from_numpy_array(np.isclose(d,minimum,atol=1e-8).astype(int))
assert len(R)==120 and set(dict(primal.degree()).values())=={12}
tetra=set()
for i in primal:
    for j,k,l in combinations(sorted(primal[i]),3):
        if primal.has_edge(j,k) and primal.has_edge(j,l) and primal.has_edge(k,l):
            tetra.add(tuple(sorted((i,j,k,l))))
tetra=sorted(tetra);assert len(tetra)==600
faces=defaultdict(list)
for i,t in enumerate(tetra):
    for face in combinations(t,3):faces[face].append(i)
assert len(faces)==1200 and all(len(v)==2 for v in faces.values())
dual=nx.Graph();dual.add_nodes_from(range(600));dual.add_edges_from(faces.values())
assert dual.number_of_edges()==1200 and set(dict(dual.degree()).values())=={4}
target=nx.Graph();target.add_nodes_from(range(1,601))
for l in (geo/'120cell_adjacency.txt').read_text().splitlines():
    if '(' not in l:continue
    i,rest=l.split('(')
    for j in [int(x.strip(' ,)')) for x in rest.split(',')]:target.add_edge(int(i),j)
mapping=nx.vf2pp_isomorphism(dual,target)
assert mapping is not None
assert {tuple(sorted((mapping[u],mapping[v]))) for u,v in dual.edges}=={tuple(sorted(e)) for e in target.edges}
Y=np.array([R[list(t)].mean(axis=0) for t in tetra])
by_source=np.zeros((600,4))
for i,j in mapping.items():by_source[j-1]=Y[i]
original={int(l.split(':')[0]):[float(x) for x in l.split(':')[1].split()] for l in (geo/'120cell_vertices.txt').read_text().splitlines() if l.strip()}
X=np.array([original[i] for i in range(1,601)])
# The source table sets the frame/scale. Orthogonal Procrustes does not deform the geometry.
good=np.ones(600,dtype=bool)
for _ in range(4):
    u,sv,vh=np.linalg.svd(by_source[good].T@X[good]);rotation=u@vh
    scale=float(np.sum((by_source[good]@rotation)*X[good])/np.sum(by_source[good]**2))
    reconstructed=scale*by_source@rotation
    error=np.linalg.norm(X-reconstructed,axis=1)
    # Initial all-row fit is biased by the known large corruptions; trim only
    # gross residuals relative to the majority, then refit the same rigid frame.
    good=error<max(1e-3,5*float(np.median(error)))
assert good.sum()==597,(good.sum(),sorted(error)[-10:])
bad=[int(i+1) for i in np.flatnonzero(~good)]
assert bad==[327,552,578],bad
# Every old intact row independently anchors the same frame within its supplied precision.
assert float(error[good].max())<1e-4
sq=((reconstructed[:,None,:]-reconstructed[None,:,:])**2).sum(2)
mind=sq[sq>1e-10].min();geometric=nx.from_numpy_array(np.isclose(sq,mind,atol=1e-9).astype(int))
expected={tuple(sorted((u-1,v-1))) for u,v in target.edges}
assert {tuple(sorted(e)) for e in geometric.edges}==expected
assert len({tuple(np.round(row,10)) for row in reconstructed})==600
assert np.max(np.abs(np.linalg.norm(reconstructed,axis=1)-np.linalg.norm(reconstructed[0])))<1e-12
assert np.all(np.isclose(np.min(((reconstructed[:,None,:]+reconstructed[None,:,:])**2).sum(2),axis=1),0,atol=1e-12))
out=geo/'120cell_vertices.repaired.txt'
out.write_text(''.join(f'{i+1}: '+' '.join(f'{v:.10f}' for v in row)+'\n' for i,row in enumerate(reconstructed)),encoding='utf-8')
packet={'method':'Native SIM15.6 H4 roots -> nearest-neighbor 600-cell -> tetrahedral dual -> exact graph-isomorphism witness -> similarity frame from 597 intact source rows',
 'original_sha256':hashlib.sha256((geo/'120cell_vertices.txt').read_bytes()).hexdigest(),
 'root_source_sha256':hashlib.sha256(root_source.read_bytes()).hexdigest(),
 'repaired_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'generated_tetrahedra':600,'dual_edges':1200,'target_ids':600,'unique_coordinates':600,
 'native_root_coordinates':R.tolist(),
 'root_function_names':['parity','roots_h4'],
 'severely_corrupted_source_ids':bad,'intact_source_max_frame_error':float(error[good].max()),
 'scale':scale,'rotation':rotation.tolist(),'minimum_squared_distance':float(mind),
 'source_ids_to_native_tetrahedra':{str(mapping[i]):list(t) for i,t in enumerate(tetra)},
 'severe_row_corrections':{str(i):{'before':X[i-1].tolist(),'after':reconstructed[i-1].tolist(),'frame_error_before':float(error[i-1])} for i in bad},
 'checks':{'target_edge_set_exact':True,'all_degrees_four':True,'antipodal_pairs':300,'uniform_norm':True},
 'precision_note':'All 600 rows regenerated from one verified regular-polytope construction; original orientation/scale recovered numerically from intact rows. Ten-decimal coordinates are a numerical realization, not symbolic exact values. Original source table preserved byte-for-byte.',
 'legacy_generator_note':'parse_120cell_adjacency_v2.py generate_120cell_vertices returned 600 points but nearest-neighbor degrees ranged 2..4 in the bounded diagnostic; it was not used as the correction authority and remains historical.'}
(project/'artifacts/coordinate_repair.json').write_text(json.dumps(packet,indent=2)+'\n',encoding='utf-8')
(HERE/'coordinate-repair-result.json').write_text(json.dumps({k:v for k,v in packet.items() if k!='source_ids_to_native_tetrahedra'},indent=2),encoding='utf-8')
print(json.dumps({'status':'reconstructed and graph-verified','bad_ids':bad,'max_intact_error':float(error[good].max())},indent=2))
