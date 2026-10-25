"""Bounded diagnostic of supplied coordinates using the existing construction only."""
from pathlib import Path
import ast,json,math
from itertools import permutations,product
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
geo=ROOT/'projects/lisi-smg-test/reference_geometry'
source=(geo/'parse_120cell_adjacency_v2.py').read_text(encoding='utf-8-sig')
tree=ast.parse(source)
names={'get_signed_permutations','get_even_signed_permutations','generate_120cell_vertices'}
functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
env={'np':np,'permutations':permutations,'product':product}
exec(compile(functions,str(geo/'parse_120cell_adjacency_v2.py'),'exec'),env)
generated=env['generate_120cell_vertices']()
dist=((generated[:,None,:]-generated[None,:,:])**2).sum(2)
minimum=dist[dist>1e-10].min()
ag=np.isclose(dist,minimum,atol=1e-8).astype(float)
lam=float(np.mean((ag@generated)*generated)/np.mean(generated*generated))
coords={int(l.split(':')[0]):[float(x) for x in l.split(':')[1].split()] for l in (geo/'120cell_vertices.txt').read_text().splitlines() if l.strip()}
X=np.array([coords[i] for i in range(1,601)])
A=np.zeros((600,600))
for l in (geo/'120cell_adjacency.txt').read_text().splitlines():
    if '(' not in l:continue
    i,rest=l.split('(');ns=[int(x.strip(' ,)')) for x in rest.split(',')]
    for j in ns:A[int(i)-1,j-1]=1
res=np.linalg.norm(A@X-lam*X,axis=1)
print(json.dumps({'generated_count':len(generated),'minimum_squared_distance':float(minimum),
 'generated_degree_range':[float(ag.sum(1).min()),float(ag.sum(1).max())],
 'lambda':lam,'generator_eigen_residual':float(np.max(np.abs(ag@generated-lam*generated))),
 'largest_input_residuals':[(int(i+1),float(res[i])) for i in np.argsort(res)[-20:][::-1]]},indent=2))
np.savez(HERE/'coordinate-diagnostic.npz',X=X,A=A,generated=generated,ag=ag,lam=lam)
