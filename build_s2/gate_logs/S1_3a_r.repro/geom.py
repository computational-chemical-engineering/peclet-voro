import sys, numpy as np, itertools
from scipy.spatial import Voronoi
d, cell = sys.argv[1], int(sys.argv[2])
meta = dict(kv.split('=',1) for kv in open(d+'/meta').read().split())
n = int(meta['chain']); h = float(meta['h']); L = float(meta['L'])
x = np.fromfile(f'{d}/chain_{n-1}').reshape(-1,3)
c = x[cell]
dx = (x - c + L/2) % L - L/2
sel = np.where(np.linalg.norm(dx,axis=1) < 3.5*h)[0]
p = dx[sel]
v = Voronoi(p)
ic = int(np.where(sel==cell)[0][0])
reg = v.regions[v.point_region[ic]]
assert -1 not in reg
V = v.vertices[reg]
dd = np.linalg.norm(V[:,None]-V[None],axis=2); np.fill_diagonal(dd, np.inf)
print(f'cell {cell} vertices={len(reg)} min vertex distance/h={dd.min()/h:.3e}')
# per vertex: the spread of distances to the nearest generators (cosphericity: 5th nearest vs 4th)
for k,vi in enumerate(reg):
    r = np.sort(np.linalg.norm(p - v.vertices[vi], axis=1))
    gap = (r[4]-r[3])/h
    if gap < 1e-6:
        print(f'  vertex {vi}: r4-r3 gap/h={gap:.3e} (near-cospherical 5 generators)')
# facets: ridges of ic
for (a,b), rv in zip(v.ridge_points, v.ridge_vertices):
    if ic in (a,b):
        o = b if a==ic else a
        P = v.vertices[rv]
        cen = P.mean(0)
        area = 0.5*np.linalg.norm(sum(np.cross(P[i]-cen, P[(i+1)%len(P)]-cen) for i in range(len(P))))
        # edges
        e = [np.linalg.norm(P[i]-P[(i+1)%len(P)])/h for i in range(len(P))]
        if sel[o] == int(sys.argv[3]) or min(e) < 1e-4 or area/h**2 < 1e-6:
            print(f'  facet to {sel[o]}: area/h^2={area/h**2:.3e} nverts={len(rv)} min edge/h={min(e):.3e}')
# the Delaunay tetrahedra of the vertices of the facet to argv[3]: shape quality 6√2·vol/l_max³ (1 regular, 0 flat)
for (a,b), rv in zip(v.ridge_points, v.ridge_vertices):
    if ic in (a,b) and sel[b if a==ic else a] == int(sys.argv[3]):
        for vi in rv:
            r = np.linalg.norm(p - v.vertices[vi], axis=1)
            q4 = np.argsort(r)[:5]
            T = p[q4[:4]]
            vol = abs(np.linalg.det(T[1:]-T[0]))/6
            lm = max(np.linalg.norm(T[i]-T[j]) for i in range(4) for j in range(i))
            print(f'  facet vertex {vi}: generators {[int(sel[q]) for q in q4[:4]]} quality={6*np.sqrt(2)*vol/lm**3:.3e} r5-r4/h={(r[q4[4]]-r[q4[3]])/h:.2e}')
