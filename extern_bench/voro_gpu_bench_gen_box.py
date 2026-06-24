import sys, random
G = 20            # cells per axis -> G^3*6 tets
N = 1000000       # sites
L = 1000.0
# vertices (G+1)^3
nv = (G+1)**3
def vid(i,j,k): return i + (G+1)*j + (G+1)*(G+1)*k
with open("box.tet","w") as f:
    f.write(f"{nv} vertices\n{G*G*G*6} cells\n")
    for k in range(G+1):
        for j in range(G+1):
            for i in range(G+1):
                f.write(f"{i*L/G} {j*L/G} {k*L/G}\n")
    # 6-tet decomposition of each cube along diagonal 0-7; corner idx = dx+2dy+4dz
    corners=[(0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1)]
    tets=[(0,1,3,7),(0,3,2,7),(0,2,6,7),(0,6,4,7),(0,4,5,7),(0,5,1,7)]
    for ck in range(G):
        for cj in range(G):
            for ci in range(G):
                gv=[vid(ci+dx,cj+dy,ck+dz) for (dx,dy,dz) in corners]
                for t in tets:
                    f.write(f"4 {gv[t[0]]} {gv[t[1]]} {gv[t[2]]} {gv[t[3]]}\n")
random.seed(7)
with open("box_sites.xyz","w") as f:
    f.write(f"{N}\n")
    for _ in range(N):
        f.write(f"{random.random()*L} {random.random()*L} {random.random()*L}\n")
print(f"wrote box.tet ({nv} verts, {G*G*G*6} tets) and box_sites.xyz ({N} sites)")
