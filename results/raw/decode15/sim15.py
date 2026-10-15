import sim14, numpy as np
SLOT_MIB = 1867776/2**20
G=[0,1,2,6]
base = sim14.run(8393+1074, tuple(G))   # today's live: 9467 slots at short ctx
print("today (9467 slots): N=%.1f M=%.2f W=%.2f ms model %.2f tok/s" % (base[0],base[1],base[3],base[2]))
items = [("code arena 256->24 MiB",232),("mapped arena 48->16",32),("RAM-tier VRAM mirror 204 MiB",204),("vring 64->32",32),("vring 64->16",48),
         ("reserve 600->500",100),("reserve 600->400",200),("reserve 600->300",300),("MEMINFO_FIXED 1024->920",104),
         ("session pooled+rope elastic (322)",322),("static mirror embedding (322, NOT feasible)",322),("lazy shadow x1 (112)",112),("lazy shadow x6 (672)",672),
         ("lmem 73 MiB",73),("guard+frag chunks (65)",65),("MTP draft head 213",213),("1 GiB",1024)]
rows=[]
for name,mib in items:
    ex=int(mib/SLOT_MIB)
    r=sim14.run(8393+1074+ex, tuple(G))
    rows.append((name,mib,ex,r[2],100*(r[2]/base[2]-1),r[3]))
    print(f"{name:45s} {mib:5d} MiB = +{ex:4d} slots -> {r[2]:.2f} tok/s model ({100*(r[2]/base[2]-1):+.2f}%)  W {r[3]:.2f} ms")
