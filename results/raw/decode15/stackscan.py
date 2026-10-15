import sys, struct, collections, os
sys.path.insert(0, os.path.expanduser("~/shim")); sys.path.insert(0, os.path.expanduser("~/tg-strata"))
import tgcu
class FakeCtx:
    class E(dict):
        def __contains__(self, k): return True
        def __getitem__(self, k): return 0
    externs = E()
    def code_alloc(self, n): return 0
    def write_vram(self, a, b): pass
binf = sys.argv[1]
data = open(binf, "rb").read()
magic = struct.pack("<I", 0xBA55ED50)
pos = 0; n = 0; rows = []
while True:
    i = data.find(magic, pos)
    if i < 0: break
    pos = i + 4
    try:
        hdr = data[i:i+16]; _, ver, hsz, fsz = struct.unpack_from("<IHHQ", hdr)
        if ver != 1 or hsz != 16 or fsz > 200<<20: continue
        for arch, elf in tgcu.fatbin_cubins(data[i:i+hsz+fsz]):
            try: m = tgcu.Module(FakeCtx(), elf)
            except Exception as e:
                print("module fail", type(e).__name__, str(e)[:80]); continue
            n += 1
            for k in m.kernels.values():
                rows.append((k.stack, k.regs, k.name, arch))
    except Exception as e:
        print("fatbin fail at", i, type(e).__name__, str(e)[:80])
print("cubins parsed", n, "kernels", len(rows))
rows.sort(reverse=True)
c = collections.Counter(r[0] for r in rows)
print("stack histogram (bytes: count):", sorted(c.items(), reverse=True)[:12])
for r in rows[:25]: print(r[0], r[1], r[2][:150])
