#!/usr/bin/env python3
"""compare-engine.py LIVE REBUILT [ROOT_A ROOT_B] : compare two Strata engine Mach-O binaries (macOS, otool/nm needed).

The two builds may sit at different install roots as long as the root strings have the same length (the deployed engine was built at /Users/sero, the
scratch rebuild of docs/reproduce.md section 11 at /tmp/recipe: both 11 bytes); ROOT_A ROOT_B (defaults /Users/sero /tmp/recipe) are the root of LIVE and of
REBUILT, the latter is rewritten to the former byte for byte before comparing.  Prints: file sizes, the symbol table sizes (every symbol's size = distance
to the next symbol), a per-section hash table, which 4-byte instruction words of __text differ and what kind they are (adrp/add = a global's address moved),
and the embedded CUDA fatbins one by one (nvcc names some kernels with a per-compilation id, so a few fatbins differ from build to build).
Exit status 0 always: it reports, it does not judge."""
import hashlib, re, struct, subprocess, sys, collections

live, new = sys.argv[1], sys.argv[2]
ra = (sys.argv[3] if len(sys.argv) > 3 else "/Users/sero").encode()
rb = (sys.argv[4] if len(sys.argv) > 4 else "/tmp/recipe").encode()
assert len(ra) == len(rb), "the two roots must have the same length"
A = open(live, "rb").read(); B = open(new, "rb").read().replace(rb, ra)
print(f"file sizes: live {len(A)}  rebuilt {len(B)} (after rewriting {rb.decode()} -> {ra.decode()}: {open(new, 'rb').read().count(rb)} occurrences of the root)")

def run(*c): return subprocess.run(c, capture_output=True, text=True).stdout
def write_tmp(data, name):
    open(name, "wb").write(data); return name
tmp_b = write_tmp(B, "/tmp/compare-engine.rebuilt")

def sizes(p):
    L = []
    for l in run("nm", "-n", p).split("\n"):
        m = re.match(r"([0-9a-f]{16}) (\w) (.+)", l)
        if m: L.append((int(m.group(1), 16), m.group(2), m.group(3)))
    return {(t, n): L[i + 1][0] - a for i, (a, t, n) in enumerate(L[:-1])}
sa, sb = sizes(live), sizes(tmp_b)
print(f"symbols: live {len(sa)}  rebuilt {len(sb)}  only in live {len(set(sa) - set(sb))}  only in rebuilt {len(set(sb) - set(sa))}  "
      f"different size {sum(1 for k in sa if k in sb and sa[k] != sb[k])}")
for k in [k for k in sa if k in sb and sa[k] != sb[k]][:10]: print("  size differs:", k[1][:100], sa[k], sb[k])

def sections(p):
    out = run("otool", "-l", p)
    return [(m.group(2), m.group(1), int(m.group(3), 16), int(m.group(4), 16), int(m.group(5)))
            for m in re.finditer(r"sectname (\S+)\n\s+segname (\S+)\n\s+addr (\S+)\n\s+size (\S+)\n\s+offset (\d+)", out)]
SA, SB = sections(live), sections(tmp_b)
print(f"\n{'section':30s} {'live size':>12s} {'rebuilt size':>12s}  content")
for a, b in zip(SA, SB):
    ha = hashlib.sha1(A[a[4]:a[4] + a[3]]).hexdigest() if a[4] else "zerofill"
    hb = hashlib.sha1(B[b[4]:b[4] + b[3]]).hexdigest() if b[4] else "zerofill"
    print(f"{a[0] + ',' + a[1]:30s} {a[3]:12d} {b[3]:12d}  {'identical' if ha == hb else 'DIFFERENT'}")

text = next((a, b) for a, b in zip(SA, SB) if a[1] == "__text")
(a, b) = text
n = a[3] // 4
if b[3] == a[3]:
    wa = struct.unpack(f"<{n}I", A[a[4]:a[4] + a[3]]); wb = struct.unpack(f"<{n}I", B[b[4]:b[4] + b[3]])
    def kind(w):
        if (w & 0x9F000000) == 0x90000000: return "adrp"
        if (w & 0x7F800000) == 0x11000000: return "add/sub immediate"
        if (w & 0x3B000000) == 0x39000000: return "ldr/str unsigned offset"
        if (w & 0xFC000000) in (0x94000000, 0x14000000): return "bl/b"
        return "other"
    c = collections.Counter(kind(wa[i]) for i in range(n) if wa[i] != wb[i])
    print(f"\n__text: {sum(c.values())} of {n} instruction words differ: {dict(c)}")
else:
    print("\n__text sizes differ: instruction comparison skipped")

def fatbins(data, p):
    for s in sections(p):
        if s[1] == "__nv_fatbin":
            d = data[s[4]:s[4] + s[3]]; L = []; i = 0
            while True:
                i = d.find(b"\x50\xed\x55\xba", i)
                if i < 0: break
                _, hs, fs = struct.unpack_from("<HHQ", d, i + 4)
                L.append((hs + fs, hashlib.sha1(d[i:i + hs + fs]).hexdigest()[:12])); i += hs + fs
            return L
    return []
FA, FB = fatbins(A, live), fatbins(B, tmp_b)
same = sum(1 for x, y in zip(FA, FB) if x == y)
print(f"\nembedded CUDA fatbins: live {len(FA)}, rebuilt {len(FB)}, byte-identical {same}")
for i, (x, y) in enumerate(zip(FA, FB)):
    if x != y: print(f"  fatbin #{i}: live {x[0]} B ({x[1]})  rebuilt {y[0]} B ({y[1]})")
