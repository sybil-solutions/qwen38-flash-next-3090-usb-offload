#!/usr/bin/env python3
"""routecmp19.py REF.bin APPROX.bin [LRU-sim options]: align the per-request ring records of two STRATA_DEC_ROUTE_DUMP files taken on the SAME teacher-forced token streams
(exact run vs card-only run) and report, per layer depth, how well the card-only pass predicts the exact routing:
  recall of the exact top-K ids by the card-only ids (per row), and the same restricted to the experts that are NOT VRAM-resident (the ones a prefetch would need).
Segments (requests) whose ring sequences differ in shape (free-running replies) are skipped."""
import sys, numpy as np, collections
NE = 512
def load(p):
    b = np.fromfile(p, dtype='<i4'); o = 0; segs = []; cur = None
    while o + 3 <= len(b):
        l, n, k = int(b[o]), int(b[o + 1]), int(b[o + 2]); o += 3
        if l < 0: cur = []; segs.append(cur); continue
        if o + n * k > len(b): break
        ids = b[o:o + n * k].reshape(n, k); o += n * k
        if cur is not None: cur.append((l, ids))
    return segs
def main(a, b, prof=None, nres=None):
    A = load(a); B = load(b); print('segments', len(A), len(B))
    res = None
    if prof:
        import struct
        raw = open(prof, 'rb').read(); ver, nl, ne, sl, nn = struct.unpack_from('<5I', raw, 4)
        pr = [(int(l), int(e)) for l, e in np.frombuffer(raw, dtype='<u2', count=2 * nn, offset=24).reshape(-1, 2)]
        res = set(pr[:nres])   # first nres of the ranking are VRAM-resident (approximation of the placement)
    rec = collections.defaultdict(lambda: [0, 0, 0, 0]); used = 0
    for sa, sb in zip(A, B):
        if len(sa) != len(sb) or len(sa) == 0: continue
        if any(x[0] != y[0] or x[1].shape != y[1].shape for x, y in zip(sa, sb)): continue
        used += 1
        for (l, ia), (_, ib) in zip(sa, sb):
            for r in range(ia.shape[0]):
                ea = set(int(x) for x in ia[r]); eb = set(int(x) for x in ib[r])
                rec[l][0] += len(ea); rec[l][1] += len(ea & eb)
                if res is not None:
                    cold = {e for e in ea if (l, e) not in res}
                    rec[l][2] += len(cold); rec[l][3] += len(cold & eb)
    print('segments compared', used)
    tot = np.zeros(4)
    print('layer  recall(all)  recall(cold only)')
    for l in sorted(rec):
        r = rec[l]; tot += r
        if l % 4 == 0: print(f'{l:3d}   {r[1]/max(r[0],1):.3f}      {r[3]/max(r[2],1):.3f}   (n={r[0]})')
    print(f'ALL   {tot[1]/max(tot[0],1):.3f}      {tot[3]/max(tot[2],1):.3f}')
    for lo, hi in ((0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48)):
        s = np.sum([rec[l] for l in range(lo, hi) if l in rec], axis=0)
        print(f'layers {lo:2d}-{hi-1:2d}: recall {s[1]/max(s[0],1):.3f}  cold-recall {s[3]/max(s[2],1):.3f}')
if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, int(sys.argv[4]) if len(sys.argv) > 4 else None)
