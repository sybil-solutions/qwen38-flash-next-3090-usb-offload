#!/usr/bin/env python3
"""simD.py - exact-pass prefetch driven by the card-only pass's routing (self-speculative look-ahead).
Replays the teacher-forced route dumps (logs/route_d19ref.bin = exact rings, logs/route_d19skip.bin = card-only (skip-cold, route normally) rings of the SAME token streams, window by window)
through the VRAM set + mini LRU of sim8: before each exact window, every expert the card-only pass routed to (cold, not in the LRU) is read into the LRU (speculative prefetch, SSD reads counted),
then the exact window runs (misses counted).  M saved x 0.64 ms = the critical-path saving; SSD reads/window must fit the window (2.2 GB/s = 1.18 blobs/ms)."""
import sys, os, numpy as np, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
import sim7, sim8
NL, NE = 48, 512
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
def load(p):
    b = np.fromfile(p, dtype='<i4'); o = 0; segs = []; cur = None
    while o + 3 <= len(b):
        l, n, k = int(b[o]), int(b[o + 1]), int(b[o + 2]); o += 3
        if l < 0: cur = []; segs.append(cur); continue
        if o + n * k > len(b): break
        ids = b[o:o + n * k].reshape(n, k).astype(np.int64); o += n * k
        if cur is not None: cur.append((l, ids + l * NE))
    return segs
A = load(f'{LOG}/route_d19ref.bin'); B = load(f'{LOG}/route_d19skip.bin')
prof = sim7.load_prof(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'f512c.bin'))
def windows(seg):
    out = []; cur = []
    for l, ids in seg:
        cur.append((l, ids))
        if l == 47: out.append(cur); cur = []
    return out
pairs = []
for sa, sb in zip(A, B):
    if len(sa) == 0 or len(sa) != len(sb): continue
    if any(x[0] != y[0] or x[1].shape != y[1].shape for x, y in zip(sa, sb)): continue
    pairs.append((windows(sa), windows(sb)))
print('segments', len(pairs), 'windows', sum(len(a) for a, b in pairs))
def replay(slots=9958, lru=6400, wl=(0, 1, 2, 6), mode='none', rho_extra=0.0, rng=np.random.default_rng(0)):
    sm = sim8.Sim(prof, wl=wl, lru=lru, slots=slots, nlent=884, guided=False)
    M = Nc = pf = 0; nw = 0; hit_pred = 0; tot_miss = 0
    for rep in range(3):          # three passes over the segments: the first warms the LRU state, only the last pass is scored
        M = Nc = pf = nw = hit_pred = tot_miss = 0
        for wa, wb in pairs:
            for xa, xb in zip(wa, wb):
                if mode == 'prefetch':
                    for (l, ids) in xb:
                        if l in sm.wl: continue
                        for x in np.unique(ids.ravel()):
                            x = int(x)
                            if x in sm.vr or x in sm.pinned: continue
                            if x in sm.lru: sm.lru.move_to_end(x)
                            else:
                                pf += 1
                                if len(sm.lru) >= sm.lru_cap: sm.lru.popitem(last=False)
                                sm.lru[x] = 1
                for (l, ids) in xa:
                    if l in sm.wl: continue
                    for x in np.unique(ids.ravel()):
                        x = int(x)
                        if x in sm.vr or x in sm.pinned: continue
                        Nc += 1
                        if x in sm.lru: sm.lru.move_to_end(x)
                        else:
                            M += 1
                            if len(sm.lru) >= sm.lru_cap: sm.lru.popitem(last=False)
                            sm.lru[x] = 1
                nw += 1
    return M / nw, Nc / nw, pf / nw
for mode in ('none', 'prefetch'):
    M, N, pf = replay(mode=mode)
    print(f'{mode:9s}: misses/window {M:5.2f}  cold in place/window {N:6.1f}  speculative SSD reads/window {pf:5.2f}   (window = 4-6 rows teacher-forced)')
