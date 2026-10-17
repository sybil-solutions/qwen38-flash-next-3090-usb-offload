#!/usr/bin/env python3
"""sim19.py - FewerTrips19 simulation harness (calibrated model, Decode7 29-prompt routing dump).
W = 25.8 + 1.06*(48-k) + 0.0424*N_in + 0.64*M ms/window ; tok/s = 1000*tau/W ; tau ~ 2.59 (N_in = in-place cold experts (not misses), M = LRU misses).
Per-window records: rows, per-layer (cold, miss, zero_cold, exact cold ids) so ideas can be priced on the same replay.
Config C/D VRAM = 9,958 slots (884 guided lend), 4 whole layers 0,1,2,6, mini LRU 6,400, f512c ranking."""
import sys, os, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, 'data'); sys.path.insert(0, D)
import sim7, sim8
NL, NE, NI = 48, 512, 48 * 512
TAU = 2.59
WL4 = (0, 1, 2, 6)

def Wms(k, N_in, M, ring=1.06, a=0.0424, s=0.64, fixed=25.8):
    """N_in here = cold in-place count INCLUDING misses (sim14 convention): 0.0424*(N-M) + 0.64*M."""
    return fixed + ring * (48 - k) + a * (N_in - M) + s * M

def load_rows(p=os.path.join(D, 'route.bin')):
    b = np.fromfile(p, dtype='<i4'); o = 0; R = {}; cur = None
    while o + 3 <= len(b):
        l, n, k = int(b[o]), int(b[o + 1]), int(b[o + 2]); o += 3
        if l < 0: cur = len(R) + 1; R[cur] = []; continue
        ids = b[o:o + n * k].reshape(n, k); o += n * k
        if cur is not None: R[cur].append((l, ids.astype(np.int64) + l * NE))
    return R

R = load_rows(); H = sim7.load_hist(os.path.join(D, 'hist.bin'))
PROF = sim7.load_prof(os.path.join(D, 'f512c.bin'))
REQS = sorted(set(R) & set(H))

def windows(req):
    rec = R[req]
    return [rec[i:i + NL] for i in range(0, len(rec) - NL + 1, NL)]

class Rec(sim8.Sim):
    """sim8.Sim with per-ring tracing of cold / miss counts per window."""
    def request_trace(s, hist, wins, row_sel=None):
        """wins = list of windows (each 48 (layer, rows[n,10])).  row_sel(n)-> number of rows to use (default all).  Returns per-window list of dicts."""
        # reuse the request() preamble (lend / warm) by calling it with an empty ring list
        s.request(hist, [])
        out = []
        for w in wins:
            nrow = w[0][1].shape[0]; use = nrow if row_sel is None else min(nrow, row_sel)
            cold_l = np.zeros(NL, int); miss_l = np.zeros(NL, int); zero = 0; whole = 0
            for l, ids in w:
                if l in s.wl: whole += 1; continue
                u = np.unique(ids[:use].ravel())
                c = m = 0
                for x in u:
                    x = int(x)
                    if x in s.vr or x in s.pinned: continue
                    c += 1
                    if x in s.lru: s.lru.move_to_end(x)
                    else:
                        m += 1
                        if len(s.lru) >= s.lru_cap: s.lru.popitem(last=False)
                        s.lru[x] = 1
                cold_l[l] = c; miss_l[l] = m
                if c == 0: zero += 1
            out.append(dict(rows=use, N=int(cold_l.sum()), M=int(miss_l.sum()), zero=zero, cold_l=cold_l, miss_l=miss_l))
        return out

def replay(slots=9958, wl=WL4, lru=6400, nlent=884, guided=True, prof=None, row_sel=None, reqs=None):
    sm = Rec(prof or PROF, wl=wl, lru=lru, slots=slots, nlent=nlent, guided=guided)
    res = []
    for r in (reqs or REQS):
        res.append((r, sm.request_trace(H[r], windows(r), row_sel)))
    return res

def summarize(res, k=len(WL4), tau=TAU):
    allw = [w for _, ws in res for w in ws]
    N = np.mean([w['N'] for w in allw]); M = np.mean([w['M'] for w in allw]); Z = np.mean([w['zero'] for w in allw])
    W = Wms(k, N, M)
    return dict(N=N, M=M, zero=Z, W=W, toks=1000 * tau / W, nw=len(allw))

if __name__ == '__main__':
    for name, slots in (('config A(Decode14 9467)', 9467), ('config C/D 9958', 9958)):
        r = summarize(replay(slots=slots))
        print(f"{name}: windows {r['nw']} N {r['N']:.1f} M {r['M']:.2f} zero-cold rings/window {r['zero']:.2f} W {r['W']:.2f} ms  model {r['toks']:.2f} tok/s")
