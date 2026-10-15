#!/usr/bin/env python3
"""sim_batch.py - Decode15 concurrency model: replay the Decode7 29-prompt decode dump (route.bin keeps the ids of every ROW of every verify window) as c concurrent requests in the engine's batch slots.
A batch window = one row per slot (the slot's current token = row 0 of its own solo window j: the verified token), or with --batch-mtp two rows per slot (row 0 + the first draft: row 1).
Per layer the window's experts are the UNION over the slots' rows; the VRAM static set + guided lend + mini LRU run as in sim14.  W = 25.8 + 1.06*(48-k) + 0.0424*N_in + 0.64*M ms/window (the calibrated model)."""
import numpy as np, sys, collections
sys.path.insert(0, '/tmp/d14')
import sim7, sim8
NL, NE = 48, 512
def load_rows(p):
    b = np.fromfile(p, dtype='<i4'); o = 0; R = {}; cur = None
    while o + 3 <= len(b):
        l, n, k = int(b[o]), int(b[o + 1]), int(b[o + 2]); o += 3
        if l < 0: cur = len(R) + 1; R[cur] = []; continue
        ids = b[o:o + n * k].reshape(n, k); o += n * k
        if cur is not None: R[cur].append((l, [(r.astype(np.int64) + l * NE) for r in ids]))
    return R
R = load_rows('/tmp/d14/route.bin'); H = sim7.load_hist('/tmp/d14/hist.bin')
prof = sim7.load_prof('/tmp/d14/f512c.bin')
WL = (0, 1, 2, 6); K = len(WL)
def windows(req):   # -> list of windows, each a list of 48 (layer, rows)
    rec = R[req]; out = []
    for i in range(0, len(rec) - NL + 1, NL):
        out.append(rec[i:i + NL])
    return out
reqs = sorted(set(R) & set(H))
rows_stat = collections.Counter(len(r[1]) for q in reqs for r in R[q][:480])
print('rows per solo window (first 10 windows of every request):', dict(rows_stat), ' mean ids per layer-window solo: %.1f' % np.mean([len(np.unique(np.concatenate(r[1]))) for q in reqs for r in R[q][:480]]))
def W(N_in, M): return 25.8 + 1.06 * (48 - K) + 0.0424 * (N_in - M) + 0.64 * M     # sim14's W(k, N, M): N_in = cold in place (incl. misses), 0.0424*(N-M) + 0.64*M
def run(c, rows_per_slot, tok_per_window, slots, groups=None, lru=6400, nlent=884):
    """c concurrent requests (consecutive requests of the realistic set), their windows stepped in lockstep until the shortest ends."""
    res = []
    order = reqs
    for g0 in range(0, len(order) - c + 1, c):
        grp = order[g0:g0 + c]
        sm = sim8.Sim(prof, wl=WL, lru=lru, slots=slots, nlent=nlent, guided=True)
        # the lend / LRU warm of the group: the guided refill follows the FIRST request's prompt (the engine lends once per admission wave)
        sm_hist = H[grp[0]]
        wins = [windows(q) for q in grp]
        nw = min(len(w) for w in wins)
        rings = []
        for j in range(nw):
            for l in range(NL):
                ids = []
                for s in range(c):
                    rows = wins[s][j][l][1]
                    for r in range(min(rows_per_slot, len(rows))): ids.extend(int(x) for x in rows[r])
                rings.append((l, ids))
        o = sm.request(sm_hist, rings)
        w_ms = []
        mw = o['M'] / o['windows']; nw_ = o['N'] / o['windows']
        Wm = 25.8 + 1.06 * (48 - K) + 0.0424 * (nw_ - mw) + 0.64 * mw
        res.append((Wm, nw_, mw))
    a = np.array(res)
    Wm = a[:, 0].mean()
    per_req = 1000 * tok_per_window / Wm
    return dict(W=Wm, N=a[:, 1].mean(), M=a[:, 2].mean(), per_req=per_req, agg=c * per_req)
if __name__ == '__main__':
    base_slots = 8393 + 1074
    print('solo MTP spec 4 (the live c=1 path): model per-request tok/s with tau=2.59:')
    # solo: all rows of the window, tau 2.59 tokens / window
    solo = []
    sm = sim8.Sim(prof, wl=WL, lru=6400, slots=base_slots, nlent=884, guided=True)
    for q in reqs:
        rings = [(l, [int(x) for r in rows for x in r]) for w in windows(q) for (l, rows) in w]
        o = sm.request(H[q], rings); mw = o['M'] / o['windows']; nw = o['N'] / o['windows']
        solo.append((25.8 + 1.06 * (48 - K) + 0.0424 * (nw - mw) + 0.64 * mw, nw, mw))
    a = np.array(solo); print('  W %.2f ms N %.1f M %.2f -> %.2f tok/s' % (a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), 1000 * 2.59 / a[:, 0].mean()))
    main_extra = 340   # the main session's K/V + pooled + rope at 40,960 cells without the elastic K/V (0.58 GiB vs 0.24 GiB elastic at 16K)
    for (c, mtp, per_slot_mib) in [(1, False, 0), (2, False, 577), (4, False, 577), (8, False, 577), (2, True, 618), (4, True, 618), (8, True, 618)]:
        slots = int(base_slots - (c * per_slot_mib + (main_extra if c > 1 else 0)) / 1.776)     # the slot sessions' VRAM comes out of the expert cache (elastic K/V extended to the slot sessions, 16K context)
        rps = 2 if mtp else 1; tpw = 1.54 if mtp else 1.0
        r = run(c, rps, tpw, slots)
        print(f'  c={c} {"batch-mtp" if mtp else "no draft "} slots {slots}: W {r["W"]:.1f} ms N {r["N"]:.0f} M {r["M"]:.1f}  per-request {r["per_req"]:.1f} tok/s  aggregate {r["agg"]:.1f} tok/s')
