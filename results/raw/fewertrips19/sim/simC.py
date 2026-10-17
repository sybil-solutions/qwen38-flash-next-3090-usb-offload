#!/usr/bin/env python3
"""simC.py - idea C: cut USB round trips exactly.
 C1 dynamic zero-cold ring skip (device raises the flag itself; host drains the notification asynchronously): saves ring*zero_rings per window; whole-layer k sweep and the slot count.
 C2 ring transactions: per-ring host/USB budget from Decode10 HS_TRACE numbers; which term is on the critical path for which ring class.
"""
import numpy as np
from sim19 import *
G = [0, 1, 2, 6, 7, 10, 22, 5, 18, 13, 12, 38]
print("C1  dynamic zero-cold skip (ideal: skipped ring costs 0; ring=1.06 ms), config C/D slots 9,958, tau 2.59")
print(" k whole | N_in  M    zero-cold rings/win | W no-skip | W skip | tok/s no-skip | tok/s skip | skip gain")
for k in (4, 6, 8, 10, 12):
    wl = tuple(G[:k]); r = replay(slots=9958, wl=wl); s = summarize(r, k=k)
    Wsk = s['W'] - 1.06 * s['zero']
    print(f" {k:2d}      | {s['N']:5.1f} {s['M']:5.2f} {s['zero']:5.2f}               | {s['W']:7.2f}  | {Wsk:6.2f} | {s['toks']:6.2f}        | {1000*TAU/Wsk:6.2f}     | {100*(s['W']/Wsk-1):+5.2f}%")
print()
print("C1b zero-cold fraction by rows in the window (current placement, k=4): rows 1/2/3/4")
for rsel in (1, 2, 3, 4):
    sm = Rec(PROF, wl=WL4, lru=6400, slots=9958, nlent=884, guided=True); allw = []
    for q in REQS: allw += sm.request_trace(H[q], [w for w in windows(q) if w[0][1].shape[0] >= 4], row_sel=rsel)
    z = np.mean([w['zero'] for w in allw]); print(f"  rows {rsel}: zero-cold rings/window {z:.2f} of 44 = {100*z/44:.1f}%")
