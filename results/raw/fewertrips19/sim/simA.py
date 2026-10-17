#!/usr/bin/env python3
"""simA.py - idea A: self-speculative card-only drafting priced with the calibrated model.
 Window = [draft stage] + 44 rings (46.6 ms) + fixed card compute (25.8 incl. MTP draft 9 ms) + v(rows) (cold in-place + miss cost of the union of the window's rows, measured on the dump: 1/2/3/4 rows = 11.0/17.5/22.2/26.8 ms, +4.6 per further row).
 Card-only draft: d tokens, per-token agreement alpha (teacher-forced top-1 of the card-only model vs the exact greedy token), t_c ms per card-only token (no rings).
 tau(d) = 1 + sum_{j=1..d} alpha^j ; W = 25.8 - 9 (MTP head stage removed) + t_c*d + 46.64 + v(1+d).   Baseline: MTP spec 4 = tau 2.59 at ~3.2 rows (W 92.65 -> 27.96 model tok/s)."""
import numpy as np
V = {1: 83.45 - 72.44, 2: 89.91 - 72.44, 3: 94.63 - 72.44, 4: 99.22 - 72.44}
def v(rows): return V[rows] if rows in V else V[4] + 4.6 * (rows - 4)
BASE = 1000 * 2.59 / 92.65
def rate(alpha, tc, d, mtp_first=False, a_mtp=0.70, c_mtp=4.0):
    # optionally the first draft is the MTP head's (cost c_mtp, accept a_mtp) and the card-only chain extends it
    if mtp_first and d >= 1:
        P = [a_mtp] + [a_mtp * alpha ** j for j in range(1, d)]
        cost = 25.8 - 9 + c_mtp + tc * (d - 1)
    else:
        P = [alpha ** j for j in range(1, d + 1)]; cost = 25.8 - 9 + tc * d
    tau = 1 + sum(P)
    W = cost + 46.64 + v(1 + d)
    return 1000 * tau / W, tau, W
print(f"baseline MTP spec 4 (model): {BASE:.2f} tok/s (tau 2.59, W 92.65)\n")
print("best d (1..10) and tok/s vs card-only token agreement alpha and card-only token cost t_c (ms)")
print("alpha  | " + " | ".join(f"t_c={t:4.1f}" for t in (2, 4, 8, 13, 18)))
for a in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.98):
    cells = []
    for tc in (2, 4, 8, 13, 18):
        best = max(((rate(a, tc, d), d) for d in range(1, 11)), key=lambda z: z[0][0])
        cells.append(f"{best[0][0]:5.2f} d{best[1]:<2d}({100*(best[0][0]/BASE-1):+5.1f}%)")
    print(f"{a:5.2f}  | " + " | ".join(cells))
print("\nMTP first draft (alpha_mtp 0.70, 4 ms) then card-only extension:")
for a in (0.80, 0.85, 0.90, 0.95):
    cells = []
    for tc in (2, 4, 8, 13):
        best = max(((rate(a, tc, d, True), d) for d in range(1, 11)), key=lambda z: z[0][0])
        cells.append(f"{best[0][0]:5.2f} d{best[1]:<2d}({100*(best[0][0]/BASE-1):+5.1f}%)")
    print(f"{a:5.2f}  | " + " | ".join(cells))
print("\nbreak-even: alpha needed for +0% at t_c 13 ms:", end=' ')
for a in np.arange(0.5, 1.0, 0.005):
    if max(rate(a, 13, d)[0] for d in range(1, 11)) >= BASE: print(f"{a:.3f}"); break
else: print('none < 1.0')
print("break-even alpha at t_c 8 ms:", end=' ')
for a in np.arange(0.5, 1.0, 0.005):
    if max(rate(a, 8, d)[0] for d in range(1, 11)) >= BASE: print(f"{a:.3f}"); break
else: print('none < 1.0')
