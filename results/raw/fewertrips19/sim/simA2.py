#!/usr/bin/env python3
"""simA2.py - self-speculative card-only chain + exact verify with PREFETCH of the experts the chain routed to (idea A + D1).
Per window: chain of d card-only tokens (t_c ms each, no rings, alpha per token = 0.923 measured), then ONE exact window of 1+d rows whose cold experts were prefetched from the chain's routing
(misses x mu, mu = 0.30 measured by simD.py on the teacher-forced route dumps: 32.9 -> 10.0 misses/window).  The MTP head stage (9 ms) is dropped (chain replaces it).
W = (25.8 - 9) + t_c*d + 46.64 + 0.0424*(N_in - M*mu) + 0.64*M*mu ; N_in, M from the dump replay by rows (N: 97.3/166.8/228.3/285.9, M: 11.52/17.39/20.93/24.53, +58 / +3.6 per row beyond 4)."""
import sys, numpy as np
NR = {1: 97.3, 2: 166.8, 3: 228.3, 4: 285.9}; MR = {1: 11.52, 2: 17.39, 3: 20.93, 4: 24.53}
def nm(rows): return (NR[rows], MR[rows]) if rows in NR else (NR[4] + 58 * (rows - 4), MR[4] + 3.6 * (rows - 4))
def v(rows, mu=1.0):
    N, M = nm(rows); return 0.0424 * (N - M * mu) + 0.64 * M * mu
BASE = 1000 * 2.59 / 92.65
def rate(alpha, tc, d, mu, keep_mtp=False, extra_ms=0.0):
    tau = 1 + sum(alpha ** j for j in range(1, d + 1))
    W = (25.8 if keep_mtp else 25.8 - 9) + tc * d + 46.64 + v(1 + d, mu) + extra_ms
    return 1000 * tau / W, tau, W
if __name__ == '__main__':
    print(f'baseline {BASE:.2f} tok/s model. alpha 0.923 (teacher-forced, skip-cold), mu = miss fraction left after prefetch')
    for mu in (1.0, 0.5, 0.30):
        print(f'\nmu = {mu}: best d and tok/s (gain vs MTP baseline) by card-only token cost t_c')
        print('t_c ms |' + '|'.join(f' alpha={a:.2f}      ' for a in (0.85, 0.90, 0.923, 0.95)))
        for tc in (2, 4, 6, 8, 10, 13, 16):
            cells = []
            for a in (0.85, 0.90, 0.923, 0.95):
                best = max(((rate(a, tc, d, mu), d) for d in range(1, 8)), key=lambda z: z[0][0])
                cells.append(f' {best[0][0]:5.2f} d{best[1]}({100*(best[0][0]/BASE-1):+5.1f}%)')
            print(f'{tc:5d}  |' + '|'.join(cells))
    print('\nbreak-even t_c at alpha 0.923 (best d):')
    for mu in (1.0, 0.5, 0.30):
        for tc in np.arange(1, 30, 0.25):
            if max(rate(0.923, tc, d, mu)[0] for d in range(1, 8)) < BASE: print(f'  mu {mu}: t_c <= {tc-0.25:.2f} ms pays'); break
