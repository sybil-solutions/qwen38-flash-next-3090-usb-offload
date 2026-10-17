#!/usr/bin/env python3
"""simB.py - idea B: contiguous layer blocks run ENTIRELY on the mini (attention/GDN + router + all routed experts from the mini LRU + SSD; only the residual crosses USB at block boundaries).
VRAM slots of the block layers are re-spent on the other layers (profile order); the block layers' experts are served by the LRU / SSD (misses counted by the same replay).
Cost model per block layer (ms/window): saves ring 1.06 (no doorbell) + card compute C_CARD (the card's dense time of that layer, ~0.30 ms of the 25.8 fixed) ; pays T_DENSE (dense attn/GDN/router/shared/hc weights streamed on the M4: bytes/bandwidth + dispatch) ;
+ 0.0424 per in-place cold expert + 0.64 per miss (same calibration) ; + 2 x 1.06 per block (boundary: x pull + y push = one ring each way, the card still hands over the residual)."""
import sys, numpy as np, collections
from sim19 import *

DENSE_MB_PER_LAYER = 2019/48 + 1200/48 + 120/48   # native projections + hc + router, MiB per layer (census, Decode15)
BW = 100.0   # GB/s practical M4 (120 GB/s peak)
def t_dense(eff, launch_ms=0.25):  # ms per layer: bytes / (BW*eff) + dispatch floor
    return DENSE_MB_PER_LAYER*1.0486/1000/ (BW*eff) * 1000 + launch_ms

def run_block(blk, slots=9958, wl=WL4, lru=6400):
    blk = set(blk)
    prof = [x for x in PROF if x//NE not in blk] + [x for x in PROF if x//NE in blk]   # block layers' experts last = not resident
    sm = Rec(prof, wl=wl, lru=lru, slots=slots, nlent=884, guided=True)
    allw = []
    for q in REQS: allw += sm.request_trace(H[q], windows(q))
    # block layers: all experts cold; N counts them as cold in place; per-layer arrays give the split
    N = np.mean([w['N'] for w in allw]); M = np.mean([w['M'] for w in allw])
    Nb = np.mean([sum(w['cold_l'][l] for l in blk) for w in allw]) if blk else 0
    Mb = np.mean([sum(w['miss_l'][l] for l in blk) for w in allw]) if blk else 0
    return N, M, Nb, Mb

if __name__ == '__main__':
    base = run_block([]); W0 = Wms(4, base[0], base[1])
    print(f"baseline (config C/D): N {base[0]:.1f} M {base[1]:.2f} W {W0:.2f} ms -> {1000*TAU/W0:.2f} tok/s")
    # candidate blocks: the layers with the fewest VRAM-resident experts (least ring-saving value is irrelevant: pick lowest-resident = cheapest to give up)
    res = collections.Counter(x//NE for x in PROF[:9958-884])
    order = sorted([l for l in range(NL) if l not in WL4], key=lambda l: res[l])
    print('lowest-resident layers:', [(l, res[l]) for l in order[:10]])
    for eff, lab in ((1.0, 'bandwidth-perfect'), (0.6, 'realistic 60%'), (0.35, 'pessimistic 35%')):
        td = t_dense(eff)
        print(f"\nT_DENSE per mini layer = {td:.2f} ms ({lab}; {DENSE_MB_PER_LAYER:.0f} MiB/layer at {BW*eff:.0f} GB/s + 0.25 ms dispatch)")
        print(' blk layers | dN dM | ring+card saved | dense paid | cold paid | boundary | net dW ms | tok/s | vs base')
        for nb in (1, 2, 4, 6, 8):
            blk = order[:nb]            # (non-contiguous allowed for the lower bound; contiguous blocks only add boundary cost)
            N, M, Nb, Mb = run_block(blk)
            saved = nb*(1.06+0.30); dense = nb*td
            cold = 0.0424*(N-M-(base[0]-base[1])) + 0.64*(M-base[1])
            bound = 2*1.06
            dW = -saved + dense + cold + bound
            W = W0 + dW
            print(f' {nb:2d} {str(blk):26s}| {N-base[0]:+6.1f} {M-base[1]:+5.2f} | {-saved:+7.2f} | {dense:+7.2f} | {cold:+7.2f} | {bound:+5.2f} | {dW:+7.2f} | {1000*TAU/W:5.2f} | {100*(W0/W-1):+6.2f}%')
