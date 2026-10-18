#!/usr/bin/env python3
"""stress21.py [iters] [seed] (ON the mini, server DOWN): byte-exact h2d/d2h stress through the shim's transfer paths, random sizes/offsets, host data patterns; prints the first mismatch and a summary."""
import os, sys, time, random
os.environ["TGCU_PROF"] = "1"; os.environ.pop("TGCU_PROF_USB", None)
import tgcu_api
tgcu_api.init()
ctx = tgcu_api.ctx; ace = ctx.ace
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 200
rnd = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
BUF = 24 << 20
va = tgcu_api.malloc(BUF)
sizes_pool = [4, 8, 60, 1000, 4096, 16380, 32768, 40000, 65536, 100000, 111104, 196096, 196608, 200000, 392192, 409600, 500000, 983040, 1048576, 1300000, 2000000, 5439488, 5439492, 8 << 20]
bad = 0; t0 = time.perf_counter(); moved = 0
maxs = float(os.getenv("STRESS_SECS", "0"))
for it in range(iters):
    if maxs and time.perf_counter() - t0 > maxs: iters = it; break
    n = rnd.choice(sizes_pool) if rnd.random() < 0.7 else rnd.randrange(1, 3 << 20) & ~3
    n = max(4, n & ~3)
    off = rnd.randrange(0, 4096) & ~3
    kind = rnd.choice(["rand", "ramp", "zero", "ff"])
    if kind == "rand": data = os.urandom(n)
    elif kind == "ramp": data = bytes((i * 7 + it) & 255 for i in range(min(n, 4096))) * (n // min(n, 4096) + 1)
    elif kind == "zero": data = bytes(n)
    else: data = b"\xff" * n
    data = data[:n]
    # poison the destination first so a short transfer cannot pass
    ace.h2d(va + off, bytes(rnd.randrange(256) for _ in range(1)) * n if n < 70000 else b"\xa5" * n)
    ace.h2d(va + off, data)
    back = ace.d2h(va + off, n)
    moved += 2 * n
    if bytes(back) != data:
        bad += 1; i = next(i for i in range(n) if back[i] != data[i])
        print(f"MISMATCH it={it} n={n} off={off} kind={kind} first diff at {i}: {back[i]} vs {data[i]}", flush=True)
        if bad >= 3: break
dt = time.perf_counter() - t0
print(f"stress21: {iters} iterations, {moved / 1e6:.0f} MB moved in {dt:.1f} s, mismatches {bad}", flush=True)
