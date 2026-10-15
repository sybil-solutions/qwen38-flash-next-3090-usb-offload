#!/usr/bin/env python3
"""mkseq18.py TAG [FLAG_A=7 FLAG_B=15 MAXTOK=300] (ON the mini): the in-server A/B sequence for run_seq.py.
Pass 1 walks the realistic prompts p01..p29 (Decode7 corpus, 3.7-19.8K tokens) with the flag alternating A,B,A,B..;
pass 2 walks them in reverse with the opposite flags, every prompt prefixed with a fresh salt line (no prefix-cache reuse).
Each flag therefore sees every prompt once, half of its requests in each pass (the pass-2 LRU warm bias cancels).
Writes ~/strata-prefill18/prompts/<TAG>_<n>.txt and prints the run_seq ITEM list (path:flag:maxtok)."""
import json, os, sys, random
tag = sys.argv[1]
fa = int(sys.argv[2]) if len(sys.argv) > 2 else 7
fb = int(sys.argv[3]) if len(sys.argv) > 3 else 15
mt = int(sys.argv[4]) if len(sys.argv) > 4 else 300
D = os.path.expanduser("~/strata-decode7/prompts")
O = os.path.expanduser("~/strata-prefill18/prompts")
os.makedirs(O, exist_ok=True)
names = sorted(f[:-5] for f in os.listdir(D) if f.endswith(".json") and f.startswith("p"))
items = []
rnd = random.Random(1815)
for ps, order in ((1, names), (2, names[::-1])):
    for i, n in enumerate(order):
        flag = (fa, fb)[(i + (ps - 1)) % 2]
        text = json.load(open(f"{D}/{n}.json"))["text"]
        path = f"{O}/{tag}_{ps}{n}.txt"
        with open(path, "w") as f:
            f.write(f"Session {rnd.randrange(10**9)}.\n\n{text}")
        items.append(f"{path}:{flag}:{mt}")
print(" ".join(items))
