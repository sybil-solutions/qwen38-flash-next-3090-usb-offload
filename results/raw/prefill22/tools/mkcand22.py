#!/usr/bin/env python3
"""mkcand22.py OUT_TAG  (ON the mini): decide whether plew=1 goes into the candidate from the matched-token-count PLE repeats (logs/ple22.jsonl + ple22b.jsonl: f1 = plew vs f0), print the decision and the mkcfg21 env list.
plew goes in when the mean f1/f0 ratio over the three sizes is >= +0.5% and no size is worse than -3% (pooled over all repeats)."""
import json, statistics as st, os, collections
D = os.path.expanduser("~/strata-prefill21/logs")
rows = [json.loads(l) for f in ("ple22.jsonl", "ple22b.jsonl") if os.path.exists(f"{D}/{f}") for l in open(f"{D}/{f}")]
by = collections.defaultdict(list)
for r in rows: by[(r["size"], r["arm"])].append(r["tok_s"])
rat = {}
for n in sorted({r["size"] for r in rows}):
    a, b = st.mean(by[(n, "f0")]), st.mean(by[(n, "f1")]); rat[n] = b / a - 1
    print(f"PLE pooled {n}: f0 {a:.1f} ({len(by[(n,'f0')])}) f1 {b:.1f} ({len(by[(n,'f1')])}) {100*rat[n]:+.1f}%")
m = st.mean(rat.values()); ok = m >= 0.005 and min(rat.values()) >= -0.03
print(f"PLEW {'IN' if ok else 'OUT'}: mean ratio {100*m:+.2f}%, worst {100*min(rat.values()):+.2f}%")
open(os.path.expanduser("~/strata-prefill21/logs/plew_decision.txt"), "w").write("1\n" if ok else "0\n")
