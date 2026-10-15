#!/usr/bin/env python3
"""sum18.py RUN.jsonl [PROBE.log]: per-flag means of a dec18.sh arm (requests = run_seq records: flag, prompt_n, ttft_s, n, tok_s, wall_s)."""
import json, sys, statistics as st
r = [json.loads(l) for l in open(sys.argv[1])]
for fl in sorted({x["flag"] for x in r}):
    q = [x for x in r if x["flag"] == fl]
    print("flag %-3d n %2d | decode tok/s %.2f | ttft %.2f s | wall %.2f s | prompt tok/s(ttft) %.1f" % (
        fl, len(q), st.mean(x["tok_s"] for x in q), st.mean(x["ttft_s"] for x in q), st.mean(x["wall_s"] for x in q),
        st.mean(x["prompt_n"] / x["ttft_s"] for x in q if x["ttft_s"] > 0)))
byname = {}
for x in r: byname.setdefault(x["name"].split("/")[-1][len(x["name"].split("/")[-1].split("_")[0]) + 2:], {})[x["flag"]] = x
fl = sorted({x["flag"] for x in r})
if len(fl) == 2:
    d = [(v[fl[1]]["tok_s"] / v[fl[0]]["tok_s"] - 1, v[fl[1]]["wall_s"] - v[fl[0]]["wall_s"]) for v in byname.values() if len(v) == 2]
    if d: print("paired (flag %d vs %d): decode tok/s %+.1f%% mean, wall %+.2f s mean, n %d" % (fl[1], fl[0], 100 * st.mean(a for a, b in d), st.mean(b for a, b in d), len(d)))
if len(sys.argv) > 2:
    v = [float(l.split()[2]) for l in open(sys.argv[2]) if l.startswith("decode tok/s")]
    if v: print("probes mean %.2f (%d)" % (st.mean(v), len(v)))
