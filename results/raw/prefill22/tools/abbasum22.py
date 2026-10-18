#!/usr/bin/env python3
"""abbasum22.py LOGDIR TAGS...: per-arm decode (realistic mean, probes mean), prefill rows mean per size, agent-turn rows; paired E-vs-C decode over the 29 prompts.  Tags 'E1','C1','C2','E2' -> group by first letter."""
import json, sys, re, statistics as st, collections
d = sys.argv[1]; tags = sys.argv[2:]
seq = collections.defaultdict(lambda: collections.defaultdict(list)); probe = collections.defaultdict(list); rows = collections.defaultdict(lambda: collections.defaultdict(list)); small = collections.defaultdict(lambda: collections.defaultdict(list))
for t in tags:
    g = t[0]
    for l in open(f"{d}/abba_{t}.jsonl"):
        r = json.loads(l); seq[g][r["name"]].append(r["tok_s"])
    for l in open(f"{d}/abba_{t}.probe.log"):
        m = re.search(r"decode tok/s ([0-9.]+)", l)
        if m: probe[g].append(float(m.group(1)))
    for l in open(f"{d}/abba_{t}.rows.log"):
        r = json.loads(l); rows[g][r["target"]].append((r["tok_s"], r["prompt_n"], r["prompt_ms"]))
    for l in open(f"{d}/abba_{t}.small.log"):
        if l.startswith("{"):
            r = json.loads(l); small[g][r["want_new"]].append((r["read"], r["prompt_ms"], r["wall_s"]))
        elif l.startswith("base"): small[g]["base"].append((l.split()[1], float(l.split()[3]) if len(l.split())>3 else 0))
for g in sorted(seq):
    allv = [x for v in seq[g].values() for x in v]
    print(f"{g}: realistic {st.mean(allv):.2f} tok/s (n={len(allv)}), probes {st.mean(probe[g]):.2f} (n={len(probe[g])})")
if "E" in seq and "C" in seq:
    dd = [st.mean(seq["C"][n]) / st.mean(seq["E"][n]) - 1 for n in seq["C"] if n in seq["E"]]
    print(f"decode C vs E paired over {len(dd)} prompts: {100*st.mean(dd):+.2f}% (se {100*st.stdev(dd)/len(dd)**0.5:.2f}, C better in {sum(1 for x in dd if x>=0)}/{len(dd)})")
    pp = st.mean(probe["C"]) / st.mean(probe["E"]) - 1
    print(f"probes C vs E {100*pp:+.2f}%")
print("prefill rows (tok/s of the engine's prompt_ms, per arm; each arm's values):")
for n in sorted({k for g in rows for k in rows[g]}):
    print(f"  {n:6d}: " + "  ".join(f"{g}: {' '.join(str(round(x[0])) for x in rows[g][n])} (mean {st.mean(x[0] for x in rows[g][n]):.0f}, {rows[g][n][0][1]} tok)" for g in sorted(rows)))
print("agent turns (new tokens: read / prompt_ms / wall_s):")
for n in sorted({k for g in small for k in small[g] if k != "base"}):
    print(f"  {n:5}: " + "  ".join(f"{g}: " + " ".join(f"{x[0]}/{x[1]}/{x[2]}" for x in small[g][n]) for g in sorted(small)))
