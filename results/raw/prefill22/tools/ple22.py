#!/usr/bin/env python3
"""ple22.py FLAGSTR OUT.jsonl [reps=2]  (ON the mini, server = cfg/ab21.json, flag file ~/strata-prefill21/pfcfg.txt)
In-server A/B of the PLE first-group warm-ahead (plew=1) on FRESH text with MATCHED token counts: every request gets its own non-overlapping haystack slice (no row of the PLE table was read by an
earlier slice of this run), and the two arms of a pair get slices of exactly the same prompt token count (checked with the server's count_tokens, then again against the reply's prompt_n).
f0 = FLAGSTR (the decode-neutral shim levers), f1 = FLAGSTR + plew=1.  Order: rep 0 f0,f1  rep 1 f1,f0 (ABBA) at 8K / 18K / 33K."""
import json, sys, time, os, random, urllib.request, subprocess
HOME = os.path.expanduser("~"); D = HOME + "/strata-prefill21"
fs = sys.argv[1].replace(",", " "); OUT = sys.argv[2]; REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
CFG = HOME + "/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json"
KEY = json.load(open(CFG))["api_key"]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + KEY, "x-api-key": KEY, "anthropic-version": "2023-06-01"}
sys.path.insert(0, HOME + "/strata/tools")
import needle_bench as nb
SIZES = [8000, 18000, 33000]
ARMS = {"f0": fs, "f1": (fs + " plew=1").strip()}

def usercheck():
    r = subprocess.run(["zsh", D + "/tools/usercheck21.sh"], capture_output=True, text=True)
    if r.returncode != 0 or os.path.exists(D + "/USER_ACTIVE"): print("USER ACTIVE:", r.stdout.strip(), flush=True); sys.exit(3)
def count(text):
    b = json.dumps({"model": "x", "max_tokens": 1, "messages": [{"role": "user", "content": text}]}).encode()
    return json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/messages/count_tokens", b, H), timeout=120))["input_tokens"]
def slice_for(hay, start, target, head):
    """text hay[start:start+L] with count(head + text) == target exactly (smallest L reaching it), or None"""
    lo, hi = int(target * 2.0), int(target * 5.0)
    if start + hi > len(hay): return None, 0
    while lo < hi:
        mid = (lo + hi) // 2
        if count(head + hay[start:start + mid]) >= target: hi = mid
        else: lo = mid + 1
    return (hay[start:start + lo], lo) if count(head + hay[start:start + lo]) == target else (None, lo)

pos = int(os.getenv("PLE_START", "0")); plan = []
hay = nb.haystack(pos + 900_000)
for rep in range(REPS):
    for n in SIZES:
        order = ["f0", "f1"] if rep % 2 == 0 else ["f1", "f0"]
        t0 = None; item = {}
        for arm in order:
            head = f"Session {random.randrange(10**9)}. Here are some notes. Reply with OK only.\n\n"
            tgt = n if t0 is None else t0
            for _try in range(40):
                body, L = slice_for(hay, pos, tgt, head)
                if body is not None: break
                pos += 1   # the tokenizer merged across the cut: shift the start by a character
            else:
                # the first arm of a pair may reach any count near the target; the second must hit the first's exactly (retry a few starts)
                raise SystemExit(f"no slice for {tgt}")
            if t0 is None: t0 = tgt
            pos += L + 1000
            plan.append((rep, n, arm, head + body, tgt))
print("slices:", len(plan), "hay used", pos, "of", len(hay), flush=True)
if pos > len(hay): print("WARNING: haystack wrapped?", flush=True)
for rep, n, arm, text, tgt in plan:
    usercheck()
    open(D + "/pfcfg.txt", "w").write(ARMS[arm] + "\n"); time.sleep(0.45)
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 1, "reasoning_effort": "none", "temperature": 0}).encode()
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, H), timeout=7200))
    tm = r["timings"]
    row = {"arm": arm, "size": n, "rep": rep, "count_tokens": tgt, "prompt_n": tm["prompt_n"], "cache_n": tm.get("cache_n"), "prompt_ms": round(tm["prompt_ms"]), "tok_s": round(tm["prompt_n"] / tm["prompt_ms"] * 1000, 1),
           "flags": ARMS[arm], "ts": time.strftime("%FT%TZ", time.gmtime())}
    print(json.dumps(row), flush=True); open(OUT, "a").write(json.dumps(row) + "\n")
    subprocess.run(["zsh", D + "/tools/mark21.sh"]); time.sleep(1.5)
open(D + "/pfcfg.txt", "w").write("\n")
import statistics as st
print("tok/s by size (arm: values -> mean)")
for n in SIZES:
    for arm in ("f0", "f1"):
        v = [json.loads(l)["tok_s"] for l in open(OUT) if json.loads(l)["size"] == n and json.loads(l)["arm"] == arm]
        print(f"  {n:6d} {arm}: {v} mean {st.mean(v):.1f}")
print("PLE22DONE", flush=True)
