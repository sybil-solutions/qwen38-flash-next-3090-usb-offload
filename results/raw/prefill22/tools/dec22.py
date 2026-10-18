#!/usr/bin/env python3
"""dec22.py OUT.jsonl  (ON the mini, server = cfg/ab21.json: strata_p21b, TGCU_P21_FLAG / STRATA_PF_CFG = ~/strata-prefill21/pfcfg.txt)

Decode neutrality of the Prefill21 levers, in-server: every request carries its own arm (the flag file is written before it), prompts get a unique "[ref N]" head so no request is a prefix-cache hit
(a cache hit would skip the lazy loan and flatter whichever arm came second).  Order per prompt alternates (A B / B A) so the pairs are balanced.
  stage 1: ALL (pool=3,npipe=1,subp=1,fuse=1,npmin=32768) vs BASE ('') over the 29 realistic prompts (300 tokens) + 24 short probes (12 per arm, ABBA blocks)
  stage 2 (only when ALL is below BASE by more than TOL): bisect np / sf / pool each against BASE (29 prompts, interleaved), keep the arms that are neutral
Writes logs/flagstr22.txt = the decode-neutral flag string (k=v,k=v) for the later stages."""
import json, sys, time, os, random, urllib.request, hashlib, subprocess, statistics as st
OUT = sys.argv[1]; HOME = os.path.expanduser("~"); D = HOME + "/strata-prefill21"
CFG = HOME + "/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json"
H = {"Content-Type": "application/json", "Authorization": "Bearer " + json.load(open(CFG))["api_key"]}
ARMF = D + "/pfcfg.txt"
ARMS = {"base": "", "all": "pool=3 npipe=1 npmin=32768 subp=1 fuse=1", "np": "npipe=1 npmin=32768", "sf": "subp=1 fuse=1", "pool": "pool=3"}
SEQ = [l.split(":")[0] for l in open(HOME + "/strata-decode7/seq_on.txt").read().split()]
TOL = 0.5   # percent: a lever set is decode-neutral when its paired mean is not below BASE by more than this
PROBES = ["Write a Python function is_prime(n) with a docstring, then explain it.", "Explain how a mixture-of-experts layer routes tokens, in three paragraphs.", "Write a short story about a lighthouse keeper, about 200 words."]

def usercheck():
    r = subprocess.run(["zsh", D + "/tools/usercheck21.sh"], capture_output=True, text=True)
    if r.returncode != 0 or os.path.exists(D + "/USER_ACTIVE"):
        print("USER ACTIVE:", r.stdout.strip(), flush=True); sys.exit(3)
def mark(): subprocess.run(["zsh", D + "/tools/mark21.sh"])

def chat(text, mt, arm):
    open(ARMF, "w").write(ARMS[arm] + "\n"); time.sleep(0.45)
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": f"[ref {random.randrange(10**6):06d}] " + text}], "max_tokens": mt, "temperature": 0, "stream": True,
                       "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time(); ttft = None; txt = []; tm = {}
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, H), timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            j = json.loads(line[5:])
            if j.get("timings"): tm = j["timings"]
            for c in j.get("choices", []):
                d = c.get("delta", {}).get("content")
                if d:
                    if ttft is None: ttft = time.time() - t0
                    txt.append(d)
    return {"prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"), "ttft_s": round(ttft or -1, 2), "n": tm.get("predicted_n"), "tok_s": round(tm.get("predicted_per_second", 0), 2),
            "wall_s": round(time.time() - t0, 1), "md5": hashlib.md5("".join(txt).encode()).hexdigest()[:8]}

rows = []
def run(kind, name, text, mt, arm):
    r = chat(text, mt, arm); r.update(kind=kind, name=name, arm=arm, ts=time.strftime("%FT%TZ", time.gmtime()))
    rows.append(r); open(OUT, "a").write(json.dumps(r) + "\n"); print(json.dumps(r), flush=True); return r

def pct(kind, a, b):
    """paired mean of a vs b over the names both ran (per-name mean per arm), returns (pct, se, n, wins)"""
    by = {}
    for r in rows:
        if r["kind"] == kind and r["tok_s"]: by.setdefault((r["name"], r["arm"]), []).append(r["tok_s"])
    names = sorted({n for n, _ in by})
    dd = [st.mean(by[(n, a)]) / st.mean(by[(n, b)]) - 1 for n in names if (n, a) in by and (n, b) in by]
    if len(dd) < 2: return 0.0, 0.0, len(dd), 0
    return 100 * st.mean(dd), 100 * st.stdev(dd) / len(dd) ** 0.5, len(dd), sum(1 for x in dd if x >= 0)
def mean(kind, arm):
    v = [r["tok_s"] for r in rows if r["kind"] == kind and r["arm"] == arm and r["tok_s"]]
    return (st.mean(v), len(v)) if v else (0, 0)

def realistic(a, b, tag):
    for i, p in enumerate(SEQ):
        usercheck()
        text = json.load(open(f"{HOME}/strata-decode7/prompts/{p}.json"))["text"]
        for arm in ((a, b) if i % 2 == 0 else (b, a)): run("real", p, text, 300, arm)
        mark()
def report(tag, a, b):
    for kind in ("real", "probe"):
        ma, na = mean(kind, a); mb, nb = mean(kind, b)
        if na and nb:
            p, se, n, w = pct(kind, a, b)
            print(f"RESULT {tag} {kind}: {a} {ma:.2f} (n={na}) vs {b} {mb:.2f} (n={nb}): {p:+.2f}% paired over {n} (se {se:.2f}, {a} better in {w}/{n})", flush=True)

usercheck()
# warm-up (discarded): fills the lazy loan / LRU the way the first requests of any arm would
for p in SEQ[:2]:
    chat(json.load(open(f"{HOME}/strata-decode7/prompts/{p}.json"))["text"], 300, "base")
mark()
realistic("base", "all", "s1")
for blk, arm in enumerate(["base", "all", "all", "base"]):
    usercheck()
    for rep in range(2):
        for k, p in enumerate(PROBES): run("probe", f"{k}", p + f" (run {blk}.{rep})", 300, arm)
    mark()
# probes: pair by prompt index (names 0..2 run in 4 blocks per arm -> per-name mean per arm)
report("stage1", "all", "base")
pa, sea, na, wa = pct("real", "all", "base"); pp, sep, npr, _ = pct("probe", "all", "base")
keep = []
if pa >= -TOL:
    keep = ["all"]; print("STAGE1: ALL is decode-neutral on the realistic set", flush=True)
else:
    print(f"STAGE1: ALL regresses ({pa:+.2f}%): bisecting np / sf / pool", flush=True)
    for arm in ("np", "sf", "pool"):
        realistic(arm, "base", arm)
        report(arm, arm, "base")
        p, se, n, w = pct("real", arm, "base")
        if p >= -TOL: keep.append(arm)
print("KEEP", keep, flush=True)
flags = {"all": "pool=3,npipe=1,subp=1,fuse=1,npmin=32768", "np": "npipe=1,npmin=32768", "sf": "subp=1,fuse=1", "pool": "pool=3"}
fs = ",".join(flags[a] for a in keep)
open(HOME + "/strata-prefill21/logs/flagstr22.txt", "w").write(fs + "\n")
open(ARMF, "w").write("\n")
mark()
print("DEC22DONE flagstr=" + fs, flush=True)
