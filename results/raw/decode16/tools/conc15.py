#!/usr/bin/env python3
"""conc15.py CFG C OUT.jsonl ITEM... (ON the mini): C client threads work through the ITEMs (same format as run_seq.py: <prompt stem>:<flag>[:<max_tokens>], greedy, thinking off, streamed); a client sends its next
prompt the moment its previous reply ends, so up to C requests are in flight.  Per request: ttft, n, engine tok/s (timings.predicted_per_second), wall; at the end: aggregate tok/s = sum(n) / (last end - first start),
the mean per-request engine tok/s, mean TTFT, and the peak concurrency seen.  The flag file ~/... STRATA_D7_FLAG is written once (the first item's flag)."""
import json, sys, time, urllib.request, hashlib, os, threading, queue
cfg, C, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
items = sys.argv[4:]
H = {"Content-Type": "application/json", "Authorization": "Bearer " + json.load(open(cfg))["api_key"]}
D = os.path.expanduser("~/strata-decode7/prompts")
q = queue.Queue()
for it in items: q.put(it)
open("/tmp/d7flag", "w").write(items[0].split(":")[1])
lock = threading.Lock(); recs = []; live = [0, 0]   # in flight now, peak
def one(item):
    parts = item.split(":"); name = parts[0]; mt = int(parts[2]) if len(parts) > 2 else 300
    text = json.load(open(f"{D}/{name}.json"))["text"] if os.path.exists(f"{D}/{name}.json") else (open(name).read() if os.path.exists(name) else name)
    body = json.dumps({"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": mt, "temperature": 0, "stream": True, "reasoning_effort": "none",
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time(); ttft = None; txt = []; timings = {}
    with lock:
        live[0] += 1; live[1] = max(live[1], live[0])
    try:
        with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, H), timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                j = json.loads(line[5:])
                if j.get("timings"): timings = j["timings"]
                for c in j.get("choices", []):
                    d = c.get("delta", {}).get("content")
                    if d:
                        if ttft is None: ttft = time.time() - t0
                        txt.append(d)
    finally:
        with lock: live[0] -= 1
    t1 = time.time()
    rec = {"name": name, "t0": t0, "t1": t1, "prompt_n": timings.get("prompt_n"), "ttft_s": round(ttft or -1, 2), "n": timings.get("predicted_n"),
           "tok_s": round(timings.get("predicted_per_second", 0), 2), "wall_s": round(t1 - t0, 1), "md5": hashlib.md5("".join(txt).encode()).hexdigest()[:8]}
    with lock:
        recs.append(rec); open(out, "a").write(json.dumps(rec) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k not in ("t0", "t1")}), flush=True)
def worker():
    while True:
        try: item = q.get_nowait()
        except queue.Empty: return
        one(item)
ts = [threading.Thread(target=worker) for _ in range(C)]
tw = time.time()
for t in ts: t.start()
for t in ts: t.join()
span = max(r["t1"] for r in recs) - min(r["t0"] for r in recs)
tot = sum(r["n"] or 0 for r in recs)
dec = [r["tok_s"] for r in recs if r["tok_s"] > 0]
print(f"CONC c={C} requests={len(recs)} tokens={tot} span={span:.1f}s aggregate={tot / span:.2f} tok/s  per-request engine tok/s mean={sum(dec) / max(len(dec), 1):.2f}  ttft mean={sum(r['ttft_s'] for r in recs) / len(recs):.1f}s  wall mean={sum(r['wall_s'] for r in recs) / len(recs):.1f}s  peak in flight={live[1]}", flush=True)
