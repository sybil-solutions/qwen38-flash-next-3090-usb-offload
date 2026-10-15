#!/usr/bin/env python3
"""longdec.py CFG LENGTH [depth=50] [max_tokens=300] (ON the mini): one long-context request through the server = the needle_bench haystack (LENGTH tokens, e.g. 100k / 190k) with a code word hidden at `depth` %,
asked for the word AND a long continuation, so the same request gives the needle result, the prefill rate and the DECODE rate at that context (server timings: predicted_per_second).
Prints one JSON line: length, prompt_n, found, prefill_s, decode tok/s, decode ms, tokens, md5 of the text.  Never prints the API key."""
import json, sys, time, urllib.request, hashlib, random, os
sys.path.insert(0, os.path.expanduser("~/strata/tools"))
import needle_bench as nb
cfg, L = sys.argv[1], sys.argv[2]
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 50
mt = int(sys.argv[4]) if len(sys.argv) > 4 else 300
key = json.load(open(cfg))["api_key"]
tokens = int(int(float(L.lower().rstrip("k")) * 1024) * 0.98) if L.lower().endswith("k") else int(L)
rnd = random.Random(int(os.environ.get("LONGDEC_SEED", "7")))
text = nb.haystack(int(tokens * nb.CHARS_PER_TOKEN))
word = f"{rnd.choice(nb.WORDS)}-{rnd.choice(nb.WORDS)}-{rnd.randint(100, 999)}"
cut = int(len(text) * depth / 100); cut = text.rfind("\n", 0, cut) + 1 or cut
needle = f"\nThe secret code word for this text is: {word}. Remember it.\n"
prompt = (text[:cut] + needle + text[cut:] +
          "\n\nWhat is the secret code word mentioned in the text above? Start your reply with the code word, then write a long, detailed, multi-paragraph description of the kinds of files that appear in the text (at least 400 words).")
body = json.dumps({"model": "x", "messages": [{"role": "user", "content": prompt}], "max_tokens": mt, "temperature": 0, "stream": True,
                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
H = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
t0 = time.time(); ttft = None; txt = []; timings = {}
with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body, H), timeout=5400) as r:
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
wall = time.time() - t0; out = "".join(txt)
print(json.dumps({"length": L, "depth": depth, "prompt_n": timings.get("prompt_n"), "found": word in out, "ttft_s": round(ttft or -1, 1), "wall_s": round(wall, 1),
                  "n": timings.get("predicted_n"), "decode_tok_s": round(timings.get("predicted_per_second", 0), 2), "prefill_tok_s": round(timings.get("prompt_per_second", 0), 1),
                  "md5": hashlib.md5(out.encode()).hexdigest()[:12], "head": out[:60]}), flush=True)
