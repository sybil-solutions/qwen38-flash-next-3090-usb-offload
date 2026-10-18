#!/usr/bin/env python3
"""gate22.py: ABBA gate for promotion: candidate (C) decode paired vs E >= -0.5 % (no regression), probes >= -1 %, and C's mean prefill tok/s >= E's at 2.5K / 8K / 18K / 33K.  exit 0 = go."""
import re, subprocess, sys, os
out = subprocess.run(["python3", os.path.expanduser("~/strata-prefill21/tools/abbasum22.py"), os.path.expanduser("~/strata-prefill21/logs"), "E1", "C1", "C2", "E2"], capture_output=True, text=True).stdout
print(out)
m = re.search(r"decode C vs E paired over \d+ prompts: ([+-][0-9.]+)%", out); p = re.search(r"probes C vs E ([+-][0-9.]+)%", out)
ok = bool(m and p and float(m.group(1)) >= -0.5 and float(p.group(1)) >= -1.0)
rows = re.findall(r"^\s+(\d+): C: ([0-9 ]+) \(mean (\d+), .*?E: ([0-9 ]+) \(mean (\d+),", out, re.M)
for n, _, c, _, e in rows:
    print(f"prefill {n}: C {c} vs E {e}"); ok = ok and int(c) >= int(e)
print("GATE", "GO" if ok and len(rows) == 4 else "NO-GO"); sys.exit(0 if ok and len(rows) == 4 else 1)
