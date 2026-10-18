#!/usr/bin/env python3
"""repro22.py TAG  (ON the mini; env XP8_CFG = a config with the api key): the first 5 prompts of xp8_quality.lp (R1..R4 each; the 4th request of the 5th prompt 'chat' crashed config F with 'bulk IN short read' in hs_wait).
Prints REPRO <tag> PASS / CRASH <error>."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/strata-xport8/tools"))
import xp8_quality as q
tag = sys.argv[1]
orig = q.prompts
q.prompts = lambda variant=0: orig(variant)[:5]
try:
    q.lp("repro_" + tag, False)
    print(f"REPRO {tag} PASS", flush=True)
except Exception as e:
    print(f"REPRO {tag} CRASH {e!r}"[:300], flush=True); sys.exit(1)
