#!/usr/bin/env python3
"""make_profile_wholelayers.py OUT_DIR L0,L1,... : profile = every expert of the listed layers first (whole-layer residency: those layers need no
doorbell round trip at decode), then the rest in the shipped f512c frequency order (exl3/placement/f512c/profile.bin).
Config C uses layers 0,1,2,6 (2,048 slots):   make_profile_wholelayers.py exl3/placement/d8w4 0,1,2,6   ->  profile.bin md5 2ac3b7e1436c1720487993969c651c5c
Needs Strata's tools/make_profile.py (patched checkout: STRATA_TOOLS, default ~/strata/tools).  Run from anywhere; paths of the f512c input: F512C or exl3/placement/f512c next to this file."""
import os, sys
sys.path.insert(0, os.environ.get("STRATA_TOOLS", os.path.expanduser("~/strata/tools")))
from make_profile import read_profile, write_profile
here = os.path.dirname(os.path.abspath(__file__))
out = sys.argv[1]; whole = [int(x) for x in sys.argv[2].split(',') if x != '']
base = read_profile(os.environ.get("F512C", os.path.join(here, "placement", "f512c", "profile.bin")), 512)
assert len(base) == 48 * 512
ranked = []
for l in whole: ranked += [(ll, e) for ll, e in base if ll == l]
ranked += [(ll, e) for ll, e in base if ll not in whole]
assert len(ranked) == 48 * 512 and len(set(ranked)) == len(ranked)
os.makedirs(out, exist_ok=True)
write_profile(os.path.join(out, "profile.bin"), ranked, 512)
assert read_profile(os.path.join(out, "profile.bin"), 512) == ranked
print(out, "whole layers", whole, "= %d slots" % (512 * len(whole)))
