#!/bin/sh
# mkpatch.sh (local, in runs/prefill18): regenerate patches/0020-prefill18-lazy-loan-refill.patch = generate.cpp.orig (the Decode15 / config C source, the end of
# patches/strata/0001..0019) -> src/generate.cpp (Prefill18).
cd "$(dirname "$0")/.." || exit 1
out=patches/0020-prefill18-lazy-loan-refill.patch
cat > $out <<'EOF'
# Prefill18 (2026-10-15): lazy loan refill. STRATA_D7_MODE bit 8 (15 = guided 1 + LRU warm 2 + staged 4 + lazy 8): the guided refill of the VRAM slots a prompt lent
# to the prefill buffers no longer blocks the first token. At the end of a prompt STRATA_D18_K (config D: 300; unset = all but STRATA_D18_PEND, default 300) of the lent slots are refilled (best score
# first); the rest stay NOT resident (their experts are misses the mini LRU serves, exact; the LRU warm
# already holds them) and are copied in pieces of STRATA_D18_P slots (4) while the engine waits for the next request (STRATA_D18_IDLE, default 1) and,
# optionally, between decode windows within a stall budget of STRATA_D18_PHI x the decode time so far + STRATA_D18_GRACE ms (default 0 = off: it only moves the
# stall into decode). Every piece re-checks that its slot is free and its expert not resident; pending slots below a later loan stay pending, the ones that loan
# overwrites are dropped. STRATA_D18_CFG=<file> re-reads the knobs (K=, PEND=, P=, W=, IDLE=, PHI=, GRACE=) at every refill (A/B).
# Measured (config D = config C + STRATA_D7_MODE=15 STRATA_D18_K=300, G=6): prefill 2.5K/8K/18K/33K 176/382-422/444-473/550-579 -> 233/506-511/517/614 tok/s, paired in-server decode -1.5%.
# Base: Strata v0.1.41 (fb58e0dbc839) + patches/strata/0001..0019, applied in order with `git apply -p1`.

diff --git a/src/program/generate.cpp b/src/program/generate.cpp
EOF
diff -u --label a/src/program/generate.cpp --label b/src/program/generate.cpp src/generate.cpp.orig src/generate.cpp >> $out
grep -c '^[+-]' $out
