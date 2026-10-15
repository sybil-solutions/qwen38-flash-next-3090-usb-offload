#!/bin/zsh
# mkbatch15.sh <base.json> (ON the mini): stage the concurrency configs from <base> into ~/strata-decode15/cfg: ctl40 (base, --max-context 40960, no batch), batch2, batch4 (+ --batch N), batch2m / batch4m (+ STRATA_BATCH_MTP=1: a slot also verifies one MTP proposal per window).
# --max-context is 40960 for every arm: a batch slot carves a whole session at the full context (2.39 GiB at 204,800 cells), the elastic K/V is off with --batch (generate.cpp:3849), so a 200K context cannot be had with slots.
b=$1; M=~/strata-decode15/tools/mkcfg15.py
python3 $M $b ctl40 @--max-context=40960
python3 $M $b batch2 @--max-context=40960 +--batch +2
python3 $M $b batch4 @--max-context=40960 +--batch +4
python3 $M $b batch2m @--max-context=40960 +--batch +2 STRATA_BATCH_MTP=1
python3 $M $b batch4m @--max-context=40960 +--batch +4 STRATA_BATCH_MTP=1
