#!/bin/zsh
# cen15.sh (ON the mini, card owner; server already up on the census config): VRAM census scenario.  Output /tmp/tgcu_census.out (+ ~/strata-decode15/logs/census.out)
cfg=${1:-~/strata-decode15/cfg/cen.json}
export PF600_CFG=$cfg
req(){ echo $1 > /tmp/tgcu_census_req; for i in {1..50}; do [ -e /tmp/tgcu_census_req ] || break; sleep 0.2; done; }
rm -f /tmp/tgcu_census.out
req idle_after_start
# short decode request (the probe prompt, 300 tokens): census in the middle of it
python3 ~/shim/split_probe2.py cen15dec 300 > ~/strata-decode15/logs/cen_dec.log 2>&1 &
sleep 4; req decode_mid_run1
wait
req idle_after_decode
# prefill 8K: census at +6 s, +12 s, +18 s
python3 ~/shim/pf600_bench.py rows 8000 --runs 1 --tag cen15_8k > ~/strata-decode15/logs/cen_pf8k.log 2>&1 &
sleep 6; req prefill8k_t6
sleep 6; req prefill8k_t12
sleep 5; req prefill8k_t17
wait
req idle_after_prefill8k
# decode after the prefill (steady decode at ~8K context)
python3 ~/shim/split_probe2.py cen15dec2 300 > ~/strata-decode15/logs/cen_dec2.log 2>&1 &
sleep 4; req decode_mid_after_prefill
wait
# 33K prefill (K/V growth): census +10 s, +30 s, +50 s
python3 ~/shim/pf600_bench.py rows 33000 --runs 1 --tag cen15_33k > ~/strata-decode15/logs/cen_pf33k.log 2>&1 &
sleep 10; req prefill33k_t10
sleep 20; req prefill33k_t30
sleep 20; req prefill33k_t50
wait
req idle_after_prefill33k
tag=${2:-cen}; cp /tmp/tgcu_census.out ~/strata-decode15/logs/$tag.census.out; cp /tmp/tgcu_alloc.log ~/strata-decode15/logs/$tag.alloc.log
echo CEN15DONE
