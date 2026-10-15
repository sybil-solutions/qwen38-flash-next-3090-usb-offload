#!/bin/zsh
# q18.sh <base-config.json> (ON the mini, card owner): teacher-forced quality gate for the Prefill18 final config (q15.sh with the p18 tags): restart on <base> + STRATA_LOGPOS,
# the 8 prompts / 705 positions of the saved baseline ~/shim/xp8_ref.json, prime smoke (short + after the lane), then lpcmp p-off -> p18cur and d15cur -> p18cur.  Log ~/shim/d18_q.log
zsh ~/shim/sleep_guard.sh || exit 1
base=$1; L=~/shim/d18_q.log; : > $L
cfg=~/strata-prefill18/cfg/q-cur.json
python3 ~/strata-decode8/tools/mkcfg8.py $base $cfg STRATA_LOGPOS=$HOME/shim/xp8_logpos_p18cur.tsv STRATA_LOGPOS_TOPK=64
rm -f ~/shim/xp8_logpos_p18cur.tsv
echo "### deploy $(date +%T)" >> $L
zsh ~/strata-decode15/tools/restart15.sh $cfg > ~/shim/d18_deploy_q.log 2>&1 || { echo "DEPLOY FAILED" >> $L; exit 1; }
export XP8_CFG=$cfg PF600_CFG=$cfg
python3 ~/strata-xport8/tools/xp8_quality.py lp p18cur >> $L 2>&1
echo "-- prime (short)" >> $L; python3 ~/shim/split_probe2.py d18_q_cur 300 >> $L 2>&1
echo "-- prime (after lane)" >> $L; python3 ~/strata-decode5/prime_lane.py p18cur $cfg >> $L 2>&1
echo "### SCORES" >> $L
for pair in "p-off p18cur" "d15cur p18cur"; do
  set -- ${=pair}; echo "== lpcmp $1 -> $2" >> $L
  python3 ~/strata-xport8/tools/xp8_quality.py lpcmp $1 $2 >> $L 2>&1
done
echo "== text cmp d15cur p18cur" >> $L; python3 ~/strata-xport8/tools/xp8_quality.py cmp d15cur p18cur >> $L 2>&1
echo Q18DONE >> $L
