#!/bin/zsh
# q15.sh <base-config.json> (ON the mini, card owner): teacher-forced quality gate for the FINAL config only (the Decode7 q7.sh 'cur' arm): the base config + STRATA_LOGPOS, restart, xp8_quality.py lp (8 prompts, 705 positions, replies of the saved
# baseline ~/shim/xp8_ref.json), prime smoke (short + after the lane), then lpcmp p-off -> d15cur and d7cur -> d15cur (the Decode7 stack's saved logpos, if present).  Log ~/shim/d15_q.log
zsh ~/shim/sleep_guard.sh || exit 1
base=$1; L=~/shim/d15_q.log; : > $L
cfg=~/strata-decode15/cfg/q-cur.json
python3 ~/strata-decode8/tools/mkcfg8.py $base $cfg STRATA_LOGPOS=$HOME/shim/xp8_logpos_d15cur.tsv STRATA_LOGPOS_TOPK=64
rm -f ~/shim/xp8_logpos_d15cur.tsv
echo "### deploy $(date +%T)" >> $L
zsh ~/strata-decode15/tools/restart15.sh $cfg > ~/shim/d15_deploy_q.log 2>&1 || { echo "DEPLOY FAILED" >> $L; exit 1; }
export XP8_CFG=$cfg PF600_CFG=$cfg
python3 ~/strata-xport8/tools/xp8_quality.py lp d15cur >> $L 2>&1
echo "-- prime (short)" >> $L; python3 ~/shim/split_probe2.py d15_q_cur 300 >> $L 2>&1
echo "-- prime (after lane)" >> $L; python3 ~/strata-decode5/prime_lane.py d15cur $cfg >> $L 2>&1
echo "### SCORES" >> $L
for pair in "p-off d15cur" "d13cur d15cur" "d12cur d15cur"; do
  set -- ${=pair}; echo "== lpcmp $1 -> $2" >> $L
  python3 ~/strata-xport8/tools/xp8_quality.py lpcmp $1 $2 >> $L 2>&1
done
echo "== text cmp d13cur d15cur" >> $L; python3 ~/strata-xport8/tools/xp8_quality.py cmp d13cur d15cur >> $L 2>&1
echo Q14DONE >> $L
