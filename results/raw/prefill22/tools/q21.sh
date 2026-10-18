#!/bin/zsh
# q21.sh <base-config.json> <tag> (ON the mini, card owner): teacher-forced quality arm on <base> + STRATA_LOGPOS/TOPK/ROUTE_DUMP, then lpcmp vs p-off.  Outputs ~/shim/xp8_lp_<tag>.json, logs/q21_<tag>.log (ends with the lpcmp verdict).
D=~/strata-prefill21
zsh ~/shim/sleep_guard.sh || exit 1
zsh $D/tools/usercheck21.sh || exit 3
base=$1; tag=$2; L=$D/logs/q21_$tag.log; : > $L
cfg=$D/cfg/q_$tag.json
rm -f ~/shim/xp8_logpos_$tag.tsv $D/logs/route_$tag.bin
python3 $D/tools/mkcfg21.py $base q_$tag STRATA_LOGPOS=$HOME/shim/xp8_logpos_$tag.tsv STRATA_LOGPOS_TOPK=64 STRATA_DEC_ROUTE_DUMP=$D/logs/route_$tag.bin >> $L
echo "### deploy $(date +%T)" >> $L
zsh $D/tools/restart21.sh $cfg > $D/logs/deploy_q_$tag.log 2>&1 || { echo "DEPLOY FAILED" >> $L; exit 1; }
zsh $D/tools/mark21.sh
export XP8_CFG=$cfg PF600_CFG=$cfg
elog=~/shim/serve-engine.log; mark=$(wc -l < $elog)
echo "### lp $(date +%T)" >> $L
python3 ~/strata-xport8/tools/xp8_quality.py lp $tag >> $L 2>&1
zsh $D/tools/mark21.sh
tail -n +$((mark+1)) $elog | grep -a "strata hs:\|decode timing\|ERR\|GPU timeout" | cut -c1-400 | tail -30 >> $L
echo "### lpcmp p-off $tag $(date +%T)" >> $L
python3 ~/strata-xport8/tools/xp8_quality.py lpcmp p-off $tag >> $L 2>&1; echo "lpcmp rc=$?" >> $L
echo "### lpcmp e19cur $tag (informational)" >> $L
python3 ~/strata-xport8/tools/xp8_quality.py lpcmp e19cur $tag >> $L 2>&1; echo "lpcmp rc=$?" >> $L
echo Q21DONE >> $L
