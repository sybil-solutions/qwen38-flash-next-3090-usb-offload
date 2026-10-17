#!/bin/zsh
# q19.sh <base-config.json> <tag> [K=V ...] (ON the mini, card owner): teacher-forced scoring arm: restart on <base> + STRATA_LOGPOS/TOPK/ROUTE_DUMP + extra env, run xp8_quality lp <tag> (8 prompts / ~705 positions),
# then restore the live config D.  Outputs ~/shim/xp8_lp_<tag>.json, ~/shim/xp8_logpos_<tag>.tsv, ~/strata-fewertrips19/logs/route_<tag>.bin, logs/q19_<tag>.log
zsh ~/shim/sleep_guard.sh || exit 1
[ -e ~/strata-fewertrips19/USER_ACTIVE ] && { echo USER_ACTIVE; exit 3; }
base=$1; tag=$2; shift 2
L=~/strata-fewertrips19/logs/q19_$tag.log; : > $L
cfg=~/strata-fewertrips19/cfg/q_$tag.json
rm -f ~/shim/xp8_logpos_$tag.tsv ~/strata-fewertrips19/logs/route_$tag.bin
python3 ~/strata-fewertrips19/tools/mkcfg19.py $base q_$tag STRATA_LOGPOS=$HOME/shim/xp8_logpos_$tag.tsv STRATA_LOGPOS_TOPK=64 STRATA_DEC_ROUTE_DUMP=$HOME/strata-fewertrips19/logs/route_$tag.bin "$@" >> $L
echo "### deploy $(date +%T)" >> $L
zsh ~/strata-decode15/tools/restart15.sh $cfg > ~/strata-fewertrips19/logs/deploy_q_$tag.log 2>&1 || { echo "DEPLOY FAILED" >> $L; exit 1; }
export XP8_CFG=$cfg PF600_CFG=$cfg
elog=~/shim/serve-engine.log; mark=$(wc -l < $elog)
echo "### lp $(date +%T)" >> $L
python3 ~/strata-xport8/tools/xp8_quality.py lp $tag >> $L 2>&1
tail -n +$((mark+1)) $elog | grep -a "strata hs:\|decode timing\|ERR\|GPU timeout" | cut -c1-400 | tail -30 >> $L
echo "### done $(date +%T)" >> $L
echo Q19DONE >> $L
