#!/bin/zsh
# long14.sh <cfg> <tag> [restart=1] (ON the mini, card owner): restart on <cfg>, then ~100K and ~190K needle requests with a long continuation (longdec.py: needle, prefill rate AND decode rate at that context),
# then a short probe round (does the cache come back after the long contexts?).  Log ~/strata-decode15/logs/long_<tag>.log (+ the engine's K/V lines).
zsh ~/shim/sleep_guard.sh || exit 1
cfg=$1; tag=$2; D=~/strata-decode15; L=$D/logs/long_$tag.log; elog=~/shim/serve-engine.log; : > $L
[ "${3:-1}" = 1 ] && { zsh $D/tools/restart15.sh $cfg > $D/logs/long_$tag.deploy.log 2>&1 || { echo DEPLOY FAILED >> $L; exit 1; }; }
mark=$(wc -l < $elog)
echo "### 100K $(date +%T)" >> $L; perl -e 'alarm 3000; exec @ARGV' python3 $D/tools/longdec.py $cfg 100k 50 300 2>&1 | cut -c1-400 >> $L
echo "### probe after 100K $(date +%T)" >> $L; python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-80 >> $L
echo "### 190K $(date +%T)" >> $L; perl -e 'alarm 3700; exec @ARGV' python3 $D/tools/longdec.py $cfg 190k 50 300 2>&1 | cut -c1-400 >> $L
echo "### probe after 190K $(date +%T)" >> $L; python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-80 >> $L
tail -n +$((mark+1)) $elog | grep -a "K/V grown\|K/V trimmed\|elastic K/V\|ERR\|GPU timeout\|strata d7:" | cut -c1-300 >> $L
tail -n +$((mark+1)) $elog | grep -a "strata serve: prompt" | cut -c1-260 >> $L
echo "wired $(( $(vm_stat | awk '/wired down/{print $4}' | tr -d .) * 16 / 1024 )) MiB swap: $(sysctl -n vm.swapusage | tr -s ' ')" >> $L
echo LONGDONE >> $L
