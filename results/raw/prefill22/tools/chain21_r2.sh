#!/bin/zsh
# chain21_r2.sh NAME=ENVLIST ... (ON the mini): crash bisect in the teacher-forced flow: for each arm a config = configF.json + ENVLIST (comma-separated K=V, K=- removes), fresh restart, repro22.py (first 5 prompts of xp8_quality lp), short-read count.  Ends with restore E.
D=~/strata-prefill21; O=$D/logs/chain_r2.out; : > $O; F=$HOME/strata-prefill22/cfg/configF.json
LP="STRATA_LOGPOS=$HOME/shim/xp8_logpos_rq.tsv STRATA_LOGPOS_TOPK=64"; RD="STRATA_DEC_ROUTE_DUMP=$D/logs/route_rq.bin"
for a in "$@"; do
  name=${a%%=*}; extra=${a#*=}; [ "$extra" = "-" ] && extra=""
  extra=${extra//,/ }; extra=${extra//LP/$LP}; extra=${extra//RD/$RD}
  zsh $D/tools/usercheck21.sh >> $O 2>&1 || { echo "user active" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit 3; }
  rm -f ~/shim/xp8_logpos_rq.tsv $D/logs/route_rq.bin
  python3 $D/tools/mkcfg21.py $F rq_$name ${=extra} > /dev/null || { echo "mkcfg failed $name" >> $O; exit 1; }
  zsh $D/tools/restart21.sh $D/cfg/rq_$name.json > $D/logs/chain_r2.deploy.log 2>&1 || { echo "deploy failed $name" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit 1; }
  zsh $D/tools/mark21.sh; n0=$(grep -ac "short read" ~/shim/serve-engine.log)
  XP8_CFG=$D/cfg/rq_$name.json PF600_CFG=$D/cfg/rq_$name.json python3 $D/tools/repro22.py $name 2>&1 | grep -a "REPRO" | cut -c1-260 >> $O
  zsh $D/tools/mark21.sh
  echo "arm $name ($extra): short reads +$(( $(grep -ac 'short read' ~/shim/serve-engine.log) - n0 ))" >> $O
done
zsh $D/tools/restore21.sh >> $O 2>&1
echo CHAINR2DONE >> $O
