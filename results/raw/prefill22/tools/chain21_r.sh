#!/bin/zsh
# chain21_r.sh ARMNAME=FLAGS ... (ON the mini): for each arm: fresh restart on cfg/ab21.json (flag-file shim levers), flags written, repro22.py (first 5 prompts of the teacher-forced quality flow); short-read count of the engine log per arm.  Ends with restore E.
D=~/strata-prefill21; O=$D/logs/chain_r.out; : > $O
export XP8_CFG=$D/cfg/ab21.json PF600_CFG=$D/cfg/ab21.json
for a in "$@"; do
  name=${a%%=*}; fl=${a#*=}; [ "$fl" = "-" ] && fl=""
  zsh $D/tools/usercheck21.sh >> $O 2>&1 || { echo "user active" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit 3; }
  echo "$fl" > $D/pfcfg.txt
  zsh $D/tools/restart21.sh $D/cfg/ab21.json > $D/logs/chain_r.deploy.log 2>&1 || { echo "deploy failed $name" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit 1; }
  echo "$fl" > $D/pfcfg.txt; zsh $D/tools/mark21.sh
  n0=$(grep -ac "short read" ~/shim/serve-engine.log)
  python3 $D/tools/repro22.py $name 2>&1 | grep -a "REPRO\|^chat\|^math" | cut -c1-260 >> $O
  zsh $D/tools/mark21.sh
  echo "arm $name ($fl): short reads +$(( $(grep -ac 'short read' ~/shim/serve-engine.log) - n0 ))" >> $O
done
echo "" > $D/pfcfg.txt
zsh $D/tools/restore21.sh >> $O 2>&1
echo CHAINRDONE >> $O
