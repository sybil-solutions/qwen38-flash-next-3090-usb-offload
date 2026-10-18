#!/bin/zsh
# chain21_a.sh (ON the mini): stage 1 + 2 of Prefill22: restart cfg/ab21.json (in-server flag arms), decode neutrality (dec22.py), PLE warm-ahead repeats (ple22.py).  Any user activity: restore config E (restore21.sh) and stop.
D=~/strata-prefill21; O=$D/logs/chain_a.out; : > $O
zsh ~/shim/sleep_guard.sh || { echo "sleep guard" >> $O; exit 1; }
zsh $D/tools/usercheck21.sh >> $O 2>&1 || { echo "user active at start" >> $O; exit 3; }
: > $D/pfcfg.txt
zsh $D/tools/restart21.sh $D/cfg/ab21.json > $D/logs/chain_a.deploy.log 2>&1 || { echo "DEPLOY FAILED" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit 1; }
zsh $D/tools/mark21.sh
rm -f $D/logs/dec22.jsonl $D/logs/ple22.jsonl
elog=~/shim/serve-engine.log; mk=$(wc -l < $elog)
python3 $D/tools/dec22.py $D/logs/dec22.jsonl > $D/logs/dec22.out 2>&1; rc=$?
grep -a "RESULT\|STAGE1\|KEEP\|DEC22DONE" $D/logs/dec22.out >> $O
[ $rc -ne 0 ] && { echo "dec22 rc=$rc: restoring config E" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit $rc; }
fs=$(cat $D/logs/flagstr22.txt)
echo "flagstr='$fs'" >> $O
python3 $D/tools/ple22.py "$fs" $D/logs/ple22.jsonl 2 > $D/logs/ple22.out 2>&1; rc=$?
tail -12 $D/logs/ple22.out >> $O
tail -n +$((mk+1)) $elog > $D/logs/chain_a.engine.log
echo "GPU timeouts: $(grep -ac 'GPU timeout' $D/logs/chain_a.engine.log)" >> $O
[ $rc -ne 0 ] && { echo "ple22 rc=$rc: restoring config E" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit $rc; }
zsh $D/tools/mark21.sh
echo CHAINADONE >> $O
