#!/bin/zsh
# chain21_b.sh <candcfg>: fresh-restart ABBA, config E (live file) vs the candidate: E1 C1 C2 E2.  User activity -> restore config E and stop.
D=~/strata-prefill21; O=$D/logs/chain_b.out; : > $O; cand=$1
E=~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json
for pair in "E1 $E" "C1 $cand" "C2 $cand" "E2 $E"; do
  set -- ${=pair}; tag=$1; cfg=$2
  [ -e $D/USER_ACTIVE ] && { echo "USER_ACTIVE stop" >> $O; exit 3; }
  echo "### $tag $(date +%T)" >> $O
  zsh $D/tools/arm21.sh $tag $cfg >> $O 2>&1; rc=$?
  tail -2 $D/logs/abba_$tag.out >> $O
  [ $rc -ne 0 ] && { echo "arm $tag rc=$rc: restoring E" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit $rc; }
done
python3 $D/tools/abbasum22.py $D/logs E1 C1 C2 E2 >> $O 2>&1
zsh $D/tools/mark21.sh
echo CHAINBDONE >> $O
