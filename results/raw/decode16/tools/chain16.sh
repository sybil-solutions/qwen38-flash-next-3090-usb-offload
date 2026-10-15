#!/bin/zsh
# chain16.sh: wait for b1 (abconc15b) to finish its b4mA arm, stop it, then run the 4-prompt + short-prompt arms and restore the live config.
D=~/strata-decode15; C=$D/cfg; L=~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json
while [ $(grep -ac "GPU timeouts in log" $D/logs/abconc_b1.out) -lt 2 ]; do sleep 1; grep -aq "ABCONC15BDONE-FAULT" $D/logs/abconc_b1.out && exit 3; done
pkill -f "abconc15b.sh"; pkill -f "restart15.sh"; pkill -f ls_srv.sh; sleep 1
echo "b1 stopped after b4mA $(date +%T)" >> $D/logs/abconc_b1.out
PHASES="real short" zsh $D/tools/abconc15c.sh b2 b2mA:$C/batch2m.json b4A:$C/batch4.json b2A:$C/batch2.json
PHASES="short" zsh $D/tools/abconc15c.sh b3 ctlS:$C/ctl40.json b4mS:$C/batch4m.json
echo "chain end, restoring live $(date +%T)" > $D/logs/chain16.end
zsh $D/tools/restart15.sh $L > $D/logs/restore_after_chain.log 2>&1; tail -3 $D/logs/restore_after_chain.log >> $D/logs/chain16.end; echo RESTORED >> $D/logs/chain16.end
