#!/bin/zsh
# chain21_c.sh (ON the mini): 2 more matched-count PLE repeats on the running ab21 server, plew decision, candidate config candF (decode-neutral levers + plew when it passes), then the fresh-restart ABBA (chain21_b.sh).
D=~/strata-prefill21; O=$D/logs/chain_c.out; : > $O
zsh ~/shim/sleep_guard.sh || exit 1
zsh $D/tools/usercheck21.sh >> $O 2>&1 || { echo "user active at start" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit 3; }
fs=$(cat $D/logs/flagstr22.txt)
PLE_START=800000 python3 $D/tools/ple22.py "$fs" $D/logs/ple22b.jsonl 2 > $D/logs/ple22b.out 2>&1; rc=$?
tail -9 $D/logs/ple22b.out >> $O
[ $rc -ne 0 ] && { echo "ple22b rc=$rc: restoring E" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit $rc; }
zsh $D/tools/mark21.sh
python3 $D/tools/mkcand22.py x >> $O 2>&1
pw=$(cat $D/logs/plew_decision.txt)
[ "$(md5 -q ~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json)" = 078235c5bfec8bac85dc0a39d562485d ] || { echo "live file changed" >> $O; exit 5; }
extra=(); [ "$pw" = 1 ] && extra=(STRATA_PF_PLE_WARM_FIRST=1)
python3 $D/tools/mkcfg21.py ~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json candF "!exe=$D/run-p21.sh" D7_BIN=$D/bin/strata_p21b TGCU_LAZY_POOL=3 TGCU_USB_NPIPE=1 TGCU_USB_SUBP=1 TGCU_USB_FUSE=1 TGCU_USB_FENCE=11 $extra >> $O 2>&1 || { echo "mkcfg failed" >> $O; exit 1; }
echo "candF md5 $(md5 -q $D/cfg/candF.json) plew=$pw" >> $O
zsh $D/tools/chain21_b.sh $D/cfg/candF.json
cat $D/logs/chain_b.out >> $O
echo CHAINCDONE >> $O
