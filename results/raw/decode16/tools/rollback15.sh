#!/bin/zsh
# rollback15.sh (ON the mini): back to the Decode14 final stack: the Decode14 shim files (tgcu.py 73133371, tgcu_api.py 77bfa682, libcudart.dylib 24f1cd9a) and the Decode14 config (md5 6f31b09d, binary strata_d14a 9af487e1), then restart the server.
set -e
D=~/strata-decode15; S=~/shim
[ "$(md5 -q $D/cfg/live_d14.json)" = "6f31b09d5590b8159cac95b205df1c85" ] || { echo FATAL cfg md5; exit 1; }
for f in tgcu.py tgcu_api.py libcudart.dylib; do cp -p $D/orig/$f $S/.$f.rb && mv $S/.$f.rb $S/$f; done
rm -rf $S/__pycache__/tgcu.cpython* $S/__pycache__/tgcu_api.cpython* 2>/dev/null || true
cp -p $D/cfg/live_d14.json $S/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json
md5 -q $S/tgcu.py $S/tgcu_api.py $S/libcudart.dylib $S/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json
zsh $D/tools/restart15.sh $S/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json
