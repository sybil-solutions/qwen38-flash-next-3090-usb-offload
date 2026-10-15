#!/bin/zsh
# deploy14.sh (ON the mini, card owner; the server must be STOPPED or restarted afterwards): install the Decode15 shim (tgcu.py / tgcu_api.py / libcudart.dylib with the cuMem* VMM
# emulation) into ~/shim.  The originals are in ~/strata-decode15/orig (md5 tgcu.py aa3d57b7, tgcu_api.py e03bc623, libcudart.dylib d7a11095).  New inodes (mv), never an overwrite in place.
D=~/strata-decode15; S=~/shim
for f in tgcu.py tgcu_api.py libcudart.dylib; do [ -f $D/orig/$f ] || { echo "no backup of $f"; exit 1; }; done
cp $D/src/tgcu.py $S/.tgcu.py.new && mv $S/.tgcu.py.new $S/tgcu.py
cp $D/src/tgcu_api.py $S/.tgcu_api.py.new && mv $S/.tgcu_api.py.new $S/tgcu_api.py
cp $D/build/libcudart.dylib $S/.libcudart.dylib.new && mv $S/.libcudart.dylib.new $S/libcudart.dylib
rm -rf $S/__pycache__/tgcu.cpython* $S/__pycache__/tgcu_api.cpython* 2>/dev/null
md5 -q $S/tgcu.py $S/tgcu_api.py $S/libcudart.dylib
