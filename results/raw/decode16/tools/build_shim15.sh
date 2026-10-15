#!/bin/zsh
# build_shim15.sh (ON the mini, card-free): build ~/strata-decode15/build/libcudart.dylib from ~/strata-decode15/src/{tgcudart,tgcublas}.cpp (the hs2 shim source + the cuMem* entry points)
set -e
D=~/strata-decode15; mkdir -p $D/build; cd $D/src
PY=$HOME/.local/share/uv/python/cpython-3.12.15-macos-aarch64-none
SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
clang++ -std=c++17 -O2 -shared -fPIC -isysroot $SDK -I$PY/include/python3.12 -I$HOME/cuda-12.8/include -I$HOME/shim/include -I. \
  -Wno-everything -L$PY/lib -lpython3.12 -Wl,-install_name,@rpath/libcudart.dylib -o $D/build/libcudart.dylib tgcudart.cpp tgcublas.cpp
codesign -f -s - $D/build/libcudart.dylib
ls -la $D/build/libcudart.dylib; md5 -q $D/build/libcudart.dylib; otool -L $D/build/libcudart.dylib | head -4
