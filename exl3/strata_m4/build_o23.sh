#!/bin/zsh
# libstrata_o23.dylib (Opt23, config G): host stub + C API (exl3o_doorbell, exl3o_add_h) embedding ../exl3_o23.fatbin, linked to the tgcu libcudart like libstrata_exl3.dylib.
# Run ../build.sh exl3_o23 first (nvcc 13.4 sm_86 fatbin).  Strata dlopens it when STRATA_O23 is set (STRATA_O23_LIB, default $HOME/strata-opt23/lib/libstrata_o23.dylib; the config template sets $HOME/exl3/strata_m4/libstrata_o23.dylib).
set -e
cd ${0:A:h}
SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
clang++ -x cuda --cuda-host-only --no-offload-new-driver --cuda-path=$HOME/cuda-12.8 -nocudalib -include cuda_runtime.h \
  -include $HOME/shim/tgcuda_launch.h -Xclang -fcuda-include-gpubinary -Xclang ../exl3_o23.fatbin -std=c++17 -O2 -fPIC -isystem $HOME/shim/include \
  -isystem $HOME/cuda-12.8/include -isysroot $SDK -Wno-unknown-cuda-version -c exl3_o23_host.cu -o exl3_o23_host.o
clang++ -shared -O2 exl3_o23_host.o -o libstrata_o23.dylib -L$HOME/shim -lcudart -Wl,-rpath,$HOME/shim -isysroot $SDK
rm -f exl3_o23_host.o
nm -gU libstrata_o23.dylib | grep exl3o_
shasum -a 256 libstrata_o23.dylib
