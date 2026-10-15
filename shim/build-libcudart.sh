#!/bin/zsh
# libcudart.dylib: the CUDA runtime/driver/cuBLAS-subset replacement that Strata links on macOS. Every call goes into tgcu_api.py
# (embedded CPython) -> tgcu.py -> tinygrad NV over USB.  Plain C++ build, no CUDA toolkit libraries; only CUDA 12.8 *headers*.
# Usage: build-libcudart.sh [OUT_DIR]   (default: this directory; the engine finds it through -Wl,-rpath,$HOME/shim)
# Env: PY (python prefix; default the uv-managed CPython 3.12.15 that tinygrad's venv uses), CUDA_HOST (default ~/cuda-12.8), SDK.
# The flags are those of the deployed build (campaign script build_shim15.sh: no -arch, no extra rpath, ad-hoc codesign afterwards), run from this
# directory with relative source names; with the mini's toolchain (Apple clang 21, MacOSX26.5.sdk, CPython 3.12.15) the result is byte-identical to the
# deployed libcudart.dylib of config C (docs/reproduce.md section 11).
set -e
cd ${0:A:h}
OUT=${1:-$PWD}; mkdir -p $OUT
PY=${PY:-$HOME/.local/share/uv/python/cpython-3.12.15-macos-aarch64-none}
CUDA_HOST=${CUDA_HOST:-$HOME/cuda-12.8}
SDK=${SDK:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}
clang++ -std=c++17 -O2 -shared -fPIC -isysroot $SDK -I$PY/include/python3.12 -I$CUDA_HOST/include -I$PWD/include -I. \
  -Wno-everything -L$PY/lib -lpython3.12 -Wl,-install_name,@rpath/libcudart.dylib -o $OUT/libcudart.dylib tgcudart.cpp tgcublas.cpp
codesign -f -s - $OUT/libcudart.dylib
echo built $OUT/libcudart.dylib
shasum -a 256 $OUT/libcudart.dylib
