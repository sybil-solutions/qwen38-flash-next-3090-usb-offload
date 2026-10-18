#!/bin/zsh
# libtgusb.dylib (Prefill21, config F): the libusb asynchronous-transfer executor that tgcu.py loads when TGCU_USB_NPIPE / TGCU_USB_SUBP / TGCU_USB_FUSE (or TGCU_P21_FLAG) is set.
# Plain C, no libusb header or library: the libusb entry points are passed in at run time (tgu_init); the engine never links it, tgcu.py loads it from its own directory.
# Usage: build-libtgusb.sh [OUT_DIR]   (default: this directory).  Env: SDK (default the Command Line Tools MacOSX26.5 SDK).  The flags are those of the deployed build (ad-hoc codesign afterwards).
set -e
cd ${0:A:h}
OUT=${1:-$PWD}; mkdir -p $OUT
SDK=${SDK:-/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk}
clang -O2 -Wall -Wno-unused-function -shared -fPIC -isysroot $SDK -o $OUT/libtgusb.dylib tgusb.c -install_name @rpath/libtgusb.dylib
codesign -f -s - $OUT/libtgusb.dylib
echo built $OUT/libtgusb.dylib
shasum -a 256 $OUT/libtgusb.dylib
