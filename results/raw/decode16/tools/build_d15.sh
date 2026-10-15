#!/bin/zsh
# build_d15.sh [verify] [expert_source] [generate] [prefill] (ON the mini, card-free): compile the named objects from ~/strata-decode15/src (in parallel), swap them into COPIES of the lockstep engine archive / the xport8 prefill archive,
# relink ~/strata-decode15/build/strata.next.  Same flags as the lockstep (core/generate) and xport8 (prefill) builds.  Nothing outside ~/strata-decode15 is touched.  No arg = all four.
set -euo pipefail
D=~/strata-decode15; B=$D/build; L=~/strata-mac-lockstep; X=~/strata-xport8/build; LI=~/strata-lockstep/include; XI=~/strata-xport8/include
mkdir -p $B/obj; cd $B
what=("$@"); [ ${#what} -eq 0 ] && what=(verify expert_source layer generate prefill)
COMMON=(-isystem $HOME/shim/x86compat -isystem $HOME/simde -DSTRATA_NATIVE_EXPERTS=1 '-DSTRATA_VERSION="0.1.41"' -I$HOME/strata-linux/_deps/strata_llamacpp-src/ggml/src/../include -isystem $HOME/cuda-12.8/include -O3 -DNDEBUG -std=c++20 -Wall -Wextra -isystem $HOME/shim/include -isystem $HOME/cuda-12.8/include -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk -mcpu=apple-m4 -DSTRATA_TGCU=1 -D_DARWIN_C_SOURCE -Wno-unknown-cuda-version -Wno-everything)
rm -f strata.next strata.next.tmp
pids=(); names=()
for o in $what; do
  case $o in
    verify|expert_source|layer) ( nice -n 10 clang++ -x c++ -include $LI/strata/platform/macos_compat.hpp -I$LI $COMMON -c $D/src/core/$o.cpp -o obj/$o.cpp.o.tmp && mv obj/$o.cpp.o.tmp obj/$o.cpp.o ) > obj/$o.log 2>&1 & ;;
    generate) ( nice -n 10 clang++ -x c++ -include $LI/strata/platform/macos_compat.hpp -DSTRATA_PREFILL_FUSED=1 -DSTRATA_PREFILL_MMQ=1 -I$LI $COMMON -c $D/src/program/generate.cpp -o obj/generate.cpp.o.tmp && mv obj/generate.cpp.o.tmp obj/generate.cpp.o ) > obj/generate.log 2>&1 & ;;
    prefill) ( nice -n 10 clang++ -x c++ -include $XI/strata/platform/macos_compat.hpp -DSTRATA_PREFILL_FUSED=1 -DSTRATA_PREFILL_MMQ=1 -I$XI $COMMON -c $D/src/prefill/prefill.cpp -o obj/prefill.cpp.o.tmp && mv obj/prefill.cpp.o.tmp obj/prefill.cpp.o ) > obj/prefill.log 2>&1 & ;;
    *) echo "unknown object $o"; exit 2 ;;
  esac
  pids+=($!); names+=($o)
done
fail=0
for i in {1..${#pids}}; do wait ${pids[$i]} || { echo "FAILED: ${names[$i]}"; grep -a "error" obj/${names[$i]}.log | head -30; fail=1; }; done
[ $fail -eq 0 ] || exit 1
for o in verify expert_source layer prefill generate; do [ -s obj/$o.cpp.o ] || { echo "object $o missing (build all four once)"; exit 1; }; done
cp $L/libstrata_engine.a libstrata_engine.a
for o in verify expert_source layer; do cp obj/$o.cpp.o $o.cpp.o; ar r libstrata_engine.a $o.cpp.o 2>/dev/null; rm -f $o.cpp.o; done
ranlib libstrata_engine.a
cp $X/libstrata_prefill.a libstrata_prefill.a
cp obj/prefill.cpp.o prefill.cpp.o; ar r libstrata_prefill.a prefill.cpp.o 2>/dev/null; rm -f prefill.cpp.o; ranlib libstrata_prefill.a
sed -e "s#$L/libstrata_engine.a#$B/libstrata_engine.a#g" -e "s#$X/libstrata_prefill.a#$B/libstrata_prefill.a#g" -e "s#-o $X/strata.next.tmp#-o $B/strata.next.tmp#" -e "s#$L/CMakeFiles/strata.dir/src/program/generate.cpp.o#$B/obj/generate.cpp.o#" $X/cmd_link.sh > cmd_link.sh
grep -q "strata.next.tmp" cmd_link.sh || { echo "cmd_link.sh rewrite failed"; exit 1; }
grep -q "$B/obj/generate.cpp.o" cmd_link.sh || { echo "generate object not in link"; exit 1; }
nice -n 10 zsh cmd_link.sh 2> link.err || { cat link.err; exit 1; }
grep -v "duplicate libraries" link.err || true
[ -s strata.next.tmp ] || { echo "link produced no binary"; exit 1; }
mv strata.next.tmp strata.next
ls -la strata.next; echo "strata.next md5 $(md5 -q strata.next)"
