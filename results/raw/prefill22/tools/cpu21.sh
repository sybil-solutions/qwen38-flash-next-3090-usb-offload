#!/bin/zsh
# cpu21.sh <cfg> [secs=150] [seeds...] (ON the mini): CPU-load USB byte-exact check of the shim levers of <cfg> (its TGCU_* env, fence default 11): stop the server (card free), 8 CPU hogs, stress21.py (random sizes/offsets/patterns, h2d + d2h byte compare) per seed, then restore config E (the single restore path).
D=~/strata-prefill21; cfg=$1; secs=${2:-150}; shift 2; seeds=(${@:-1 2})
O=$D/logs/cpu21.out; : > $O
zsh $D/tools/stop21.sh >> $O 2>&1 || { echo "stop failed" >> $O; exit 3; }
sleep 15
source ~/.chestnut_env
eval "$(python3 $D/tools/envof.py $cfg)"
SITE=$(~/tinygrad/.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
export DEV=NV:PTX PYTHONPATH=$HOME/strata-prefill22/shim:$HOME/tg-strata:$SITE PYTHONNOUSERSITE=1 DYLD_LIBRARY_PATH=$HOME/strata-prefill22/shim
env | grep -a "^TGCU_USB\|^TGCU_LAZY" | sort >> $O
cd /tmp
for seed in $seeds; do
  for i in 1 2 3 4 5 6 7 8; do (perl -e "alarm $((secs+20)); while(1){}" &) ; done
  echo "== seed $seed, 8 CPU hogs, $(date +%T)" >> $O
  for try in 1 2 3; do
    STRESS_SECS=$secs perl -e "alarm $((secs+120)); exec @ARGV" ~/tinygrad/.venv/bin/python $D/tools/stress21.py 100000 $seed 2>&1 | tail -5 > $D/logs/cpu21.seed
    grep -q "Failed to acquire lock" $D/logs/cpu21.seed || break
    echo "(lock still held, retry $try)" >> $O; sleep 20
  done
  cat $D/logs/cpu21.seed >> $O
  pkill -f "while(1){}"; sleep 3
done
sleep 1; echo "== restore $(date +%T)" >> $O
zsh $D/tools/restore21.sh >> $O 2>&1
echo CPU21DONE >> $O
