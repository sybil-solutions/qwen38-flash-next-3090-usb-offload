#!/bin/zsh
# abconc15c.sh <tag> <arm:cfg.json>... (ON the mini, card owner): like abconc15b.sh with two phases per arm, selected by env: PHASES="real short" (default both).
#   real  = conc15.py at c = 1, 2, 4 over $ITEMS (default: the 4 realistic prompts p26 p22 p10 p15, 300 tok): prefill-heavy (6-9K prompts interleave with the other slots decoding);
#   short = conc15.py at c = 1, 2, 4 over 8 short (<= 30 token) prompts, 300 tok: the batch decode rate itself (no long prefill in flight).
# aggregate = generated tokens / (last end - first start) of the whole run, TTFT included.  Stops at the first fault (GPU timeout, ERR, dead engine, missing summary).  Log ~/strata-decode15/logs/abconc_<tag>.out, jsonl per (arm, phase, c).
zsh ~/shim/sleep_guard.sh || exit 1
D=~/strata-decode15; O=$D/logs/abconc_$1.out; : > $O; tag=$1; shift
phases=(${=PHASES:-real short})
real=(${=ITEMS:-p26:7:300 p22:7:300 p10:7:300 p15:7:300})
short=("Write a Python function that checks whether a number is prime.:7:300" "Explain how a mixture of experts gating network routes tokens.:7:300" "Write a short story about a lighthouse keeper who stops speaking.:7:300" "Describe how a hash table handles collisions.:7:300" "Give a detailed summary of how TCP congestion control works.:7:300" "Write a poem about the sea at night in four stanzas.:7:300" "Explain the difference between a process and a thread.:7:300" "List and describe the layers of the OSI model.:7:300")
for pair in "$@"; do
  arm=${pair%%:*}; cfg=${pair#*:}
  echo "### $arm $(basename $cfg) $(date +%T)" >> $O
  zsh $D/tools/restart15.sh $cfg > $D/logs/abconc_$arm.deploy.log 2>&1 || { echo "DEPLOY FAILED $arm" >> $O; tail -5 $D/logs/abconc_$arm.deploy.log >> $O; continue; }
  grep -a "batch\|slot sessions\|elastic K/V\|expert cache auto\|expert cache [0-9]" ~/shim/serve-engine.log | tail -6 | cut -c1-200 >> $O
  export PF600_CFG=$cfg; elog=~/shim/serve-engine.log; mark=$(wc -l < $elog)
  for ph in $phases; do
    for c in 1 2 4; do
      f=$D/logs/abconc_${arm}_${ph}_c$c.jsonl; rm -f $f
      if [ $ph = real ]; then its=("${real[@]}"); else its=("${short[@]}"); fi
      echo "[$ph]" >> $O
      perl -e "alarm 2400; exec @ARGV" python3 $D/tools/conc15.py $cfg $c $f "${its[@]}" 2>&1 | tail -1 >> $O
      if tail -n +$((mark+1)) $elog | grep -aq "GPU timeout\|ERR verify\|ERR BGEN\|ERR batch"; then echo "FAULT in $arm $ph c=$c: stopping" >> $O; tail -n +$((mark+1)) $elog | grep -a "GPU timeout\|ERR " | head -5 | cut -c1-240 >> $O; echo ABCONC15CDONE-FAULT >> $O; exit 3; fi
      pgrep -f strata_d1 > /dev/null || { echo "ENGINE DEAD in $arm $ph c=$c: stopping" >> $O; echo ABCONC15CDONE-FAULT >> $O; exit 3; }
      grep -q "CONC c=$c" <(tail -3 $O) || { echo "conc run $ph c=$c produced no summary: stopping" >> $O; echo ABCONC15CDONE-FAULT >> $O; exit 3; }
    done
  done
  python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-80 >> $O
  grep -ac "GPU timeout" ~/shim/serve-engine.log | sed "s/^/GPU timeouts in log: /" >> $O
done
echo ABCONC15CDONE >> $O
