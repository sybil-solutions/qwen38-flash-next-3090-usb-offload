#!/bin/zsh
# abconc15b.sh <tag> <arm:cfg.json>... (variant of abconc15.sh: an explicit arm list instead of the fixed ctlA batA batB ctlB)
# abconc15.sh <ctl.json> <batch.json> <tag> (ON the mini, card owner): concurrency ABBA = ctlA batchA batchB ctlB; each arm = restart + conc15.py at c = 1, 2, 4 over the same 8 realistic prompts (p26 p22 p10 p15 p11 p06 p02 p24, 300 tok),
# then 12 short probes (c=1).  The ctl arm has no --batch: concurrent requests queue (FIFO), so its aggregate is the solo rate.  Log ~/strata-decode15/logs/abconc_<tag>.out, one jsonl per (arm, c).
zsh ~/shim/sleep_guard.sh || exit 1
D=~/strata-decode15; O=$D/logs/abconc_$1.out; : > $O; tag=$1; shift
items=(p26:7:300 p22:7:300 p10:7:300 p15:7:300 p11:7:300 p06:7:300 p02:7:300 p24:7:300)
for pair in "$@"; do
  arm=${pair%%:*}; cfg=${pair#*:}
  echo "### $arm $(basename $cfg) $(date +%T)" >> $O
  zsh $D/tools/restart15.sh $cfg > $D/logs/abconc_$arm.deploy.log 2>&1 || { echo "DEPLOY FAILED $arm" >> $O; tail -5 $D/logs/abconc_$arm.deploy.log >> $O; continue; }
  grep -a "batch\|slot sessions\|kv-grow\|elastic K/V\|VRAM free\|expert cache auto\|expert cache [0-9]" ~/shim/serve-engine.log | tail -12 | cut -c1-200 >> $O
  export PF600_CFG=$cfg
  elog=~/shim/serve-engine.log; mark=$(wc -l < $elog)
  for c in 1 2 4; do
    rm -f $D/logs/abconc_${arm}_c$c.jsonl
    perl -e 'alarm 2400; exec @ARGV' python3 $D/tools/conc15.py $cfg $c $D/logs/abconc_${arm}_c$c.jsonl $items 2>&1 | tail -1 >> $O
    # soak_guard rule: stop at the first fault (a GPU timeout line, a dead engine or a failed request) - no more load on a faulted card
    if tail -n +$((mark+1)) $elog | grep -aq "GPU timeout\|ERR verify\|ERR BGEN\|ERR batch"; then echo "FAULT in $arm c=$c: stopping" >> $O; tail -n +$((mark+1)) $elog | grep -a "GPU timeout\|ERR " | head -5 | cut -c1-240 >> $O; echo ABCONC15BDONE-FAULT >> $O; exit 3; fi
    pgrep -f strata_d1 > /dev/null || { echo "ENGINE DEAD in $arm c=$c: stopping" >> $O; echo ABCONC15BDONE-FAULT >> $O; exit 3; }
    grep -q "CONC c=$c" <(tail -3 $O) || { echo "conc run c=$c produced no summary: stopping" >> $O; echo ABCONC15BDONE-FAULT >> $O; exit 3; }
  done
  python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-80 >> $O
  grep -ac "GPU timeout" ~/shim/serve-engine.log | sed 's/^/GPU timeouts in log: /' >> $O
done
echo ABCONC15BDONE >> $O
