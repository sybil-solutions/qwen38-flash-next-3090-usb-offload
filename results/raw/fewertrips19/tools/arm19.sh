#!/bin/zsh
# arm15.sh <cfg> <tag> [seqfile] (ON the mini, card owner): restart on <cfg> (fresh LRU), record placement/slots/swap, run the realistic request sequence (29 prompts, STRATA_D7_FLAG file = 7),
# then the short-probe pair (d2_probe x2 = 12 requests) and the swap/wired readout.  Outputs ~/strata-fewertrips19/logs/<tag>.{deploy.log,jsonl,engine.log,probe.log,out}.
zsh ~/shim/sleep_guard.sh || exit 1
cfg=$1; tag=$2; seq=${3:-~/strata-decode13/seq10.txt}
D=~/strata-fewertrips19/logs; elog=~/shim/serve-engine.log; : > $D/$tag.out
echo 7 > /tmp/d7flag
zsh ~/strata-decode15/tools/restart15.sh $cfg > $D/$tag.deploy.log 2>&1 || { echo DEPLOY FAILED >> $D/$tag.out; tail -5 $D/$tag.deploy.log >> $D/$tag.out; exit 1; }
mark=$(wc -l < $elog)
st(){ echo "$1 $(date +%T) wired_MiB $(( $(vm_stat | awk '/wired down/{print $4}' | tr -d .) * 16 / 1024 )) swap: $(sysctl -n vm.swapusage | tr -s ' ') engine_swapped: $(vmmap -summary $(pgrep -x strata) 2>/dev/null | awk '/^MALLOC_LARGE  /{print "MALLOC_LARGE " $5}')" >> $D/$tag.out; echo "$1 mem: swapins $(vm_stat | awk '/Swapins/{print $2}') swapouts $(vm_stat | awk '/Swapouts/{print $2}') compr_occupied_pages $(vm_stat | awk '/occupied by compressor/{print $5}') free_pct $(memory_pressure 2>/dev/null | awk '/free percentage/{print $5}')" >> $D/$tag.out }
st ready
rm -f $D/$tag.jsonl
python3 ~/strata-decode7/run_seq.py $cfg /tmp/d7flag $D/$tag.jsonl $(cat $seq) >> $D/$tag.out 2>&1
st after_seq
: > $D/$tag.probe.log
for i in 1 2; do python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-80 >> $D/$tag.probe.log; done
st after_probe
if [ -n "$ARM8_IMAGE" ]; then export XP8_CFG=$cfg PF600_CFG=$cfg; python3 ~/strata-xport8/tools/regress_extra.py image 2>&1 | cut -c1-300 > $D/$tag.image.log; st after_image; fi
tail -n +$((mark+1)) $elog > $D/$tag.engine.log
echo "$tag done: $(wc -l < $D/$tag.jsonl) requests, probe mean $(awk '/decode tok\/s/{s+=$3;n++}END{printf "%.2f (%d)", s/n, n}' $D/$tag.probe.log)" | tee -a $D/$tag.out
grep -a "K/V grown\|K/V trimmed\|elastic K/V\|ERR\|GPU timeout" $D/$tag.engine.log | cut -c1-260 | tail -20 >> $D/$tag.out
