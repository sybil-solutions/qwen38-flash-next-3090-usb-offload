#!/bin/zsh
# dec18.sh <cfg> <tag> (ON the mini, server up on <cfg>): the in-server A/B decode arm: 58 realistic requests (29 prompts x 2 passes, flag
# alternating FLAG_A / FLAG_B, mkseq18.py), then the 12 short probes at flag B.  Outputs ~/strata-prefill18/logs/<tag>.jsonl / .probe.log
cfg=$1; tag=$2; fa=${FLAG_A:-7}; fb=${FLAG_B:-15}
D=~/strata-prefill18/logs; elog=~/shim/serve-engine.log
zsh ~/shim/sleep_guard.sh || exit 1
mark=$(wc -l < $elog); rm -f $D/$tag.jsonl
items=$(python3 ~/strata-prefill18/tools/mkseq18.py $tag $fa $fb 300)
python3 ~/strata-decode7/run_seq.py $cfg /tmp/d7flag $D/$tag.jsonl ${=items} > $D/$tag.out 2>&1
echo $fb > /tmp/d7flag
: > $D/$tag.probe.log
for i in 1 2; do python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-80 >> $D/$tag.probe.log; done
tail -n +$((mark+1)) $elog > $D/$tag.engine.log
grep -a "ERR\|GPU timeout" $D/$tag.engine.log | cut -c1-200 | tail -5 >> $D/$tag.out
python3 ~/strata-prefill18/tools/sum18.py $D/$tag.jsonl $D/$tag.probe.log
