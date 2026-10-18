#!/bin/zsh
# arm21.sh <cfg> <tag> (ON the mini, card owner): fresh restart on <cfg>, realistic decode set (29, flag 7), probes (d2_probe x2 = 12 requests), prefill rows 2.5K/8K/18K/33K x2, agent-turn rows ('small'), swap readout.
# Outputs ~/strata-prefill21/logs/abba_<tag>.{deploy.log,jsonl,probe.log,rows.log,small.log,engine.log,out}.  Stops (exit 3) when a request I did not send reached the server.
D=~/strata-prefill21; P=$D/logs/abba_$1; cfg=$2; tag=$1
cfg=$2
elog=~/shim/serve-engine.log; : > $P.out
zsh ~/shim/sleep_guard.sh || exit 1
zsh $D/tools/usercheck21.sh >> $P.out 2>&1 || { echo "USER ACTIVE before $tag" >> $P.out; exit 3; }
echo 7 > /tmp/d7flag
zsh $D/tools/restart21.sh $cfg > $P.deploy.log 2>&1 || { echo "DEPLOY FAILED $tag" >> $P.out; tail -5 $P.deploy.log >> $P.out; exit 1; }
zsh $D/tools/mark21.sh; mark=$(wc -l < $elog)
export PF600_CFG=$cfg XP8_CFG=$cfg
st(){ echo "$1 $(date +%T) wired_MiB $(( $(vm_stat | awk '/wired down/{print $4}' | tr -d .) * 16 / 1024 )) swap: $(sysctl -n vm.swapusage | tr -s ' ')" >> $P.out; zsh $D/tools/mark21.sh; }
chk(){ zsh $D/tools/usercheck21.sh >> $P.out 2>&1 || { echo "USER ACTIVE during $tag ($1)" >> $P.out; exit 3; }; }
st ready; echo "config md5 $(md5 -q $cfg) engine $(ps -o command= -p $(pgrep -f 'strata_' | head -1) | grep -o '[^ ]*/strata_[a-z0-9]*' | head -1)" >> $P.out
rm -f $P.jsonl; chk seq
python3 ~/strata-decode7/run_seq.py $cfg /tmp/d7flag $P.jsonl $(cat ~/strata-decode7/seq_on.txt) >> $P.out 2>&1; st after_seq; chk probes
: > $P.probe.log
for i in 1 2; do python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-80 >> $P.probe.log; done; st after_probe; chk rows
python3 ~/shim/pf600_bench.py rows 2500 8000 18000 33000 --runs 2 --tag abba_$tag 2>&1 | grep '^{' | cut -c1-240 > $P.rows.log; st after_rows; chk small
python3 ~/shim/pf600_bench.py small --tag abba_$tag 2>&1 | cut -c1-200 > $P.small.log; st after_small
tail -n +$((mark+1)) $elog > $P.engine.log
echo "$tag done: $(wc -l < $P.jsonl) seq requests, probe mean $(awk '/decode tok\/s/{s+=$3;n++}END{printf "%.2f (%d)", s/n, n}' $P.probe.log), GPU timeouts $(grep -ac 'GPU timeout' $P.engine.log)" >> $P.out
