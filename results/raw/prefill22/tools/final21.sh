#!/bin/zsh
# final21.sh <cfg> (ON the mini, card owner): the Prefill22 regression on <cfg> (fresh restart): realistic decode set (29), decode probes x2 (12 requests), tool reuse, image, prefill rows 2.5K/8K/17K/33K x2, ~190K needle,
#   agent-turn 'small' pattern, 5 successive ~32K needles, decode probe after; swap / wired readout after every stage.  Log ~/shim/d22_final.log.  User activity (foreign request) -> stop (exit 3).
L=~/shim/d22_final.log; : > $L; D=~/strata-prefill21
cfg=$1
st(){ echo "SWAP $1 $(date +%T) wired_MiB $(( $(vm_stat | awk '/wired down/{print $4}' | tr -d .) * 16 / 1024 )) swap: $(sysctl -n vm.swapusage | tr -s ' ') engine MALLOC_LARGE swapped: $(vmmap -summary $(pgrep -f 'strata_[a-z0-9]* --serve' | head -1) 2>/dev/null | awk '/^MALLOC_LARGE  /{print $5}')" >> $L; zsh $D/tools/mark21.sh; }
chk(){ zsh $D/tools/usercheck21.sh >> $L 2>&1 || { echo "USER ACTIVE ($1): stopping" >> $L; exit 3; }; }
echo "config md5 $(md5 -q $cfg)  binary md5 $(md5 -q $(python3 -c "import json;print(json.load(open('$cfg'))['env'].get('D7_BIN',''))"))" >> $L
echo "### start $(date +%T)" >> $L
zsh $D/tools/restart21.sh $cfg >> $L 2>&1 || { echo "START FAILED" >> $L; exit 1; }
zsh $D/tools/mark21.sh; st ready
zsh ~/shim/sleep_guard.sh || exit 1
export PF600_CFG=$cfg XP8_CFG=$cfg
key=$(python3 -c "import json,os;print(json.load(open(os.environ['XP8_CFG']))['api_key'])")
elog=~/shim/serve-engine.log; mark0=$(wc -l < $elog)
echo "### realistic decode set (29 prompts, 300 tok) $(date +%T)" >> $L
echo 7 > /tmp/d7flag; rm -f ~/shim/d22_final_seq.jsonl; chk seq
python3 ~/strata-decode7/run_seq.py $cfg /tmp/d7flag ~/shim/d22_final_seq.jsonl $(cat ~/strata-decode7/seq_on.txt) 2>&1 | cut -c1-200 >> $L
python3 -c "
import json,statistics as s
r=[json.loads(l) for l in open('$HOME/shim/d22_final_seq.jsonl')]
print('REALISTIC29 requests',len(r),'mean tok/s %.2f'%s.mean(x['tok_s'] for x in r),'ttft mean %.2f'%s.mean(x['ttft_s'] for x in r),'wall mean %.2f'%s.mean(x['wall_s'] for x in r))" >> $L
st after_realistic; chk probes
echo "### decode probes x2 (12 requests) $(date +%T)" >> $L
for r in a b; do python3 ~/shim/split_probe2.py d22final$r 300 2>&1 | cut -c1-200 >> $L; python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-80 >> $L; done
st after_decode; chk toolreuse
echo "### tool reuse / image $(date +%T)" >> $L
python3 ~/strata-xport8/tools/regress_extra.py toolreuse 2>&1 | cut -c1-300 >> $L; st after_toolreuse
python3 ~/strata-xport8/tools/regress_extra.py image 2>&1 | cut -c1-400 >> $L; st after_image; chk rows
echo "### prefill rows $(date +%T)" >> $L
python3 ~/shim/pf600_bench.py rows 2500 8000 17000 33000 --runs 2 --tag d22_final 2>&1 | grep '^{' | cut -c1-220 >> $L; st after_prefill; chk needle190
echo "### needle ~190K $(date +%T)" >> $L
perl -e 'alarm 3700; exec @ARGV' python3 ~/strata/tools/needle_bench.py --lengths 190k --depths 50 --api-key "$key" --timeout 3600 --out ~/shim/d22_needle190.json 2>&1 | grep -a "found\|FOUND\|miss\|MISS\|needle" | tail -2 | cut -c1-200 >> $L; st after_190k; chk small
echo "### small (agent-turn pattern) $(date +%T)" >> $L
python3 ~/shim/pf600_bench.py small --tag d22_final 2>&1 | cut -c1-200 >> $L; st after_small; chk needle5
echo "### needle x5 $(date +%T)" >> $L
d=(10 50 90 30 70); mark=$(wc -l < $elog)
for i in 1 2 3 4 5; do
  perl -e 'alarm 900; exec @ARGV' python3 ~/strata/tools/needle_bench.py --lengths 32k --depths ${d[$i]} --api-key "$key" 2>&1 | grep -a "found\|FOUND\|miss\|MISS\|needle" | tail -2 | cut -c1-160 >> $L
  tail -n +$((mark+1)) $elog | grep -aq "GPU timeout" && { echo "GPU TIMEOUT" >> $L; break; }
done
echo "GPU timeouts in the whole regression: $(tail -n +$((mark0+1)) $elog | grep -ac 'GPU timeout')" >> $L
st after_needles
echo "-- decode probe after the regression" >> $L; python3 ~/strata-decode2/d2_probe.py $cfg 2>&1 | cut -c1-70 >> $L
st end
echo "-- elastic K/V / ERR events" >> $L; tail -n +$((mark0+1)) $elog | grep -a "strata d18\|K/V grown\|K/V trimmed\|elastic K/V\|ERR\|GPU timeout" | cut -c1-260 | tail -60 >> $L
echo FINAL22DONE >> $L
