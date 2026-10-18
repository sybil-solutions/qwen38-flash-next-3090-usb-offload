#!/bin/zsh
# soak21.sh <cfg> [n=5] (ON the mini, card owner): soak_guard.sh's logic on the Prefill21/22 restart path: fresh restart on <cfg>, then n successive ~32K needle prompts (depths 10 50 90 30 70), each under a perl alarm;
# STOPS at the first failed request or the first 'GPU timeout' line of the engine log, or at a foreign request.  Log logs/soak21.out.
D=~/strata-prefill21; O=$D/logs/soak21.out; : > $O
zsh ~/shim/sleep_guard.sh || exit 1
zsh $D/tools/usercheck21.sh >> $O 2>&1 || { echo "user active" >> $O; exit 3; }
cfg=$1; n=${2:-5}
zsh $D/tools/restart21.sh $cfg > $D/logs/soak21.deploy.log 2>&1 || { echo DEPLOY FAILED >> $O; exit 1; }
zsh $D/tools/mark21.sh
export XP8_CFG=$cfg
key=$(python3 -c "import json,os;print(json.load(open(os.environ['XP8_CFG']))['api_key'])")
elog=~/shim/serve-engine.log; mark=$(wc -l < $elog)
d=(10 50 90 30 70)
for i in $(seq 1 $n); do
  zsh $D/tools/usercheck21.sh >> $O 2>&1 || { echo "user active before prompt $i: stopping" >> $O; exit 3; }
  out=$(perl -e 'alarm 900; exec @ARGV' python3 ~/strata/tools/needle_bench.py --lengths 32k --depths ${d[$i]} --api-key "$key" 2>&1 | tail -1)
  zsh $D/tools/mark21.sh
  echo "prompt $i: $out" >> $O
  if tail -n +$((mark+1)) $elog | grep -aq "GPU timeout"; then echo "GPU TIMEOUT after prompt $i: stopping" >> $O; break; fi
  case "$out" in *"1 of 1"*) ;; *) echo "request failed: stopping" >> $O; break;; esac
done
echo "engine alive: $(pgrep -f 'strata_[a-z0-9]* --serve' | wc -l); GPU timeouts $(tail -n +$((mark+1)) $elog | grep -ac 'GPU timeout')" >> $O
echo SOAK21DONE >> $O
