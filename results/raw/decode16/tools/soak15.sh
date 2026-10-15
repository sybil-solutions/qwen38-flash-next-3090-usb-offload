#!/bin/zsh
# soak15.sh <cfg.json> [n=5] (ON the mini): soak_guard.sh logic on the ALREADY-RUNNING server: n successive ~32K needle prompts, each under a perl alarm; STOPS at the first failed request, GPU timeout or dead engine.
zsh ~/shim/sleep_guard.sh || exit 1
export XP8_CFG=$1; n=${2:-5}
key=$(python3 -c "import json,os;print(json.load(open(os.environ[\"XP8_CFG\"]))[\"api_key\"])")
elog=~/shim/serve-engine.log; mark=$(wc -l < $elog); d=(10 50 90 30 70)
for i in $(seq 1 $n); do
  out=$(perl -e "alarm 900; exec @ARGV" python3 ~/strata/tools/needle_bench.py --lengths 32k --depths ${d[$i]} --api-key "$key" 2>&1 | tail -1)
  echo "prompt $i: $out"
  if tail -n +$((mark+1)) $elog | grep -aq "GPU timeout"; then echo "GPU TIMEOUT after prompt $i: stopping"; break; fi
  case "$out" in *"1 of 1"*) ;; *) echo "request failed: stopping"; break;; esac
  pgrep -f strata_d1 >/dev/null || { echo "ENGINE DEAD"; break; }
done
echo "engine alive: $(pgrep -f strata_d1 | wc -l)"
grep -a "serve: prompt" $elog | tail -$n | cut -c1-200
echo "free VRAM lines:"; tail -n +$((mark+1)) $elog | grep -a "VRAM free\|K/V grown\|K/V trimmed\|OOM\|ERR\|out of memory" | cut -c1-220 | tail -12
