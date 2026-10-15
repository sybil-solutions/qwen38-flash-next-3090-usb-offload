#!/bin/zsh
# pf18.sh <cfg> <tag> [sizes...] (ON the mini, server up on <cfg>, which reads /tmp/d7flag and /tmp/d18cfg): prefill rows with the flag
# interleaved ABBA per size (A = $FLAG_A default 7 = config C behaviour, B = $FLAG_B default 15 = lazy refill), one request per
# pf600_bench call (a ~1.5 s gap, like a client), then the agent-turn 'small' pattern for each flag.  Output: ~/shim/pf600_<tag>_f<flag>.jsonl
cfg=$1; tag=$2; shift 2
sizes=${@:-2500 8000 17000 33000}
fa=${FLAG_A:-7}; fb=${FLAG_B:-15}
export PF600_CFG=$cfg
zsh ~/shim/sleep_guard.sh || exit 1
echo $fb > /tmp/d7flag; python3 ~/shim/pf600_bench.py rows 2500 --runs 1 --tag ${tag}_warmup > /dev/null 2>&1
for sz in ${=sizes}; do
  for fl in $fa $fb $fb $fa; do
    echo $fl > /tmp/d7flag
    python3 ~/shim/pf600_bench.py rows $sz --runs 1 --tag ${tag}_f$fl 2>&1 | grep '^{' | cut -c1-230 | sed "s/^/flag $fl /"
  done
done
if [ -z "$NO_SMALL" ]; then
for fl in $fa $fb $fb $fa; do
  echo $fl > /tmp/d7flag
  echo "== small (agent-turn pattern) flag $fl"
  python3 ~/shim/pf600_bench.py small --tag ${tag}_f$fl 2>&1 | cut -c1-200
done
fi
