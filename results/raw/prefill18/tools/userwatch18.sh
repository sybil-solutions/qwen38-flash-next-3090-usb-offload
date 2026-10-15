#!/bin/zsh
# userwatch18.sh (ON the mini): every 2 s look for a NON-loopback peer on :8080 (the user via Pi; every client of mine is 127.0.0.1).
# Two hits in a row: stop every Prefill18 script, restore config C (live file md5 bc8450a1), leave ~/strata-prefill18/USER_ACTIVE.
D=~/strata-prefill18; rm -f $D/USER_ACTIVE; hits=0
while true; do
  p=$(netstat -an | awk '$4 ~ /\.8080$/ && $6=="ESTABLISHED" && $5 !~ /^127\.0\.0\.1\./ {print $5}' | head -1)
  if [ -n "$p" ]; then hits=$((hits+1)); else hits=0; fi
  if [ $hits -ge 2 ]; then
    echo "$(date +%T) USER peer $p" > $D/USER_ACTIVE
    pkill -f "pf18.sh|dec18.sh|ab18.sh|run_seq.py|pf600_bench|needle_bench|soak18|final18|q18.sh|d2_probe|split_probe2|regress_extra"
    sleep 3; zsh ~/strata-decode15/tools/restart15.sh ~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json > $D/logs/restore_user.log 2>&1
    exit 0
  fi
  sleep 2
done
