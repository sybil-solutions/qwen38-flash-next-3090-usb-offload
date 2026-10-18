#!/bin/zsh
# restore22.sh (ON the mini): restart whatever the live config file is (config F after the promotion), verify a request is served, refresh the engine-log mark.
LIVE=~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json; D=~/strata-prefill21
zsh $D/tools/restart21.sh $LIVE --restore > ~/strata-prefill22/logs/restore22.log 2>&1 || { echo "restore22: restart failed"; tail -5 ~/strata-prefill22/logs/restore22.log; exit 1; }
key=$(python3 -c "import json;print(json.load(open('$LIVE'))['api_key'])")
out=$(perl -e 'alarm 120; exec @ARGV' curl -s -m 100 http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $key" -H "Content-Type: application/json" -d '{"model":"x","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}')
echo "$out" | grep -q '"choices"' || { echo "restore22: verify failed: $(echo $out | cut -c1-160)"; exit 2; }
wc -l < ~/shim/serve-engine.log > $D/logmark; echo "restore22: OK live md5 $(md5 -q $LIVE)"
