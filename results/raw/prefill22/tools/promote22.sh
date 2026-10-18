#!/bin/zsh
# promote22.sh (ON the mini): promote config F: back up the live config E file, install configF.json as the live file, restart through the serialised restart path, verify a request is served by the F engine.
# Rollback: zsh ~/strata-prefill22/rollback22.sh (copies the backup back, restarts config E / strata_d19b / ~/shim stack).
D=~/strata-prefill21; P=~/strata-prefill22; LIVE=~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json; BK=$P/backup/strata-exl3-full512-final-g4-lru-ht-prefix-p600.configE.json
EMD5=078235c5bfec8bac85dc0a39d562485d; FMD5=b8d72f718d1101508ccd5fbc15255b76
zsh $D/tools/usercheck21.sh || exit 3
mkdir -p $P/backup
[ "$(md5 -q $LIVE)" = $EMD5 ] && cp -p $LIVE $BK || [ "$(md5 -q $BK)" = $EMD5 ] || { echo "no config E to back up (live md5 $(md5 -q $LIVE))"; exit 5; }
cp -p ~/shim/tgcu.py ~/shim/libcudart.dylib $P/backup/ 2>/dev/null   # the E shim files (untouched by this promotion, kept for the record)
[ "$(md5 -q $P/cfg/configF.json)" = $FMD5 ] || { echo "configF.json md5 changed"; exit 5; }
cp $P/cfg/configF.json $LIVE
zsh $D/tools/restart21.sh $LIVE --restore > $P/logs/promote.deploy.log 2>&1 || { echo "F did not start: rolling back"; zsh $P/rollback22.sh; exit 1; }
key=$(python3 -c "import json;print(json.load(open('$LIVE'))['api_key'])")
out=$(perl -e 'alarm 120; exec @ARGV' curl -s -m 100 http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $key" -H "Content-Type: application/json" -d '{"model":"x","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8}')
echo "$out" | grep -q '"choices"' || { echo "F verify request failed: rolling back"; zsh $P/rollback22.sh; exit 2; }
pid=$(pgrep -f 'strata_p22 --serve' | head -1); bin=$(ps -o command= -p $pid | grep -o '[^ ]*strata_p22' | head -1)
echo "engine pid $pid $bin md5 $(md5 -q $bin); live config md5 $(md5 -q $LIVE); reply: $(echo $out | cut -c1-160)"
wc -l < ~/shim/serve-engine.log > $D/logmark
echo "promote22: OK F live $(date +%T)"
