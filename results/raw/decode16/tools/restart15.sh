#!/bin/zsh
# restart15.sh <config.json> (ON the mini, card owner): ls_srv.sh with a proper wait: the old server + engine (the engine binary is named strata_d12a / strata_d14a, so ls_srv's `pgrep -x strata` never waits) must be gone and port 8080 free before the new one starts.
[ -f ~/shim/server.pid ] && kill -TERM $(cat ~/shim/server.pid) 2>/dev/null
for i in {1..180}; do pgrep -f 'strata_d1|bin/strata|serve/server.py' >/dev/null || break; sleep 1; done
pgrep -f 'strata_d1|bin/strata|serve/server.py' >/dev/null && { echo "old server/engine still alive after 180 s"; exit 2; }
for i in {1..30}; do lsof -nP -iTCP:8080 -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
sleep 2
exec zsh ~/shim/ls_srv.sh "$@"
