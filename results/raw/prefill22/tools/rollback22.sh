#!/bin/zsh
# rollback22.sh (ON the mini): config F -> config E: restore the backed-up E file as the live file, restart (strata_d19b + ~/shim stack), verify a request.  Never touches ~/shim code or strata_d19b.
P=~/strata-prefill22; LIVE=~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json; BK=$P/backup/strata-exl3-full512-final-g4-lru-ht-prefix-p600.configE.json
[ "$(md5 -q $BK)" = 078235c5bfec8bac85dc0a39d562485d ] || { echo "backup md5 wrong"; exit 5; }
cp $BK $LIVE
zsh ~/strata-prefill21/tools/restore21.sh
