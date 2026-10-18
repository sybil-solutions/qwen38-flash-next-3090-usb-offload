#!/bin/zsh
# chain21_e.sh (ON the mini): Prefill22 round 2 after the pool finding: config F = candF WITHOUT TGCU_LAZY_POOL (the pool crashes the engine in the teacher-forced flow, 'bulk IN short read' in hs_wait, deterministic; every arm without it passes).
# Fresh-restart ABBA E vs F, gate, regression, teacher-forced quality, guarded soak, CPU-load USB check (restores config E at its end).  Any foreign request: stop + restore E.
D=~/strata-prefill21; O=$D/logs/chain_e.out; : > $O; F=$HOME/strata-prefill22/cfg/configF.json
mkdir -p $D/logs/pool_run; for t in E1 C1 C2 E2; do mv $D/logs/abba_$t.* $D/logs/pool_run/ 2>/dev/null; done
mv $D/logs/d22_final_F.log $D/logs/q21_F.log $D/logs/soak21.out $D/logs/cpu21.out $D/logs/pool_run/ 2>/dev/null
[ -f $HOME/strata-prefill22/cfg/configF_pool.json ] || mv $F $HOME/strata-prefill22/cfg/configF_pool.json
python3 - <<'PY'
import json, os
h = os.path.expanduser("~"); c = json.load(open(h + "/strata-prefill22/cfg/configF_pool.json"))
c["env"].pop("TGCU_LAZY_POOL", None)
json.dump(c, open(h + "/strata-prefill22/cfg/configF.json", "w"), indent=1)
PY
echo "config F (no pool) md5 $(md5 -q $F)" >> $O
stage(){ [ -e $D/USER_ACTIVE ] && { echo "USER_ACTIVE before $1" >> $O; exit 3; }; echo "### $1 $(date +%T)" >> $O; }
stage abba; zsh $D/tools/chain21_b.sh $F; cat $D/logs/chain_b.out >> $O
python3 $D/tools/gate22.py >> $O 2>&1 || { echo "ABBA gate NO-GO: leaving the server on config E" >> $O; exit 6; }
stage regression; zsh $D/tools/final21.sh $F >> $O 2>&1; rc=$?; cp ~/shim/d22_final.log $D/logs/d22_final_F.log
[ $rc -ne 0 ] && { echo "final21 rc=$rc: restoring E" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit $rc; }
stage quality; zsh $D/tools/q21.sh $F F >> $O 2>&1; rc=$?; tail -30 $D/logs/q21_F.log >> $O
[ $rc -ne 0 ] && { echo "q21 rc=$rc: restoring E" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit $rc; }
stage soak; zsh $D/tools/soak21.sh $F 5 >> $O 2>&1; rc=$?; cat $D/logs/soak21.out >> $O
[ $rc -ne 0 ] && { echo "soak21 rc=$rc: restoring E" >> $O; zsh $D/tools/restore21.sh >> $O 2>&1; exit $rc; }
stage cpu; zsh $D/tools/cpu21.sh $F 150 1 2 >> $O 2>&1; cat $D/logs/cpu21.out >> $O
echo CHAINEDONE >> $O
