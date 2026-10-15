#!/bin/zsh
# run-d7.sh: run-strata-x8.sh with the engine binary taken from $D7_BIN (default the live xport8 binary d78b2237); the Decode7 test configs set D7_BIN in their env.  Shim = ~/shim (current).
source ~/.chestnut_env
SITE=$(~/tinygrad/.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
export DEV=NV:PTX PYTHONPATH=$HOME/shim:$HOME/tg-strata:$SITE PYTHONNOUSERSITE=1
[[ "$*" == *exl3* ]] && export STRATA_POOL_SPIN_US=${STRATA_POOL_SPIN_US:-0}
[[ "$*" == *exl3* ]] && export STRATA_EXL3_SLIM=${STRATA_EXL3_SLIM:-1}
[[ "$*" == *exl3* ]] && export STRATA_HS_PRE=${STRATA_HS_PRE:-1} STRATA_HS_DEFER=${STRATA_HS_DEFER:-1} STRATA_MAC_PULL=${STRATA_MAC_PULL:-1}
exec ${D7_BIN:-$HOME/strata-xport8/bin/strata} "$@"
