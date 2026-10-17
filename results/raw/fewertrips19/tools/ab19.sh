#!/bin/zsh
# ab19.sh: C D H H D C  (H = hoistc: strata_d19b + STRATA_D19_HOIST=1), each arm = restart + 29 realistic prompts + 12 probes (+ image on H arms). Log ~/strata-fewertrips19/logs/ab19.out; restores live D at the end.
D=~/strata-fewertrips19; O=$D/logs/ab19.out; : > $O
for pair in "cA configC" "dA configD" "hA hoistc" "hB hoistc" "dB configD" "cB configC"; do
  set -- ${=pair}; tag=$1; cfg=$D/cfg/$2.json
  [ -e $D/USER_ACTIVE ] && { echo "USER_ACTIVE stop" >> $O; exit 3; }
  echo "### $tag $cfg $(date +%T)" >> $O
  [[ $tag == h* ]] && export ARM8_IMAGE=1 || unset ARM8_IMAGE
  zsh $D/tools/arm19.sh $cfg $tag ~/strata-decode7/seq_on.txt >> $O 2>&1
  tail -3 $D/logs/$tag.out >> $O
done
[ -e $D/USER_ACTIVE ] || { zsh ~/strata-decode15/tools/restart15.sh ~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json > $D/logs/restore_ab.log 2>&1; echo "restored live md5 $(md5 -q ~/shim/strata-exl3-full512-final-g4-lru-ht-prefix-p600.json)" >> $O; }
echo AB19DONE >> $O
