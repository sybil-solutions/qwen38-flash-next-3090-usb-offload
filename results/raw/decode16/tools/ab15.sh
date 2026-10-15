#!/bin/zsh
# ab15.sh <newcfg> (ON the mini, card owner): ABBA = ctlA newA newB ctlB (ctl = the Decode14 final config ~/strata-decode15/cfg/live_d14.json), each arm = restart + 10 realistic prompts + 12 probes (+ image on the new arms).  Log ~/strata-decode15/logs/ab15.out
D=~/strata-decode15; O=$D/logs/ab15.out; : > $O; new=$1
for pair in "ctlA $D/cfg/live_d14.json" "newA $new" "newB $new" "ctlB $D/cfg/live_d14.json"; do
  set -- ${=pair}; tag=$1; cfg=$2
  echo "### $tag $cfg $(date +%T)" >> $O
  [[ $tag == new* ]] && export ARM8_IMAGE=1 || unset ARM8_IMAGE
  zsh $D/tools/arm15.sh $cfg $tag ~/strata-decode7/seq_on.txt >> $O 2>&1
  tail -12 $D/logs/$tag.out >> $O
done
echo AB15DONE >> $O
