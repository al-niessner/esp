#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Static analysis of excalibur"
context="continuous-integration/02/esp-pylint"

post_state "$context" "$description" "$state"

if current_state
then
    docker run --rm -v $PWD:$PWD -u $UID -w $PWD esp_cit:$(cit_version) pylint \
               --rcfile=$PWD/.ci/pylint.rc $PWD/excalibur | tee pylint.rpt.txt
    python3 <<EOF
mn = '<unknown>'
count = 0
with open ('pylint.rpt.txt', 'rt') as f:
    for l in f.readlines():
        if l.startswith ('***'): mn = l.split()[2]
        if len (l) < 2: continue
        if l[0] not in 'CEFIRW' or l[1] != ':': continue
        if 0 < l.find ('(missing-docstring)'): continue
        if 0 < l.find ('(locally-disabled)'): continue
        count += 1
        print (count, mn, l.strip())
        pass
    pass
if 0 < count:
    print ('pylint check failed', count)
    with open ('.ci/status.txt', 'tw') as f: f.write ('failure')
else: print ('pylint check success')
EOF
    state=`get_state`
fi

post_state "$context" "$description" "$state"
git checkout .ci/status.txt
