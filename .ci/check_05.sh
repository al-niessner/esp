#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Compliancy check to ensure DAWGIE is satisfied"
context="continuous-integration/05/esp-dawgie-compliance"

post_state "$context" "$description" "$state"

if current_state
then
    docker run --rm -v $PWD:$PWD -u $UID -w $PWD -e USERNAME=$USERNAME esp_cit:$(cit_version) python3 -m dawgie.tools.compliant -v --ae-dir $PWD/excalibur --ae-pkg excalibur | tee $PWD/compliant.rpt.txt
    
    errs=`grep False $PWD/compliant.rpt.txt | wc -l`
    [[ $errs -ne 0 ]] && echo -n "failure" > $PWD/.ci/status.txt
    state=`get_state`
fi

post_state "$context" "$description" "$state"
