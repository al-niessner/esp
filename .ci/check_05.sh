#! /usr/bin/env bash

. .ci/util.sh

state="pending" # "success" "pending" "failure" "error"
description="Compliancy check to ensure DAWGIE is satisfied"
context="continuous-integration/05/esp-dawgie-compliance"

post_state "$context" "$description" "$state"

if current_state
then
    python3 -m dawgie.tools.compliant --ae-dir $PWD/excalibur --ae-pkg excalibur
    state=`get_state`
fi

post_state "$context" "$description" "$state"
