#! /usr/bin/env bash

if [[ $# -ne 2 ]]
then
    echo "usage: execute_on_pp.sh <task name> <target name>"
    echo "   <task name> the python package name after excalbur as in cerberus"
    echo "               given the full package nameexcalibur.cerberus"
    echo ""
    echo "   <target name> is the exactly that"
    echo ""
    echo "   Both <task name> and <target name> are required."
    echo ""
    echo "   Environment variable RUNID can be used to set the runid for this"
    echo "   job. It too is optional and defaults to 17 if not set."
    echo ""
    exit -1
fi

docker exec \
       -e DISPLAY=${DISPLAY} \
       -e FE_PORT=${DAWGIE_FE_PORT:-9990} \
       -e RUNID=${RUNID:-17} \
       -e TARGET_NAME="${2}" \
       -e USER=${USER} \
       -e USERNAME=${USERNAME} \
       -it \
       -u ${UID}:1512 \
       ${USER}_privatepl python3 -m excalibur.${1}
