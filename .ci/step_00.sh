#! /usr/bin/env bash

. .ci/util.sh

# https://developer.github.com/v3/repos/statuses/

state="pending" # "success" "pending" "failure" "error"
description="Build an environment for CI"
context="continuous-integration/00/environment"

post_state "$context" "$description" "$state"

if current_state
then
    lv="$(layer_versions)"
    pyVersion="${lv:0:16}"
    citVersion="${lv:17:16}"
    echo "Python layer $pyVersion"
    echo "CIT Version: $citVersion"
    .ci/dcp.py --server .ci/Dockerfile.1 .ci/Dockerfile.2 &
    while [ ! -f .ci/Dockerfile.2.dcp ]
    do
        sleep 3
    done

    if [ -n "${USER_APIKEY}" ]
    then
        docker login -p ${USER_APIKEY} -u ${USER_NAME} cae-artifactory.jpl.nasa.gov:16001
        docker login -p ${USER_APIKEY} -u ${USER_NAME} cae-artifactory.jpl.nasa.gov:16002
        docker login -p ${USER_APIKEY} -u ${USER_NAME} cae-artifactory.jpl.nasa.gov:16003
        for destination in `destinations`
        do
            source="`lookup_source $destination`"
            version="`lookup_version $destination`"
            echo "V: ${version} D: ${destination} S:${source}"
            docker pull cae-artifactory.jpl.nasa.gov:$(which_port ${version})${source}:${version}
            docker tag cae-artifactory.jpl.nasa.gov:$(which_port ${version})${source}:${version} ${destination}:${version}
        done
        docker logout cae-artifactory.jpl.nasa.gov:16001
        docker logout cae-artifactory.jpl.nasa.gov:16002
        docker logout cae-artifactory.jpl.nasa.gov:16003
    fi

    if [ -z "$(docker images | awk -e '{print $1":"$2}' | grep py:$pyVersion)" ]
    then
        echo "   Building python layer $pyVersion"
        docker build --network=host -t esp_py:${pyVersion} - < .ci/Dockerfile.1.dcp
    fi

    if [ -z "$(docker images | awk -e '{print $1":"$2}' | grep cit:$citVersion)" ]
    then
        echo "   Building CI Tools layer $citVersion"
        docker build --network=host -t esp_cit:${citVersion} - < .ci/Dockerfile.2.dcp
    fi

    rm .ci/Dockerfile.1 .ci/Dockerfile.2
    rm .ci/Dockerfile.1.dcp .ci/Dockerfile.2.dcp
    state=`get_state`
fi

post_state "$context" "$description" "$state"
