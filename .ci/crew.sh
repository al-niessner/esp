#! /usr/bin/env bash

if [ -f /tmp/crew.ops ]
then
    flock --unlock /tmp/crew.ops $(dirname $0)/spawn.sh
fi
