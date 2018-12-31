#! /usr/bin/env bash

while [ -f /tmp/crew.ops ]
do
    flock /tmp/crew.ops $(dirname $0)/spawn.sh
done
