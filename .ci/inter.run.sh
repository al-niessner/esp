#! /usr/bin/env bash

if [ -z "$(docker ps -a | grep masoleum)" ]
then
    docker run --detach --rm \
	   --env USER=$USER \
	   --env USERNAME=$USERNAME \
	   --name mausoleum \
	   --network masops \
           --publish 8080:8080 \
	   --user $UID \
	   -v $(realpath $(dirname $0)):/proj/data \
           -v $(realpath $(dirname $0))/fe:/proj/src/front-end \
	   -v ${PG_PASS}:/proj/.pgpass:ro \
	   -v /etc/group:/etc/group:ro \
	   -v /etc/passwd:/etc/passwd:ro \
	   mausoleum:latest --context-db-host=<postgres hostname> \
                            --context-db-path=<pg user>:<pg pwd> \
                            --context-db-port=<postgres port>
else
    echo "Pipeline is already running"
fi
