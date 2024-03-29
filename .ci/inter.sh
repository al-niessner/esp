#! /usr/bin/env bash

if [[ $# -ne 1 ]]
then
    echo "usage: $(basename $0) <release name>"
fi

[ ! -d ${1} ] && mkdir -p ${1}
di=$(docker images | grep esp_server | head -n 1 | gawk -e '{print $1":"$2}')
echo "docker image: $di"
cd $(realpath $(dirname $0)/..)
aerev=$(git rev-parse HEAD)
.ci/dcp.py --server .ci/inter.txt &
while [ ! -f .ci/inter.txt.dcp ]
do
    sleep 3
done
pn=$(cat .ci/inter.txt.dcp)
docker build --network host --tag mausoleum:latest - <<EOF
FROM ${di}
ENV DAWGIE_DOCKERIZED_AE_GIT_REVISION ${aerev}
RUN set -ex && \
    mkdir -p /proj/src/ae/ && \
    /bin/dcp.py --port ${pn} -r excalibur /proj/src/ae
EOF
docker save -o ${1}/server.docker.image mausoleum:latest
docker rmi mausoleum:latest
rm .ci/inter.txt.dcp
kill %1
echo "work on data"
/home/niessner/Projects/DAWGIE/Python/dawgie/db/tools/inter.py \
    -B /proj/sdp/data/dbs \
    -O ${1} \
    postgres -b /proj/sdp/data/db/ops.00.bck < /proj/sdp/data/nexsci.sv.txt
cp .ci/inter.run.sh ${1}/run.sh
cp .ci/inter.README.txt ${1}/README.txt
cp -r /proj/sdp/ops/front-end/* ${1}/fe
echo "building tarfile ${1}.tgz"
cd $(realpath ${1}/..)
tar -czf ${1}.tgz $(basename ${1})
