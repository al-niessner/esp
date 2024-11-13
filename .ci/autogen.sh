#! /usr/bin/env bash

repodir=$(realpath $(dirname $0)/..)
. ${repodir}/.ci/util.sh
docker run --rm -v ${repodir}:${repodir} -u ${UID} -w ${repodir} esp_cit:$(cit_version) pyxbgen --schema-location=excalibur/runtime/levers.xsd --module=binding --module-prefix=excalibur.runtime
cd ${repodir}/excalibur/runtime
md5sum levers.xsd binding.py > autogen.md5
