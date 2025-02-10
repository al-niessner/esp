#! /usr/bin/env bash

rdir=$(realpath $(dirname $0)/..)
baseVersion=$(python <<EOF
try:
    import pyblake2 as hashlib
except:
    import hashlib

with open ('$rdir/.docker/Dockerfile.base', 'br') as f: data = f.read()
with open ('$rdir/requirements.txt', 'br') as f: data = f.read()
k = hashlib.blake2b (data, digest_size=8)
print (k.hexdigest())
EOF
           )
python <<EOF
import os,sys

with open ('$rdir/.docker/.env', 'rt') as f: config = f.read()
k = 'ESP_VERSION=#{ESP_VERSION:-'.replace ('#','$')
i = config.find (k)
if i > -1:
    v = config[i+len(k):config.find('}',i+len(k))]
    if 'KEEP_CHANGES' in os.environ:
        config = config.replace (v, '$baseVersion')
        v = '$baseVersion'
        with open ('$rdir/.docker/.env', 'tw') as f: f.write(config)
    if v == '$baseVersion': sys.exit(0)
    else: sys.exit(-2)
else: sys.exit(-1)
EOF
