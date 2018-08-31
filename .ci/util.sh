
basedir=$(cd "$(dirname "$0")"/..; pwd)
GHE4JPL_API_URL=https://github-fn.jpl.nasa.gov/api/v3
GHE4JPL_TOKEN=fbf04b881c8b6a73b37d99f3e33bcab6a1e65d3e
ghrVersion=${ghrVersion:-"`git describe --tags`"}
PATH=/usr/local/python3/bin:${PATH}
REPO=EXCALIBUR/esp

export GHE4JPL_API_URL PATH 

cit_version ()
{
    lv="$(layer_versions)"
    rm .ci/Dockerfile.1 .ci/Dockerfile.2
    echo "${lv:17:16}"
}

current_state ()
{
    test `cat ${basedir}/.ci/status.txt` == "success"
}

destinations ()
{
    grep -v '#' ${basedir}/app.dependency.manifest | awk -e '{print $1}'
}

download ()
{
    curl -L \
         -H "Authorization: token ${GHE4JPL_TOKEN}" \
         ${ghrReleaseTarball} > $1
}

get_state ()
{
    cat ${basedir}/.ci/status.txt
}

layer_versions ()
{
    for destination in `destinations`
    do
        [[ "$destination" == "dawgie" ]] && dawgieVersion="$(lookup_version $destination)"
    done
    python3 <<EOF
with open ('.ci/Dockerfile.py', 'rt') as f: text = f.read()
with open ('.ci/Dockerfile.1', 'tw') as f: f.write (text.replace ("ghrVersion", "${dawgieVersion}"))
EOF
    pyVersion=$(python3 <<EOF
try:
    import pyblake2 as hashlib
except:
    import hashlib

with open ('.ci/Dockerfile.1', 'br') as f: data = f.read()
k = hashlib.blake2b (data, digest_size=8)
print (k.hexdigest())
EOF
           )
    python3 <<EOF
with open ('.ci/Dockerfile.cit', 'rt') as f: text = f.read()
with open ('.ci/Dockerfile.2', 'tw') as f: f.write (text.replace ("ghrVersion", "${pyVersion}"))
EOF
    citVersion=$(python3 <<EOF
try:
    import pyblake2 as hashlib
except:
    import hashlib

with open ('.ci/Dockerfile.2', 'br') as f: data = f.read()
k = hashlib.blake2b (data, digest_size=8)
print (k.hexdigest())
EOF
           )
    echo $pyVersion $citVersion
}

lookup ()
{
    grep -v '#' ${basedir}/app.dependency.manifest | grep ${1}
}

lookup_source ()
{
    lookup $1 | awk -e '{print $2}'
}

lookup_version ()
{
    lookup $1 | awk -e '{print $3}'
}

post_state ()
{
    if current_state && [ "$3" == "pending" ]
    then
        echo "$1 -- $2"
    else
        echo "$1 -- completion state: $3"
    fi

    curl -XPOST \
         -H "Authorization: token ${GHE4JPL_TOKEN}" \
         ${GHE4JPL_API_URL}/repos/${REPO}/statuses/${ghprbActualCommit} \
         -d "{\"state\": \"${3}\", \"target_url\": \"${BUILD_URL}/console\", \"description\": \"${2}\", \"context\": \"${1}\"}" > /dev/null 2>&1
}

which_port ()
{
    python3 <<EOF
v = "${1:-${ghrVersion}}".split ('.')
if len (v) == 3:
    if v[0].isdigit() and v[1].isdigit() and v[2].isdigit(): port = 16003  #  release port
    elif v[0].isdigit() and v[1].isdigit() and 0 < v[2].find ('-rc') and v[2].split('-')[0].isdigit(): port = 16002  # staging port
    else: port = 16001  #  devel port
else: port = 16001  # devel port

print (port)
EOF
}
