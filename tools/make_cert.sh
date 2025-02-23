#! /usr/bin/env bash

if [[ $# -ne 2 ]]
then
    echo "usage: make_cert.sh <filename>"
fi

# make CSR
openssl req -newkey rsa:2048 -nodes -keyout device.key \
        -subj "/C=US/ST=CA/L=LA/O=None/CN=excalibur.jpl.nasa.gov" -out device.csr
# write the v3.ext file
echo "authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
subjectAltName = @alt_names

[alt_names]
DNS.1 = excalibur.jpl.nasa.gov
DNS.2 = mentor0.jpl.nasa.gov
DNS.3 = mentor3.jpl.nasa.gov
DNS.4 = mentor4.jpl.nasa.gov
DNS.5 = mentor5.jpl.nasa.gov
DNS.6 = mentor6.jpl.nasa.gov
DNS.7 = mentor7.jpl.nasa.gov
DNS.8 = mentor8.jpl.nasa.gov
DNS.9 = mentor9.jpl.nasa.gov
" > v3.ext

# build the certificate
openssl x509 -req -in device.csr -signkey device.key -out device.crt \
        -sha256 -extfile v3.ext -days 36500 
# build the complete pem and just public bit for being a guest
cat device.key device.crt > $1
chmod 600 $1
mv device.crt $1.public
rm device.csr device.key v3.ext

if [ -d /proj/sdp/data/certs ]
then
    cp $1.public /proj/sdp/data/certs/dawgie.public.pem.${USER}
    chmod 664 /proj/sdp/data/certs/dawgie.public.pem.${USER}
else
    echo "You need to manually copy ${1}.public to /proj/sdp/data/certs/dawgie.public.pem.${USER}"
fi
