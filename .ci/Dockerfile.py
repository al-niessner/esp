FROM ubuntu:18.04
RUN set -ex && \
    chmod 755 /bin/dcp.py && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y \
            git \
            graphviz \
            haveged \
            python3 \
            python3-dev \
            python3-gdbm \
            python3-numpy \
            python3-pip \
            python3-psycopg2 \
            python3-setuptools && \
    apt-get clean && apt-get autoremove && \
    ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    /usr/bin/pip3 install astropy==3.0.4 \
                          dawgie==1,2.3 \
                          ldtk==1.0 \
                          lmfit==0.9.11 \
                          matplotlib==2.2.3 \
                          pymc3==3.6 \
                          scipy==1.1.0 && \
   rm -rf ${HOME}/.cache && \
   mkdir /.theano && chmod 777 /.theano && \
   mkdir -p /proj/data/ldtk && chmod 777 /proj/data/ldtk
