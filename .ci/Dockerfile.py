FROM dawgie:ghrVersion
RUN set -ex && \
    apt-get update && \
    apt-get install -y libblas3 libblas-dev python-liblas && \
    apt-get clean && apt-get autoremove && \
    /usr/bin/pip3 install astropy ldtk lmfit matplotlib scipy pymc && \
   rm -rf ${HOME}/.cache
