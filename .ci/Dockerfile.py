FROM dawgie:ghrVersion
RUN set -ex && \
    apt-get install python3.6-gdbm && \
    /usr/bin/pip3 install astropy==3.0.4 \
                          ldtk==1.0 \
                          lmfit==0.9.11 \
                          matplotlib==2.2.3 \
                          pymc3==3.6 \
                          scipy==1.1.0 && \
   rm -rf ${HOME}/.cache && \
   mkdir /.theano && chmod 777 /.theano && \
   mkdir -p /proj/sdp/data/ldtk && chmod 777 /proj/sdp/data/ldtk
