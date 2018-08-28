FROM dawgie:ghrVersion
RUN set -ex && \
    /usr/bin/pip3 install astropy==3.0.4 \
                          ldtk==1.0 \
                          lmfit==0.9.11 \
                          matplotlib==2.2.3 \
                          pymc3==3.5 \
                          scipy==1.1.0 && \
   rm -rf ${HOME}/.cache
