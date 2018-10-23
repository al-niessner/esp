# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import logging; log = logging.getLogger(__name__)

import excalibur.eclipse as ecl

import excalibur.system as sys
import excalibur.transit.algorithms as trnalg
import excalibur.target.edit as trgedit
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '').__len__() > 0]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
# ECLIPSE CLASSES INHERIT FROM TRANSIT CLASSES
class normalization(trnalg.normalization):
    def __init__(self):
        trnalg.normalization.__init__(self)
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'eclipse'
        return

    @staticmethod
    def _failure(errstr):
        log.log(31, '--< ECLIPSE NORMALIZATION: '+errstr+' >--')
        return
    pass

class whitelight(trnalg.whitelight):
    def __init__(self):
        trnalg.whitelight.__init__(self)
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'eclipse'
        return

    def previous(self):
        return [dawgie.ALG_REF(ecl.factory, self.__nrm),
                dawgie.ALG_REF(sys.factory, self.__fin)]

    @staticmethod
    def _failure(errstr):
        log.warning('--< ECLIPSE WHITE LIGHT: ' + errstr + ' >--')
        return
    pass

class spectrum(trnalg.spectrum):
    def __init__(self):
        trnalg.spectrum.__init__(self)
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'eclipse'
        return

    def previous(self):
        return [dawgie.ALG_REF(sys.factory, self.__fin),
                dawgie.ALG_REF(ecl.factory, self.__nrm),
                dawgie.ALG_REF(ecl.factory, self.__wht)]

    @staticmethod
    def _failure(errstr):
        log.warning('--< ECLIPSE SPECTRUM: ' + errstr + ' >--')
        return
    pass
# ---------------- ---------------------------------------------------
