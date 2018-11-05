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
    pass

class whitelight(trnalg.whitelight):
    def __init__(self):
        trnalg.whitelight.__init__(self, nrm=normalization())
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'eclipse'
        return

    def previous(self):
        return [dawgie.ALG_REF(ecl.factory, self._nrm),
                dawgie.ALG_REF(sys.factory, self.__fin)]
    pass

class spectrum(trnalg.spectrum):
    def __init__(self):
        trnalg.spectrum.__init__(self, nrm=normalization(), wht=whitelight())
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'eclipse'
        return

    def previous(self):
        return [dawgie.ALG_REF(sys.factory, self.__fin),
                dawgie.ALG_REF(ecl.factory, self._nrm),
                dawgie.ALG_REF(ecl.factory, self._wht)]
    pass
# ---------------- ---------------------------------------------------
