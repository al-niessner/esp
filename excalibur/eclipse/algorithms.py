'''eclipse algorithms dc'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import logging; log = logging.getLogger(__name__)

import excalibur.eclipse as ecl

import excalibur.runtime.binding as rtbind
import excalibur.system as sys
import excalibur.transit.algorithms as trnalg
import excalibur.transit.core as trncore

# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# FILTERS
fltrs = [str(fn) for fn in rtbind.filter_names.values()]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
# ECLIPSE CLASSES INHERIT FROM TRANSIT CLASSES
class normalization(trnalg.normalization):
    '''Normalize to out ot transit, inherits from transit.normalization'''
    def __init__(self):
        '''__init__ ds'''
        trnalg.normalization.__init__(self)
        self._version_ = trncore.normversion()
        self._type = 'eclipse'
        return
    pass

class whitelight(trnalg.whitelight):
    '''Create White Light Curves, inherits from transit.whitelight'''
    def __init__(self):
        '''__init__ ds'''
        trnalg.whitelight.__init__(self, nrm=normalization())
        self._version_ = trncore.wlversion()
        self._type = 'eclipse'
        return

    def previous(self):
        '''Input State Vectors: eclipse.normalization, system.finalize'''
        return [dawgie.ALG_REF(ecl.task, self._nrm),
                dawgie.ALG_REF(sys.task, self.__fin)]
    pass

class spectrum(trnalg.spectrum):
    '''Create emission spectrum, inherits from transit.spectrum'''
    def __init__(self):
        '''__init__ ds'''
        trnalg.spectrum.__init__(self, nrm=normalization(), wht=whitelight())
        self._version_ = trncore.spectrumversion()
        self._type = 'eclipse'
        return

    def previous(self):
        '''Input State Vectors: system.finalize, eclipse.normalization,
        eclipse.whitelight'''
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(ecl.task, self._nrm),
                dawgie.ALG_REF(ecl.task, self._wht)]
    pass
# ---------------- ---------------------------------------------------
