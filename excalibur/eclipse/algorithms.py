# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import logging; log = logging.getLogger(__name__)

import excalibur.data.algorithms as datalg
import excalibur.system.algorithms as sysalg
import excalibur.transit.algorithms as trnalg
import excalibur.transit.states as trnstates
import excalibur.target.edit as trgedit
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# VERBOSE AND DEBUG
verbose = False
debug = False
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '').__len__() > 0]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class normalization(trnalg.normalization):
    def __init__(self):
        trnalg.normalization.__init__(self)
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'eclipse'
        self.__verbose = verbose
        self.__cal = datalg.calibration()
        self.__tme = datalg.timing()
        self.__fin = sysalg.finalize()
        self.__out = [trnstates.NormSV(ext) for ext in fltrs]
        return
    pass
# ---------------- ---------------------------------------------------
