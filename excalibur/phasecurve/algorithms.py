# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import excalibur.phasecurve.states as phcstates

import excalibur.data as dat
import excalibur.data.algorithms as datalg
import excalibur.target.edit as trgedit
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# VERBOSE AND DEBUG
verbose = False
debug = False
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if len(t.replace(' ', '')) > 0]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class normalization(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__cal = datalg.calibration()
        self.__tme = datalg.timing()
        self.__out = [phcstates.NormSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'normalization'

    def previous(self):
        return [dawgie.ALG_REF(dat.factory, self.__cal),
                dawgie.ALG_REF(dat.factory, self.__tme)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        return
    pass
# ---------------- ---------------------------------------------------
