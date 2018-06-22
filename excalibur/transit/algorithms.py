# -- IMPORTS -- ------------------------------------------------------
import os
import pdb

import dawgie
import dawgie.context

import excalibur.transit as trn
import excaliburtransit.core as trncore
import excalibur.transit.states as trnstates

import excalibur.data as dat
import excalibur.data.algorithms as datalg
import excalibur.system as sys
import excalibur.system.algorithms as sysalg
import excalibur.target.edit as trgedit
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# VERBOSE AND DEBUG
verbose = False
debug = False
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if (len(t.replace(' ', '')) > 0)]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class normalization(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__type = 'transit'
        self.__cal = datalg.calibration()
        self.__tme = datalg.timing()
        self.__fin = sysalg.finalize()
        self.__out = [trnstates.NormSV(ext) for ext in fltrs]
        return
    
    def name(self):
        return 'normalization'

    def previous(self):
        return [dawgie.ALG_REF(dat.factory, self.__cal),
                dawgie.ALG_REF(dat.factory, self.__tme),
                dawgie.ALG_REF(sys.factory, self.__fin)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        fin = self.__fin.sv_as_dict()['parameters']
        vfin, sfin = trncore.checksv(fin)
        for ext in fltrs:
            update = False
            index = fltrs.index(ext)
            cal = self.__cal.sv_as_dict()[ext]
            vcal, scal = trncore.checksv(cal)
            tme = self.__tme.sv_as_dict()[ext]
            vtme, stme = trncore.checksv(tme)
            if vcal and vtme and vfin:
                update = self._norm(cal, tme, fin, self.__out[index])
                pass
            else:
                errstr = [m for m in [scal, stme, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if len(self.__out) > 0: ds.update()
        return
    
    def _norm(self, cal, tme, fin, out):
        normed = trncore.norm(cal, tme, fin, out, self.__type,
                              verbose=verbose, debug=debug)
        return normed
    
    def _failure(self, errstr):
        errmess = '--< TRANSIT NORMALIZATION: ' + errstr + ' >--'
        if verbose: print(errmess)
        return
    pass
# ---------------- ---------------------------------------------------
