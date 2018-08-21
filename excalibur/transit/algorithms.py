# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import excalibur.transit as trn
import excalibur.transit.core as trncore
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
fltrs = [t.strip() for t in fltrs if len(t.replace(' ', '')) > 0]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class normalization(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'transit'
        self.__verbose = verbose
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
        vfin, sfin = trncore.checksv(self.__fin.sv_as_dict()['parameters'])
        for ext in fltrs:
            update = False
            vcal, scal = trncore.checksv(self.__cal.sv_as_dict()[ext])
            vtme, stme = trncore.checksv(self.__tme.sv_as_dict()[ext])
            if vcal and vtme and vfin:
                update = self._norm(self.__cal.sv_as_dict()[ext],
                                    self.__tme.sv_as_dict()[ext],
                                    self.__fin.sv_as_dict()['parameters'],
                                    fltrs.index(ext))
                pass
            else:
                errstr = [m for m in [scal, stme, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[fltrs.index(ext)])
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        return

    def _norm(self, cal, tme, fin, index):
        normed = trncore.norm(cal, tme, fin, fltrs[index],
                              self.__out[index], self._type,
                              verbose=self.__verbose, debug=debug)
        return normed

    def _failure(self, errstr):
        errmess = '--< TRANSIT NORMALIZATION: ' + errstr + ' >--'
        if self.__verbose: print(errmess)
        return
    pass

class whitelight(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'transit'
        self.__verbose = verbose
        self.__nrm = normalization()
        self.__fin = sysalg.finalize()
        self.__out = [trnstates.WhiteLightSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'whitelight'

    def previous(self):
        return [dawgie.ALG_REF(trn.factory, self.__nrm),
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
            nrm = self.__nrm.sv_as_dict()[ext]
            vnrm, snrm = trncore.checksv(nrm)
            if vnrm and vfin:
                update = self._whitelight(nrm, fin, self.__out[index])
                pass
            else:
                errstr = [m for m in [snrm, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        return

    def _whitelight(self, nrm, fin, out):
        wl = trncore.whitelight(nrm, fin, out, self._type,
                                verbose=self.__verbose, debug=debug)
        return wl

    def _failure(self, errstr):
        errmess = '--< TRANSIT WHITE LIGHT: ' + errstr + ' >--'
        if self.__verbose: print(errmess)
        return
    pass

class spectrum(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self._type = 'transit'
        self.__verbose = verbose
        self.__fin = sysalg.finalize()
        self.__nrm = normalization()
        self.__wht = whitelight()
        self.__out = [trnstates.SpectrumSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'spectrum'

    def previous(self):
        return [dawgie.ALG_REF(sys.factory, self.__fin),
                dawgie.ALG_REF(trn.factory, self.__nrm),
                dawgie.ALG_REF(trn.factory, self.__wht)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        vfin, sfin = trncore.checksv(self.__fin.sv_as_dict()['parameters'])
        for index, ext in enumerate(fltrs):
            update = False
            vnrm, snrm = trncore.checksv(self.__nrm.sv_as_dict()[ext])
            vwht, swht = trncore.checksv(self.__wht.sv_as_dict()[ext])
            if vfin and vnrm and vwht:
                update = self._spectrum(self.__fin.sv_as_dict()['parameters'],
                                        self.__nrm.sv_as_dict()[ext],
                                        self.__wht.sv_as_dict()[ext],
                                        self.__out[index])
                pass
            else:
                errstr = [m for m in [sfin, snrm, swht] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        return

    def _spectrum(self, fin, nrm, wht, out):
        s = trncore.spectrum(fin, nrm, wht, out, self._type,
                             verbose=self.__verbose, debug=debug)
        return s

    def _failure(self, errstr):
        errmess = '--< TRANSIT SPECTRUM: ' + errstr + ' >--'
        if self.__verbose: print(errmess)
        return
    pass
# ---------------- ---------------------------------------------------
