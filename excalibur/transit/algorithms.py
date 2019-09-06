# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import logging; log = logging.getLogger(__name__)

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
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]

# Only G141 makes it through
# fltrs = [f for f in fltrs if ('WFC3' in f) or ('G750' in f)]

# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class normalization(dawgie.Algorithm):
    '''
G. ROUDIER: Light curve normalization by Out Of Transit data
    '''
    def __init__(self):
        self._version_ = trncore.normversion()
        self._type = 'transit'
        self.__cal = datalg.calibration()
        self.__tme = datalg.timing()
        self.__fin = sysalg.finalize()
        self.__out = [trnstates.NormSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'normalization'

    def previous(self):
        return [dawgie.ALG_REF(dat.task, self.__cal),
                dawgie.ALG_REF(dat.task, self.__tme),
                dawgie.ALG_REF(sys.task, self.__fin)]

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
                log.warning('--< %s NORMALIZATION: %s >--', self._type.upper(), ext)
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
        if self.__out: ds.update()
        return

    def _norm(self, cal, tme, fin, index):
        normed = trncore.norm(cal, tme, fin, fltrs[index], self.__out[index], self._type,
                              verbose=False)
        return normed

    def _failure(self, errstr):
        log.warning('--< %s NORMALIZATION: %s >--', self._type.upper(), errstr)
        return
    pass

class whitelight(dawgie.Algorithm):
    '''
G. ROUDIER: See inheritance and CI5 thread with A NIESSNER for __init__() method and class attributes https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/pull/86
    '''
    def __init__(self, nrm=normalization()):
        self._version_ = trncore.wlversion()
        self._type = 'transit'
        self._nrm = nrm
        self.__fin = sysalg.finalize()
        self.__out = [trnstates.WhiteLightSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'whitelight'

    def previous(self):
        return [dawgie.ALG_REF(trn.task, self._nrm),
                dawgie.ALG_REF(sys.task, self.__fin)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        fin = self.__fin.sv_as_dict()['parameters']
        vfin, sfin = trncore.checksv(fin)
        for ext in fltrs:
            update = False
            index = fltrs.index(ext)
            nrm = self._nrm.sv_as_dict()[ext]
            vnrm, snrm = trncore.checksv(nrm)
            if vnrm and vfin:
                log.warning('--< %s WHITE LIGHT: %s >--', self._type.upper(), ext)
                update = self._whitelight(nrm, fin, self.__out[index], ext)
                pass
            else:
                errstr = [m for m in [snrm, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if self.__out: ds.update()
        return

    def _whitelight(self, nrm, fin, out, ext):
        wl = trncore.whitelight(nrm, fin, out, ext, self._type,
                                chainlen=int(1e4), verbose=False)
        return wl

    def _failure(self, errstr):
        log.warning('--< %s WHITE LIGHT: %s >--', self._type.upper(), errstr)
        return
    pass

class spectrum(dawgie.Algorithm):
    '''
G. ROUDIER: See inheritance and CI5 thread with A NIESSNER for __init__() method and class attributes https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/pull/86
    '''
    def __init__(self, nrm=normalization(), wht=whitelight()):
        self._version_ = trncore.spectrumversion()
        self._type = 'transit'
        self.__fin = sysalg.finalize()
        self._nrm = nrm
        self._wht = wht
        self.__out = [trnstates.SpectrumSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'spectrum'

    def previous(self):
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(trn.task, self._nrm),
                dawgie.ALG_REF(trn.task, self._wht)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        vfin, sfin = trncore.checksv(self.__fin.sv_as_dict()['parameters'])
        for index, ext in enumerate(fltrs):
            update = False
            vnrm, snrm = trncore.checksv(self._nrm.sv_as_dict()[ext])
            vwht, swht = trncore.checksv(self._wht.sv_as_dict()[ext])
            if vfin and vnrm and vwht:
                log.warning('--< %s SPECTRUM: %s >--', self._type.upper(), ext)
                update = self._spectrum(self.__fin.sv_as_dict()['parameters'],
                                        self._nrm.sv_as_dict()[ext],
                                        self._wht.sv_as_dict()[ext],
                                        self.__out[index], ext)
                pass
            else:
                errstr = [m for m in [sfin, snrm, swht] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if self.__out: ds.update()
        return

    def _spectrum(self, fin, nrm, wht, out, ext):
        s = trncore.spectrum(fin, nrm, wht, out, ext, self._type,
                             chainlen=int(1e4), verbose=False)
        return s

    def _failure(self, errstr):
        log.warning('--< %s SPECTRUM: %s >--', self._type.upper(), errstr)
        return
    pass
# ---------------- ---------------------------------------------------
