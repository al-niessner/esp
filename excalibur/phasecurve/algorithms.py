'''phasecurve algorithms ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import logging; log = logging.getLogger(__name__)

import excalibur.phasecurve as phc
import excalibur.phasecurve.states as phcstates
import excalibur.phasecurve.core as phccore

# import excalibur.transit as trn
import excalibur.data.core as trncore
# import excalibur.transit.states as trnstatxes

import excalibur.data as dat
import excalibur.data.algorithms as datalg
import excalibur.runtime.algorithms as rtalg
import excalibur.runtime.binding as rtbind
import excalibur.system as sys
import excalibur.system.algorithms as sysalg

# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# VERBOSE AND DEBUG
verbose = False
debug = False
# FILTERS
fltrs = [str(fn) for fn in rtbind.filter_names.values()]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
# ---------------- ---------------------------------------------------
class pcnormalization(dawgie.Algorithm):
    '''
    G. ROUDIER: Light curve normalization by Out Of Transit data
    '''
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,3)
        self._type = 'phasecurve'
        self.__cal = datalg.calibration()
        self.__tme = datalg.timing()
        self.__rt = rtalg.autofill()
        self.__fin = sysalg.finalize()
        self.__out = [phcstates.NormSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'normalization'

    def previous(self):
        return [dawgie.ALG_REF(dat.task, self.__cal),
                dawgie.ALG_REF(dat.task, self.__tme),
                dawgie.ALG_REF(sys.task, self.__fin)] + \
                self.__rt.refs_for_proceed()

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        vfin, sfin = trncore.checksv(self.__fin.sv_as_dict()['parameters'])
        for ext in fltrs:
            update = False
            vcal, scal = trncore.checksv(self.__cal.sv_as_dict()[ext])
            vtme, stme = trncore.checksv(self.__tme.sv_as_dict()[ext])
            self.__rt.proceed(ext)
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
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for PHASECURVE.{self.name()}')
        return

    def _norm(self, cal, tme, fin, index):
        if 'Spitzer' in fltrs[index]:
            normed = phccore.norm_spitzer(cal, tme, fin, fltrs[index],
                                          self.__out[index], self._type)
        else:
            return True
        return normed

    def _failure(self, errstr):
        log.warning('--< %s NORMALIZATION: %s >--', self._type.upper(), errstr)
        return
    pass

class pcwhitelight(dawgie.Algorithm):
    '''
    G. ROUDIER: See inheritance and CI5 thread with A NIESSNER for __init__() method and class attributes https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/pull/86
    '''
    def __init__(self, nrm=pcnormalization()):
        self._version_ = dawgie.VERSION(1,1,2)
        self._type = 'phasecurve'
        self._nrm = nrm
        self.__rt = rtalg.autofill()
        self.__fin = sysalg.finalize()
        self.__out = [phcstates.WhiteLightSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'whitelight'

    def previous(self):
        return [dawgie.ALG_REF(phc.task, self._nrm),
                dawgie.ALG_REF(sys.task, self.__fin)] + \
                self.__rt.refs_for_proceed()

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
            self.__rt.proceed(ext)
            if vnrm and vfin:
                log.warning('--< %s WHITE LIGHT: %s >--', self._type.upper(), ext)
                update = self._whitelight(nrm, fin, self.__out[index], index)
                pass
            else:
                errstr = [m for m in [snrm, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if self.__out: ds.update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for PHASECURVE.{self.name()}')
        return

    def _whitelight(self, nrm, fin, out, index):
        if 'Spitzer' in fltrs[index]:
            wl = phccore.phasecurve_spitzer(nrm, fin, out, self._type, fltrs[index])
        else:
            return True
        return wl

    def _failure(self, errstr):
        log.warning('--< %s WHITE LIGHT: %s >--', self._type.upper(), errstr)
        return
    pass
