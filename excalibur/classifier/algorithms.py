'''classifier algoithms dc'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context

import excalibur.transit as trn
import excalibur.transit.core as trncore
import excalibur.transit.algorithms as trnalg

import excalibur.eclipse as ecl
import excalibur.eclipse.algorithms as eclalg

import excalibur.target.edit as trgedit

import excalibur.system as sys
import excalibur.system.algorithms as sysalg
# import excalibur.system.core as syscore

import excalibur.classifier.core as clscore
import excalibur.classifier.states as clsstates
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
exc_fltrs = ['Spitzer','JWST']
fltrs = [f for f in fltrs if not any(ins in f for ins in exc_fltrs)]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class inference(dawgie.Algorithm):
    '''G. ROUDIER: Data collection by filters'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = clscore.predversion()
        self.__whitelight = trnalg.whitelight()
        self.__spectrum = trnalg.spectrum()
        self.__eclwhitelight = eclalg.whitelight()
        self.__eclspectrum = eclalg.spectrum()
        self.__finalize = sysalg.finalize()
        self.__out = [clsstates.PredictSV('transit-'+ext) for ext in fltrs]
        self.__out.extend([clsstates.PredictSV('eclipse-'+ext) for ext in fltrs])
        return

    def name(self):
        '''name ds'''
        return 'inference'

    def previous(self):
        '''previous ds'''
        return [dawgie.ALG_REF(trn.task, self.__whitelight),
                dawgie.ALG_REF(trn.task, self.__spectrum),
                dawgie.ALG_REF(ecl.task, self.__eclwhitelight),
                dawgie.ALG_REF(ecl.task, self.__eclspectrum),
                dawgie.ALG_REF(sys.task, self.__finalize)]

    def state_vectors(self):
        '''state_vectors ds'''
        return self.__out

    def run(self, ds, ps):
        '''run ds'''
        svupdate = []
        vfin, sfin = trncore.checksv(self.__finalize.sv_as_dict()['parameters'])
        for ext in fltrs:
            update = False
            vwl, swl = trncore.checksv(self.__whitelight.sv_as_dict()[ext])
            vsp, ssp = trncore.checksv(self.__spectrum.sv_as_dict()[ext])
            e_vwl, e_swl = trncore.checksv(self.__eclwhitelight.sv_as_dict()[ext])
            e_vsp, e_ssp = trncore.checksv(self.__eclspectrum.sv_as_dict()[ext])
            if vwl and vsp and vfin:
                log.warning('--< TRANSIT CLASSIFICATION: %s >--', ext)
                update = self._predict(self.__whitelight.sv_as_dict()[ext],
                                       self.__spectrum.sv_as_dict()[ext],
                                       self.__finalize.sv_as_dict()['parameters'],
                                       fltrs.index(ext))
                if update: svupdate.append(self.__out[fltrs.index(ext)])
                pass
            else:
                errstr = [m for m in [swl, ssp, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if e_vwl and e_vsp and vfin:
                log.warning('--< ECLIPSE CLASSIFICATION: %s >--', ext)
                update = self._predict(self.__eclwhitelight.sv_as_dict()[ext],
                                       self.__eclspectrum.sv_as_dict()[ext],
                                       self.__finalize.sv_as_dict()['parameters'],
                                       len(fltrs)+fltrs.index(ext))
                if update: svupdate.append(self.__out[len(fltrs)+fltrs.index(ext)])
                pass
            else:
                errstr = [m for m in [e_swl, e_ssp, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            pass
        self.__out = svupdate
        if self.__out: ds.update()
        return

    def _predict(self, wl, sp, fin, index):
        '''_predict ds'''
        status = clscore.predict(wl, sp, fin['priors'], self.__out[index])
        return status

    @staticmethod
    def _failure(errstr):
        '''_failure ds'''
        log.warning('--< CLASSIFICATION: %s >--', errstr)
        return
    pass
