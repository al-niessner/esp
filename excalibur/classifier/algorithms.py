import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context

import excalibur.transit as trn
import excalibur.transit.core as trncore
import excalibur.transit.algorithms as trnalg

import excalibur.target.edit as trgedit

import excalibur.system as sys
import excalibur.system.algorithms as sysalg

import excalibur.classifier.core as clscore
import excalibur.classifier.states as clsstates
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
fltrs = [f for f in fltrs if 'Spitzer' not in f]
# ---------------------- ---------------------------------------------

# -- ALGORITHMS -- ---------------------------------------------------
class inference(dawgie.Algorithm):
    '''
G. ROUDIER: Data collection by filters
    '''
    def __init__(self):
        self._version_ = clscore.predversion()
        self.__whitelight = trnalg.whitelight()
        self.__finalize = sysalg.finalize()
        self.__out = [clsstates.PredictSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'inference'

    def previous(self):
        # can include model.pkl in train.py or the training script itself
        return [dawgie.ALG_REF(trn.task, self.__whitelight),
                dawgie.ALG_REF(sys.task, self.__finalize)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        vfin, sfin = trncore.checksv(self.__finalize.sv_as_dict()['parameters'])
        for ext in fltrs:
            update = False
            vwl, swl = trncore.checksv(self.__whitelight.sv_as_dict()[ext])

            if vwl and vfin:
                log.warning('--< CLASSIFICATION: %s >--', ext)
                update = self._predict(self.__whitelight.sv_as_dict()[ext],
                                       self.__finalize.sv_as_dict()['parameters'],
                                       fltrs.index(ext))
                pass
            else:
                errstr = [m for m in [swl, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[fltrs.index(ext)])
            pass
        self.__out = svupdate
        if self.__out: ds.update()
        return

    def _predict(self, wl, fin, index):
        status = clscore.predict(wl, fin['priors'], self.__out[index])
        return status

    @staticmethod
    def _failure(errstr):
        log.warning('--< CLASSIFICATION: %s >--', errstr)
        return
    pass