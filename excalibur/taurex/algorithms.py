'''connect the TauRex model to excalibur data'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.db

import logging; log = logging.getLogger(__name__)

import excalibur.taurex.core

import excalibur.system as sys
import excalibur.system.algorithms as sysalg
import excalibur.transit as trn
import excalibur.transit.algorithms as trnalg
import excalibur.transit.states as trnstates

# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# Injectable Targets and filters
PROCESS_FILTERS = ['HST-WFC3-IR-G141-SCAN']
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------

class TransitSpectrumInjection(dawgie.Algorithm):
    '''injects data into the transit.spectrum data stream'''
    def __init__(self):
        '''initial state'''
        self._version_ = excalibur.taurex.core.tsiversion()
        self.__fin = sysalg.finalize()
        self.__out = [trnstates.SpectrumSV(ext) for ext in PROCESS_FILTERS]
        self.__spt = trnalg.spectrum()
        sv_dict = self.__spt.sv_as_dict()
        self.__pre = [sv_dict[ext]
                      for ext in filter (lambda k,d=sv_dict:k in d,
                                          PROCESS_FILTERS)]
        return

    def name(self):
        '''excalibur algorithm name'''
        return 'spectral-sim'

    def previous(self):
        '''Input state vectors: transit.spectrum, system.finalize'''
        spts = [dawgie.SV_REF(trn.task, self.__spt, sv) for sv in self.__pre]
        return spts + [dawgie.SV_REF(sys.task, self.__fin,
                                     self.__fin.sv_as_dict()['parameters'])]

    def run(self, ds, ps):
        '''process desired excalibur data'''
        if not self.__pre or not any (sv['STATUS'][-1] for sv in self.__pre):
            raise dawgie.NoValidInputDataError('exptected filter(s) not in SV')

        excalibur.taurex.core.init()
        for spectrum in filter (lambda d:d['STATUS'][-1], self.__pre):
            excalibur.taurex.core.tsi (spectrum,
                                       self.__fin.sv_as_dict()['parameters'])
            self.sv_as_dict()[spectrum.name()].update (spectrum)
            pass
        ds.retarget('taurex sim @TS',
                    [dawgie.ALG_REF(sys.task, self.__fin)]).update()
        return

    def state_vectors(self):
        '''get the state vectors of this algorithm'''
        return self.__out
    pass
