'''ariel algorithms ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import dawgie

import excalibur.ariel.core as arielcore
import excalibur.ariel.states as arielstates
import excalibur.runtime as rtime
import excalibur.runtime.algorithms as rtalg
import excalibur.system as sys
import excalibur.system.algorithms as sysalg

# ------------- ------------------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class sim_spectrum(dawgie.Algorithm):
    '''
    Create a simulated Ariel spectrum
    '''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,0,0)
        self.__rt = rtalg.autofill()
        self.__system_finalize = sysalg.finalize()
        self.__out = arielstates.PriorsSV('parameters')
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'sim_spectrum'

    def previous(self):
        '''Input State Vectors: system.finalize'''
        return [dawgie.ALG_REF(sys.task, self.__system_finalize),
                dawgie.V_REF(rtime.task, self.__rt, self.__rt.sv_as_dict()['status'],
                             'includeMetallicityDispersion')] + \
               self.__rt.refs_for_validity()

    def state_vectors(self):
        '''Output State Vectors: ariel.sim_spectrum'''
        return [self.__out]

    def run(self, ds, ps):
        '''Top level algorithm call'''

        # stop here if it is not a runtime target
        if not self.__rt.is_valid():
            log.warning('--< ARIEL.%s: not a valid target >--', self.name().upper())

        else:
            update = False

            system_dict = self.__system_finalize.sv_as_dict()['parameters']
            valid, errstring = arielcore.checksv(system_dict)
            if valid:
                runtime = self.__rt.sv_as_dict()['status']
                runtime_params = arielcore.ARIEL_PARAMS(
                    tier=1,
                    randomSeed=123,
                    randomCloudProperties=True,
                    thorgrenMassMetals=True,
                    includeMetallicityDispersion=runtime[
                        'ariel_simulate_spectra_includeMetallicityDispersion'])
                # FIXMEE: this code needs repaired by moving out to config (Geoff added)
                update = self._simSpectrum(repr(self).split('.')[1],  # this is the target name
                                           system_dict, runtime_params,
                                           self.__out)
            else:
                self._failure(errstring)
            if update: ds.update()
            elif valid: raise dawgie.NoValidOutputDataError(
                    f'No output created for ARIEL.{self.name()}')
        return

    @staticmethod
    def _simSpectrum(target, system_dict, runtime_params, out):
        '''Core code call'''
        filled = arielcore.simulate_spectra(target, system_dict, runtime_params, out)
        return filled

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< ARIEL SIM_SPECTRUM: %s >--', errstr)
        return
