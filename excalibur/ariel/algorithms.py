'''ariel algorithms ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import dawgie

import excalibur.ariel.core as arielcore
import excalibur.ariel.states as arielstates

import excalibur.system as sys
import excalibur.system.bot as sysbot
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
        self.__system_finalize = sysalg.finalize()
        self.__out = arielstates.PriorsSV('parameters')
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'sim_spectrum'

    def previous(self):
        '''Input State Vectors: system.finalize'''
        return [dawgie.ALG_REF(sys.task, self.__system_finalize)]

    def state_vectors(self):
        '''Output State Vectors: ariel.sim_spectrum'''
        return [self.__out]

    def run(self, ds, ps):
        '''Top level algorithm call'''
        update = False

        # pylint: disable=protected-access
        target = ds._tn()

        task = sysbot.Actor('system', 4, 999, target)
        subtask = sysalg.finalize()
        dataset = dawgie.db.connect(subtask, task, target); dataset.load()
        system_dict = subtask.sv_as_dict()['parameters']
        valid, errstring = arielcore.checksv(system_dict)
        if valid:
            update = self._simSpectrum(target, system_dict, self.__out)
        else:
            self._failure(errstring)
        if update:
            ds.update()
        else:
            raise dawgie.NoValidOutputDataError(
                f'No output created for ARIEL.{self.name()}')
        return

    @staticmethod
    def _simSpectrum(target, system_dict, out):
        '''Core code call'''
        filled = arielcore.simulate_spectra(target, system_dict, out)
        return filled

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< ARIEL SIM_SPECTRUM: %s >--', errstr)
        return
