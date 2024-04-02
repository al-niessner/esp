'''runtime configuration algorithms'''

import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context
import dawgie.db
import os

from . import core
from . import states

class autofill(dawgie.Algorithm):
    '''Breaks the levers and knobs global table to target specific'''
    def __init__ (self):
        '''init autofill'''
        self._version_ = dawgie.VERSION(1,0,0)
        self.__parent = create()
        self.__status = states.StatusSV()
    def name(self):
        '''database name'''
        return 'autofill'
    def previous(self):
        '''state vectors needed for graph

        The pipeline never really runs this task. Instead, it is called from
        runtime.create where the table and target name are given
        '''
        import excalibur.runtime  # avoid circular dependency; pylint: disable=import-outside-toplevel
        return [dawgie.SV_REF(excalibur.runtime.analysis, self.__parent,
                              self.__parent.sv_as_dict()['composite'])]
    def run (self, ds, ps, table:{str:{}}=None, tn=None):
        '''isolate target specific information from the global table'''
        if table is None or tn is None:
            raise dawgie.NoValidInputDataError('never should be called except'
                                               'from runtime.create')
        core.isolate (self.__status, table, tn)
        ds.update()
    def state_vectors(self):
        '''state vectors generated from this algorithm'''
        return [self.__status]
    pass

class create(dawgie.Analyzer):
    '''Read the configuration file then turn it into state vectors'''
    def __init__ (self):
        '''init the create process'''
        self._version_ = dawgie.VERSION(1,0,0)
        self.__table = [states.ControlsSV(), states.FilterSV(),
                        states.PymcSV('cerberus'), states.PymcSV('spectrum'),
                        states.TargetsSV('run_only'),
                        states.TargetsSV('sequester')]
        self.__table.append (states.CompositeSV(self.__table))
    def name(self)->str:
        '''database name'''
        return 'settings'
    def run(self, aspects:dawgie.Aspect)->None:
        '''load the configuration file then process it'''
        try:
            core.load(self.sv_as_dict(), dawgie.db.targets())
            aspects.ds().update()
            # Now do the evil bit where the table is divided into target
            # specific state vectors where the information becomes highly
            # condensed and processed. To do this, need to act like dawgie
            # just a little bit and access some hidden information.
            import excalibur.runtime.bot as erb  # pylint: disable=import-outside-toplevel
            pbot = aspects.ds()._bot()  # pylint: disable=protected-access
            for tn in dawgie.db.targets():
                bot = erb.TaskTeam(pbot.name(), 1, pbot._runid(), tn)  # pylint: disable=protected-access
                alg = autofill()
                ds = dawgie.db.connect (alg, bot, tn)
                alg.run (ds, 1, self.sv_as_dict(), tn)
        except FileNotFoundError as e:
            log.exception(e)
            raise dawgie.AbortAEError(f'The environment variable {core.ENV_NAME} points to the non-existent file: {os.environ[core.ENV_NAME]}') from e
        except KeyError as e:
            log.exception(e)
            raise dawgie.AbortAEError(f'The environment variable {core.ENV_NAME} must be defined.') from e
        except Exception as e:
            log.exception(e)
            raise dawgie.AbortAEError(f'The contents pointed to by {core.ENV_NAME} cannot be parsed. Try validating {os.environ[core.ENV_NAME]} with xmllint.') from e
        return
    def state_vectors(self)->[dawgie.StateVector]:
        '''produce the configuration file as simple set of state vectors'''
        return self.__table
    def traits(self):
        '''no traits are required'''
        return []
    def where(self):
        '''make sure it always runs on the local cluster for disk access'''
        return dawgie.Distribution.cluster
    pass
