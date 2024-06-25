'''runtime configuration algorithms'''

import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context
import dawgie.db
import multiprocessing
import os

from . import core
from . import states

class autofill(dawgie.Algorithm):
    '''Breaks the levers and knobs global table to target specific'''
    def __init__ (self, table:{str:{}}=None, tn:str=None):
        '''init autofill'''
        self._version_ = dawgie.VERSION(1,0,0)
        self.__parent = create()
        self.__status = states.StatusSV()
        self.__table = table
        self.__tn = tn
    def is_valid(self):
        '''convenience function'''
        return self.__status['isValidTarget']
    def name(self):
        '''database name'''
        return 'autofill'
    def previous(self):
        '''state vectors needed for graph

        The pipeline never really runs this task. Instead, it is called from
        runtime.create where the table and target name are given
        '''
        # avoid circular dependency
        import excalibur.runtime  # pylint: disable=import-outside-toplevel
        return [dawgie.SV_REF(excalibur.runtime.analysis, self.__parent,
                              self.__parent.sv_as_dict()['composite'])]
    def proceed(self, ext:str=None):
        '''convenience function'''
        return self.__status.proceed(ext)
    def refs_for_proceed(self)->[dawgie.V_REF]:
        '''return minimum list for StatusSV.proceed() to work'''
        # avoid circular dependency
        import excalibur.runtime  # pylint: disable=import-outside-toplevel
        return [dawgie.V_REF(excalibur.runtime.task, self,
                             self.__status,'allowed_exts'),
                dawgie.V_REF(excalibur.runtime.task, self,
                             self.__status,'isValidTarget'),
                dawgie.V_REF(excalibur.runtime.task, self,
                             self.__status,'runTarget')]
    def refs_for_validity(self)->[dawgie.V_REF]:
        '''return minimum list for StatusSV to determine if target is valid'''
        # avoid circular dependency
        import excalibur.runtime  # pylint: disable=import-outside-toplevel
        return [dawgie.V_REF(excalibur.runtime.task, self,
                             self.__status,'isValidTarget')]
    def run (self, ds, ps):
        '''isolate target specific information from the global table'''
        if self.__table is None or self.__tn is None:
            raise dawgie.NoValidInputDataError('never should be called except'
                                               'from runtime.create')
        core.isolate (self.__status, self.__table, self.__tn)
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
    @staticmethod
    def _do(arg):
        # break circular dependencies with
        # pylint: disable=import-outside-toplevel
        import excalibur.runtime.bot as erb
        erb.TaskTeam(*arg[0],**arg[1]).do()
    def name(self)->str:
        '''database name'''
        return 'create'
    def run(self, aspects:dawgie.Aspect)->None:
        '''load the configuration file then process it'''
        try:
            log.info('starting load of config')
            core.load(self.sv_as_dict(), dawgie.db.targets())
            log.info('updating state vector')
            aspects.ds().update()
            # Now do the evil bit where the table is divided into target
            # specific state vectors where the information becomes highly
            # condensed and processed. To do this, need to act like dawgie
            # just a little bit and access some hidden information.
            # need under the hood for this, pylint: disable=protected-access
            pbot = aspects.ds()._bot()
            with multiprocessing.Pool(processes=60) as pool:
                log.info('using the pool to run in parallel')
                pool.map (create._do, [((pbot._name(), 1, pbot._runid(), tn),
                                        {'table':self.sv_as_dict(),'this_tn':tn})
                                       for tn in dawgie.db.targets()])
            # done under the hood, pylint: enable=protected-access
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
