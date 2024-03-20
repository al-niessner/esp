'''runtime configuration algorithms'''

import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context
import os

from . import core
from . import states

class autofill(dawgie.Task):
    pass

class create(dawgie.Analyzer):
    def __init__ (self):
        self.__table = [states.ControlsSV(), states.FilterSV(),
                        states.PymcSV('cerberus'), states.PymcSV('spectrum')]
        self.__table.append (states.CompositeSV(self.__table))
        return
    def name(self)->str: return 'settings'
    def run(self, aspects:dawgie.Aspect)->None:
        try:
            core.load(self.sv_as_dict())
        except FileNotFoundError:
            raise dawgie.AbortAEError(f'The environment variable {core.ENV_NAME} points to the non-existent file: {os.environ[core.ENV_NAME]}')
        except KeyError:
            raise dawgie.AbortAEError(f'The environment variable {core.ENV_NAME} must be defined.')
        except Exception as e:
            raise dawgie.AbortAEError(f'The contents pointed to by {core.ENV_NAME} cannot be parsed. Try validating {os.environ[core.ENV_NAME]} with xmllint.')
        return
    def state_vectors(self)->[dawgie.StateVector]: return self.__table
    def traits(self): return []
    def where(self): return dawgie.Distribution.cluster
    pass
    
