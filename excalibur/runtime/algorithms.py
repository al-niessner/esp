'''runtime configuration algorithms'''

import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context

from . import core
from . import states

class autofill(dawgie.Task):
    pass

class create(dawgie.Analyzer):
    def __init__ (self):
        self.__table = states.ConfigSV()
        return
    def name(self)->str: return 'settings'
    def run(self, aspects:dawgie.Aspect)->None:
        config = core.load()
        return
    def state_vectors(self)->[dawgie.StateVector]: return [self.__table]
    def traits(self): return []
    def where(self): return dawgie.Distribution.cluster
    pass
    
