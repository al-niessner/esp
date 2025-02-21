'''ancillary bot ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.classifier.algorithms as clsalg


# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    '''Actor ds'''

    def list(self) -> [dawgie.Task]:
        '''GMR: Kicked inference() from this'''
        return [clsalg.Flags()]

    pass


# --------------------------------------------------------------------
class Agent(dawgie.Analysis):
    '''Agent ds'''

    def list(self) -> [dawgie.Analyzer]:
        '''Subtasks top level ordered call'''
        return [clsalg.SummarizeFlags()]

    pass
