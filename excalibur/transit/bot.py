'''transit bot ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.transit.algorithms as trnalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    '''Actor ds'''
    def list(self)->[dawgie.Task]:
        '''Subtasks top level ordered call'''
        return [
            trnalg.normalization(),
            trnalg.whitelight(),
            trnalg.spectrum(),
            trnalg.starspots()
        ]
    pass

class Agent(dawgie.Analysis):
    '''Agent ds'''
    def list(self)->[dawgie.Analyzer]:
        '''Subtasks top level ordered call'''
        return [
            trnalg.population()
        ]
    pass
# --------- ----------------------------------------------------------
