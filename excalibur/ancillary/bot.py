'''ancillary bot ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.ancillary.algorithms as ancalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Analysis):
    '''Actor ds'''
    def list(self)->[dawgie.Analyzer]:
        '''list ds'''
        return [
            ancalg.population()
        ]
    pass

class Agent(dawgie.Task):
    '''Agent ds'''
    def list(self)->[dawgie.Task]:
        '''list ds'''
        return [
            ancalg.estimate()
        ]
    pass
# --------- ----------------------------------------------------------
