'''ancillary bot ds'''
#  Careful: Commenting out tasks below will mess up the pipeline during startup
#          (tasks listed here are compiled into the algorithm/task tree)

# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.ancillary.algorithms as ancalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Agent(dawgie.Analysis):
    '''Agent ds'''
    def list(self)->[dawgie.Analyzer]:
        '''list ds'''
        return [
            ancalg.population()
        ]
    pass

class Actor(dawgie.Task):
    '''Actor ds'''
    def list(self)->[dawgie.Task]:
        '''list ds'''
        return [
            ancalg.estimate()
        ]
    pass
# --------- ----------------------------------------------------------
