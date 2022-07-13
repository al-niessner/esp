'''data bot ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.data.algorithms as datalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    '''Actor ds'''
    def list(self)->[dawgie.Task]:
        '''Subtasks top level ordered call'''
        return [
            datalg.collect(),
            datalg.timing(),
            datalg.calibration()
        ]
    pass
# --------- ----------------------------------------------------------
