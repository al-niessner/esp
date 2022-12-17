'''taurex bot'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.taurex.algorithms
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    '''Actor ds'''
    def list(self)->[dawgie.Task]:
        '''Subtasks top level ordered call'''
        return [
            excalibur.taurex.algorithms.TransitSpectrumInjection(),
        ]
    pass
# --------- ----------------------------------------------------------
