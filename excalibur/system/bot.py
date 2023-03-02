'''system bot ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.system.algorithms as sysalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    '''Actor ds'''
    def list(self):
        '''Subtasks top level ordered call'''
        return [
            sysalg.validate(),
            sysalg.finalize()
        ]
    pass
# --------- ----------------------------------------------------------
