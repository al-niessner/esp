'''ariel bot ds'''
# -- IMPORTS ---------------------------------------------------------
import dawgie

import excalibur.ariel.algorithms as arielalg
# --------------------------------------------------------------------
# -- A&A -------------------------------------------------------------
class Actor(dawgie.Task):
    '''Actor ds'''
    def list(self):
        '''Subtasks top level ordered call'''
        return [
            arielalg.sim_spectrum(),
        ]
# --------- ----------------------------------------------------------
