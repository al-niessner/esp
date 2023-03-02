
'''cerberus bot ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.cerberus.algorithms as crbalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    '''Actor ds'''
    def list(self)->[dawgie.Task]:
        '''Subtasks top level ordered call'''
        return [
            crbalg.xslib(),
            crbalg.atmos(),
            crbalg.release()
        ]
    pass
# ---------- ---------------------------------------------------------
