'''eclipse bot ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.eclipse.algorithms as eclalg


# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    '''Actor ds'''

    def list(self) -> [dawgie.Task]:
        '''Subtasks top level ordered call'''
        return [eclalg.Normalization(), eclalg.WhiteLight(), eclalg.Spectrum()]

    pass


# --------- ----------------------------------------------------------
