# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.eclipse.algorithms as eclalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            eclalg.normalization(),
            eclalg.whitelight(),
            eclalg.spectrum()
        ]
    pass
# --------- ----------------------------------------------------------
