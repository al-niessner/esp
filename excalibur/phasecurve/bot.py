# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.phasecurve.algorithms as phcalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            phcalg.normalization()
        ]
    pass
# --------- ----------------------------------------------------------
