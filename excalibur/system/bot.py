# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.system.algorithms as sysalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    def list(self):
        return [
            sysalg.validate(),
            sysalg.finalize()
        ]
    pass
# --------- ----------------------------------------------------------
