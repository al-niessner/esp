# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.data.algorithms as datalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            datalg.collect(),
            datalg.timing(),
            datalg.calibration()
        ]
    pass
# --------- ----------------------------------------------------------
