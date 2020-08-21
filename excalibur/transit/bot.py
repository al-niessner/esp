# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.transit.algorithms as trnalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            trnalg.normalization(),
            trnalg.whitelight(),
            trnalg.spectrum()
        ]
    pass

class Agent(dawgie.Analysis):
    def list(self)->[dawgie.Analyzer]:
        return [
            trnalg.population()
        ]
    pass
# --------- ----------------------------------------------------------
