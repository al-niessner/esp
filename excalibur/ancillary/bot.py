# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.ancillary.algorithms as ancalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Analysis):
    def list(self)->[dawgie.Analyzer]:
        return [
            ancalg.population()
        ]
    pass

class Agent(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            ancalg.estimate()
        ]
    pass
# --------- ----------------------------------------------------------
