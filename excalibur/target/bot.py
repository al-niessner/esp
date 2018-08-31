# -- IMPORTS -- ------------------------------------------------------
import dawgie
import excalibur.target.algorithms as trgalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Analysis):
    def list(self)->[dawgie.Analyzer]:
        return [
            trgalg.create()
        ]
    pass

class Agent(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            trgalg.autofill(),
            trgalg.scrape()
        ]
    pass
# --------- ----------------------------------------------------------
