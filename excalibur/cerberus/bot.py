# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.cerberus.algorithms as crbalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            crbalg.xslib(),
            crbalg.atmos()
        ]
    pass
# ---------- ---------------------------------------------------------
