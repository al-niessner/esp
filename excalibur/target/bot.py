# -- IMPORTS -- ------------------------------------------------------
import os
import platform

import dawgie
import dawgie.db
import dawgie.security

import exo.spec.ae.target.algorithms as trgalg
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
if __name__ == '__main__':
    name = os.environ.get ('DO_NAME', None)
    rid = int(os.environ.get('RUNID', None))
    tn = os.environ.get('TARGET_NAME', None)
    dawgie.context._ports(int(os.environ.get('FE_PORT', None)))
    dawgie.security.initialize(os.path.expandvars
                               (os.path.expanduser
                                (dawgie.context.gpg_home)))
    dawgie.db.reopen()
    Actor('target', 4, rid).do()
    Agent('target', 4, rid, tn).do()
    dawgie.db.close()
    dawgie.security.finalize()
    pass
