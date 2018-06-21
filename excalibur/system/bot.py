# -- IMPORTS -- ------------------------------------------------------
import os
import platform

import dawgie
import dawgie.db
import dawgie.security

import exo.spec.ae.system.algorithms as sysalg
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
if __name__ == '__main__':
    name = os.environ.get ('DO_NAME', None)
    rid = int(os.environ.get('RUNID', None))
    tn = os.environ.get('TARGET_NAME', None)
    dawgie.context._ports(int(os.environ.get('FE_PORT', None)))
    dawgie.security.initialize(os.path.expandvars
                               (os.path.expanduser
                                (dawgie.context.gpg_home)))
    dawgie.db.reopen()
    Actor('system', 4, rid, tn).do()
    dawgie.db.close()
    dawgie.security.finalize()
    pass
