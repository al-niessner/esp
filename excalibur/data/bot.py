# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security

import excalibur.data.algorithms as datalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Task):
    def list(self)->[dawgie.Task]:
        return [
            datalg.collect(),
            datalg.calibration(),
            datalg.timing()
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
    Actor('data', 4, rid, tn).do()
    dawgie.db.close()
    dawgie.security.finalize()
    pass
