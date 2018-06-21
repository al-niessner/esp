# -- IMPORTS -- ------------------------------------------------------
import os
import sys
import pdb

import dawgie
import exo.spec.ae.cerberus.algorithms as crbalg
# ------------- ------------------------------------------------------
# -- TASK -- ---------------------------------------------------------
class Actor(dawgie.Task):
    def list(self):
        return [
            crbalg.xslib(),
#            crbalg.hazelib(),
#            crbalg.atmos()
        ]
    pass
# ---------- ---------------------------------------------------------
if __name__ == '__main__':
    rid = int(os.environ.get('RUNID', None))
    tn = os.environ.get('TARGET_NAME', None)
    dawgie.context._ports(int(os.environ.get('FE_PORT', None)))
    dawgie.context.db_host = os.environ.get('DB_HOST', None)
    dawgie.security.initialize(os.path.expandvars
                                 (os.path.expanduser
                                  (dawgie.context.gpg_home)))
    dawgie.db.reopen()
    if tn is None:
        targetlist = [
            '55 Cnc',
            'GJ 1214', 'GJ 3470', 'GJ 436',
            'HAT-P-1', 'HAT-P-11', 'HAT-P-12', 'HAT-P-18',
	    'HD 189733', 'HD 209458', 'HD 97658',
            'WASP-103', 'WASP-12', 'WASP-18', 'WASP-19',
            'WASP-31', 'WASP-33', 'WASP-43',
            'XO-1', 'XO-2'
        ]
        pass
    
    else: targetlist = [tn]
    for target in targetlist: Actor('cerberus', 4, rid, target).do()
    dawgie.db.close()
    dawgie.security.finalize()
    pass
