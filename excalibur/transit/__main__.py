# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security

import excalibur.transit.bot
# ------------- ------------------------------------------------------
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)
dawgie.context._ports(int(os.environ.get('FE_PORT', None)))
dawgie.security.initialize(os.path.expandvars(os.path.expanduser
                                              (dawgie.context.gpg_home)))
dawgie.db.reopen()
excalibur.transit.bot.Actor('transit', 4, rid, tn).do()
dawgie.db.close()
dawgie.security.finalize()
