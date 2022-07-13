'''ancillary __main__ ds'''
# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security
import dawgie.util

import excalibur.ancillary.bot
# ------------- ------------------------------------------------------
fep = os.environ.get('FE_PORT', None)
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)

if fep: dawgie.util.set_ports(int(fep))

dawgie.security.initialize(os.path.expandvars(os.path.expanduser
                                              (dawgie.context.gpg_home)))
dawgie.db.reopen()
excalibur.ancillary.bot.Agent('ancillary', 4, rid, tn).do()
excalibur.ancillary.bot.Actor('ancillary', 4, rid).do()
dawgie.db.close()
dawgie.security.finalize()
