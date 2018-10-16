# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security

import excalibur.target.bot
# ------------- ------------------------------------------------------
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)
dawgie.context._ports(int(os.environ.get('FE_PORT', None)))
dawgie.security.initialize(os.path.expandvars(os.path.expanduser
                                              (dawgie.context.gpg_home)))
dawgie.db.reopen()
excalibur.target.bot.Actor('target', 4, rid).do()
excalibur.target.bot.Agent('target', 4, rid, tn).do()
dawgie.db.close()
dawgie.security.finalize()
