'''cerberus __main__ ds'''
# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security
import dawgie.util

import excalibur.cerberus.bot
# ------------- ------------------------------------------------------
fep = os.environ.get('FE_PORT', None)
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)

if fep: dawgie.util.set_ports(int(fep))

dawgie.security.initialize(os.path.expandvars(os.path.expanduser
                                              (dawgie.context.gpg_home)))
dawgie.db.reopen()

if tn == '':
    excalibur.cerberus.bot.Agent('cerberus', 4, rid).do()
else:
    excalibur.cerberus.bot.Actor('cerberus', 4, rid, tn).do()

dawgie.db.close()
dawgie.security.finalize()
