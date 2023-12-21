'''target __main__ ds'''
# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security
import dawgie.util

import excalibur.target.bot
# ------------- ------------------------------------------------------
fep = os.environ.get('FE_PORT', None)
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)

if fep: dawgie.util.set_ports(int(fep))

dawgie.security.initialize(os.path.expandvars(os.path.expanduser
                                              (dawgie.context.gpg_home)))
dawgie.db.reopen()
excalibur.target.bot.Actor('target', 4, rid).do()
if tn == '':
    for tn in dawgie.db.targets():
        # excalibur.target.bot.Agent('target', 4, rid, tn).do()
        pass
else:
    excalibur.target.bot.Agent('target', 4, rid, tn).do()
dawgie.db.close()
dawgie.security.finalize()
