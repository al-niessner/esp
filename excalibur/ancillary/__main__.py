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

dawgie.security.initialize(os.path.expandvars
                           (os.path.expanduser
                            (dawgie.context.guest_public_keys)))
dawgie.db.reopen()

if tn in ['', '__all__']:
    name = 'population'
    subtasks = excalibur.ancillary.bot.Agent('ancillary', 4, rid)
else:
    name = 'estimate'
    subtasks = excalibur.ancillary.bot.Actor('ancillary', 4, rid, tn)
    pass

subtasks.do(name)
dawgie.db.close()
dawgie.security.finalize()
