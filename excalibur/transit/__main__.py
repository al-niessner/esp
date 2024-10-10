'''transit __main__ ds'''
# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security

import excalibur.transit.bot
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
    subtasks = excalibur.transit.bot.Agent('transit', 4, rid)
    pass
else:
    name = ['normalization', 'spectrum', 'whitelight', 'starspots', None][-1]  # -1 to run them all
    subtasks = excalibur.transit.bot.Actor('transit', 4, rid, tn)
    pass

subtasks.do(name)
dawgie.db.close()
dawgie.security.finalize()
