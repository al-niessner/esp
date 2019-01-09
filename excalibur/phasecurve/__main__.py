import os
import dawgie
import dawgie.db
import dawgie.security
import dawgie.util

import excalibur.phasecurve.bot

fep = os.environ.get('FE_PORT', None)
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)

if fep: dawgie.util.set_ports(int(fep))

dawgie.security.initialize(os.path.expandvars(os.path.expanduser
                                              (dawgie.context.gpg_home)))
dawgie.db.reopen()
excalibur.phasecurve.bot.Actor('phasecurve', 4, rid).do()
dawgie.db.close()
dawgie.security.finalize()
