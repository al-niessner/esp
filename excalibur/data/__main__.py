import os
import dawgie
import dawgie.db
import dawgie.security
import excalibur.data.bot

name = os.environ.get ('DO_NAME', None)
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)
dawgie.context._ports(int(os.environ.get('FE_PORT', None)))
dawgie.security.initialize(os.path.expandvars
                           (os.path.expanduser
                            (dawgie.context.gpg_home)))
dawgie.db.reopen()
excalibur.data.bot.Actor('data', 4, rid, tn).do()
dawgie.db.close()
dawgie.security.finalize()
