'''ancillary __main__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security
import dawgie.util

import excalibur.classifier.bot
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
    name = 'summarize_flags'
    subtasks = excalibur.classifier.bot.Agent('classifier', 4, rid)
else:
    name = ['inference', 'flags', None][-1]  # 0 is kicked off list, -1 to run them all
    subtasks = excalibur.classifier.bot.Actor('classifier', 4, rid, tn)
    pass

subtasks.do(name)
dawgie.db.close()
dawgie.security.finalize()
