'''ancillary __main__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.db
import dawgie.security

from excalibur.util.main import main_start

import excalibur.ancillary.bot

# ------------- ------------------------------------------------------

rid, tn = main_start()

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
