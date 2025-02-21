'''data __main__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.db
import dawgie.security

from excalibur.util.main import main_start

import excalibur.data.bot

# ------------- ------------------------------------------------------

rid, tn = main_start()

if tn in ['', '__all__']:
    pass
else:
    NAME = ['calibration', 'collect', 'timing', None][-1]  # -1 to run them all
    subtasks = excalibur.data.bot.Actor('data', 4, rid, tn)

    subtasks.do(NAME)

dawgie.db.close()
dawgie.security.finalize()
