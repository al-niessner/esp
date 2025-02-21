'''phasecurve __main__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.db
import dawgie.security

from excalibur.util.main import main_start

import excalibur.phasecurve.bot

# ------------- ------------------------------------------------------

rid, tn = main_start()

if tn in ['', '__all__']:
    pass
else:
    NAME = ['normalization', 'whitelight', None][-1]  # -1 to run them all
    SUBTASKS = excalibur.phasecurve.bot.Actor('phasecurve', 4, rid, tn)
    SUBTASKS.do(NAME)

dawgie.db.close()
dawgie.security.finalize()
