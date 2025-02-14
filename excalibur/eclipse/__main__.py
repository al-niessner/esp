'''eclipse __main__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.db
import dawgie.security

from excalibur.util.main import main_start

import excalibur.eclipse.bot

# ------------- ------------------------------------------------------

rid, tn = main_start()

if tn in ['', '__all__']:
    pass
else:
    name = ['normalize', 'spectrum', 'whitelight', None][
        -1
    ]  # -1 to run them all
    subtasks = excalibur.eclipse.bot.Actor('eclipse', 4, rid, tn)

    subtasks.do(name)

dawgie.db.close()
dawgie.security.finalize()
