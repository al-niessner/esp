'''transit __main__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.db
import dawgie.security

from excalibur.util.main import main_start

import excalibur.transit.bot

# ------------- ------------------------------------------------------

rid, tn = main_start()

if tn in ['', '__all__']:
    NAME = 'population'
    subtasks = excalibur.transit.bot.Agent('transit', 4, rid)
    pass
else:
    NAME = ['normalization', 'spectrum', 'whitelight', 'starspots', None][
        -1
    ]  # -1 to run them all
    subtasks = excalibur.transit.bot.Actor('transit', 4, rid, tn)
    pass

subtasks.do(NAME)
dawgie.db.close()
dawgie.security.finalize()
