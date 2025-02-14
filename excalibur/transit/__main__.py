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
    name = 'population'
    subtasks = excalibur.transit.bot.Agent('transit', 4, rid)
    pass
else:
    name = ['normalization', 'spectrum', 'whitelight', 'starspots', None][
        -1
    ]  # -1 to run them all
    subtasks = excalibur.transit.bot.Actor('transit', 4, rid, tn)
    pass

subtasks.do(name)
dawgie.db.close()
dawgie.security.finalize()
