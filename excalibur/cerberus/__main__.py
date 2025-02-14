'''cerberus __main__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.db
import dawgie.security

from excalibur.util.main import main_start

import excalibur.cerberus.bot

# ------------- ------------------------------------------------------

rid, tn = main_start()

if tn in ['', '__all__']:
    name = 'analysis'
    subtasks = excalibur.cerberus.bot.Agent('cerberus', 4, rid)
else:
    name = ['atmos', 'results', 'xslib', None][-1]  # -1 to run them all
    subtasks = excalibur.cerberus.bot.Actor('cerberus', 4, rid, tn)
    pass

subtasks.do(name)
dawgie.db.close()
dawgie.security.finalize()
