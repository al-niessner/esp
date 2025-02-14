'''ariel __main__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.db
import dawgie.security

from excalibur.util.main import main_start

import excalibur.ariel.bot

# ------------- ------------------------------------------------------

rid, tn = main_start()

excalibur.ariel.bot.Actor('ariel', 4, rid, tn).do()

dawgie.db.close()
dawgie.security.finalize()
