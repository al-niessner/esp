'''ariel __main__ ds'''
# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security
import dawgie.util

import excalibur.ariel.bot
# ------------- ------------------------------------------------------
fep = os.environ.get('FE_PORT', None)
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)

if fep: dawgie.util.set_ports(int(fep))

dawgie.security.initialize(os.path.expandvars
                           (os.path.expanduser
                            (dawgie.context.guest_public_keys)),
                           myname=dawgie.context.ssl_pem_myname,
                           myself=dawgie.context.ssl_pem_myself)
dawgie.db.reopen()
excalibur.ariel.bot.Actor('ariel', 4, rid, tn).do()

dawgie.db.close()
dawgie.security.finalize()
