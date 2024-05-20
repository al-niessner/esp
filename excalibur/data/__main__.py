'''data __main__ ds'''
# -- IMPORTS -- ------------------------------------------------------
import os

import dawgie
import dawgie.db
import dawgie.security
import dawgie.util

import excalibur.data.bot
# ------------- ------------------------------------------------------
fep = os.environ.get('FE_PORT', None)
rid = int(os.environ.get('RUNID', None))
tn = os.environ.get('TARGET_NAME', None)

if fep: dawgie.util.set_ports(int(fep))

dawgie.security.initialize(os.path.expandvars(os.path.expanduser
                                              (dawgie.context.gpg_home)))
dawgie.db.reopen()

if tn in ['', '__all__']: pass
else:
    # -- THIS IS EVIL PYTHON -- --------------------------------------
    subtasks = excalibur.data.bot.Actor('data', 4, rid, tn)
    fulllist = getattr(subtasks, 'list')
    def shortlist():
        '''666'''
        out = fulllist()
        # -- Change indexes as needed and look away from those lines
        # 0 datalg.collect(),
        # 1 datalg.timing(),
        # 2 datalg.calibration()
        out = out[0:]
        # ------------------------- ----------------------------------
        return out
    setattr(subtasks, 'list', shortlist)
    pass

try: subtasks.do()
except NameError: print('>-- !!! Target name needed !!!')
dawgie.db.close()
dawgie.security.finalize()
