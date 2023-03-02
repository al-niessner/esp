'''
SYSTEM manages the astrophysical parameters of the target observed
- VALIDATE parameters from target.autofill, report missing parameters
- FINALIZE parameters using target/edit.py function ppar()
'''
ignore = False

# -- IMPORTS -- ------------------------------------------------------
import excalibur.system.bot as sysbot
# ------------- ------------------------------------------------------

def task (prefix:str, ps_hint:int=0, runid:int=-1, target:str='__none__'):
    '''Factory'''
    return sysbot.Actor(prefix, ps_hint, runid, target)
