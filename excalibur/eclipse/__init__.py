'''eclipse __init__ ds'''

ignore = False

# -- IMPORTS -- ------------------------------------------------------
import excalibur.eclipse.bot as eclbot
# ------------- ------------------------------------------------------

def task (prefix:str, ps_hint:int=0, runid:int=-1, target:str='__none__'):
    '''task ds'''
    return eclbot.Actor(prefix, ps_hint, runid, target)
