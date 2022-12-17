'''transmission spectrum task'''
ignore = False

# -- IMPORTS -- ------------------------------------------------------
import excalibur.taurex.bot
# ------------- ------------------------------------------------------

def task (prefix:str, ps_hint:int=0, runid:int=-1, target:str='__none__'):
    '''tasks that need to be done'''
    return excalibur.taurex.bot.Actor(prefix, ps_hint, runid, target)
