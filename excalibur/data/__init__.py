'''data __init__ ds'''

ignore = False

# -- IMPORTS -- ------------------------------------------------------
import excalibur.data.bot as datbot
# ------------- ------------------------------------------------------
def task (prefix:str, ps_hint:int=0, runid:int=-1, target:str='__none__'):
    '''task ds'''
    return datbot.Actor(prefix, ps_hint, runid, target)
