'''cerberus __init__ ds'''

ignore = False

# -- IMPORTS -- ------------------------------------------------------
import excalibur.cerberus.bot as crbbot
# ------------- ------------------------------------------------------

def analysis (prefix:str, ps_hint:int=0, runid:int=-1):
    '''analysis ds'''
    return crbbot.Agent(prefix, ps_hint, runid)

def task (prefix:str, ps_hint:int=0, runid:int=-1, target:str='__none__'):
    '''task ds'''
    return crbbot.Actor(prefix, ps_hint, runid, target)
