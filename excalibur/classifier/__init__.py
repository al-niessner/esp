'''classifier __init__ ds'''

ignore = False

import excalibur.classifier.bot as clsbot

def analysis (prefix:str, ps_hint:int=0, runid:int=-1):
    '''analysis (aspect) ds'''
    return clsbot.Agent(prefix, ps_hint, runid)

def task(prefix:str, ps_hint:int=0, runid:int=-1, target:str='__none__'):
    '''task ds'''
    return clsbot.Actor(prefix, ps_hint, runid, target)
