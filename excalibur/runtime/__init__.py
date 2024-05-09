'''turn runtime operational parameters on disk into a state vector'''
ignore = False

# -- IMPORTS -- ------------------------------------------------------
import excalibur.runtime.bot
# ------------- ------------------------------------------------------

def analysis (prefix:str, ps_hint:int=0, runid:int=-1):
    '''configurations are global or an aspect'''
    return excalibur.runtime.bot.AnalysisTeam(prefix, ps_hint, runid)

def task (prefix:str, ps_hint:int=0, runid:int=-1, target:str='__none__'):
    '''task for target specific parts'''
    return excalibur.runtime.bot.TaskTeam(prefix, ps_hint, runid, target)
