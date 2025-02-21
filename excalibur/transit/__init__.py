'''transmission spectrum task'''

# -- IMPORTS -- ------------------------------------------------------
import excalibur.transit.bot as trnbot

# ------------- ------------------------------------------------------
DAWGIE_IGNORE = False


def analysis(prefix: str, ps_hint: int = 0, runid: int = -1):
    '''analysis (aspect) ds'''
    return trnbot.Agent(prefix, ps_hint, runid)


def task(
    prefix: str, ps_hint: int = 0, runid: int = -1, target: str = '__none__'
):
    '''task ds'''
    return trnbot.Actor(prefix, ps_hint, runid, target)
