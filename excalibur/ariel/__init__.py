'''
ARIEL simulates Ariel observations
'''

# -- IMPORTS -- ------------------------------------------------------
import excalibur.ariel.bot as arielbot

# ------------- ------------------------------------------------------

DAWGIE_IGNORE = False


def task(
    prefix: str, ps_hint: int = 0, runid: int = -1, target: str = '__none__'
):
    '''Factory'''
    return arielbot.Actor(prefix, ps_hint, runid, target)
