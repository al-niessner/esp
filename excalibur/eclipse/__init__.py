'''eclipse __init__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import excalibur.eclipse.bot as eclbot

# ------------- ------------------------------------------------------
DAWGIE_IGNORE = False


def task(
    prefix: str, ps_hint: int = 0, runid: int = -1, target: str = '__none__'
):
    '''task ds'''
    return eclbot.Actor(prefix, ps_hint, runid, target)
