'''data __init__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import excalibur.data.bot as datbot

# ------------- ------------------------------------------------------

DAWGIE_IGNORE = False


def task(
    prefix: str, ps_hint: int = 0, runid: int = -1, target: str = '__none__'
):
    '''task ds'''
    return datbot.Actor(prefix, ps_hint, runid, target)
