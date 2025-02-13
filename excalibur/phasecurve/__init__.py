'''phasecurve __init__ ds'''

# -- IMPORTS -- ------------------------------------------------------
import excalibur.phasecurve.bot as phcbot

# ------------- ------------------------------------------------------
ignore = False


def task(
    prefix: str, ps_hint: int = 0, runid: int = -1, target: str = '__none__'
):
    '''task ds'''
    return phcbot.Actor(prefix, ps_hint, runid, target)
