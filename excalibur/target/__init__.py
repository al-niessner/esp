ignore = False

# pylint: disable=redefined-builtin
def analysis (prefix, ps_hint=0, runid=-1):
    import excalibur.target.bot as trgbot
    return trgbot.Actor(prefix, ps_hint, runid)

def regress (prefix, ps_hint=0, target='__none__'):
    import excalibur.target.bot as trgbot
    return trgbot.Regress(prefix, ps_hint, target)

def task (prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.target.bot as trgbot
    return trgbot.Agent(prefix, ps_hint, runid, target)
