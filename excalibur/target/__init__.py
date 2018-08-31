ignore = False

# pylint: disable=redefined-builtin
def all (prefix, ps_hint=0, runid=-1):
    import excalibur.target.bot as trgbot
    return trgbot.Actor(prefix, ps_hint, runid)

def factory (prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.target.bot as trgbot
    return trgbot.Agent(prefix, ps_hint, runid, target)
