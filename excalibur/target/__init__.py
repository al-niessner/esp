ignore = False

def all (prefix, ps_hint=0, runid=-1):
    import exo.spec.ae.target.bot as trgbot
    return trgbot.Actor(prefix, ps_hint, runid)

def factory (prefix, ps_hint=0, runid=-1, target='__none__'):
    import exo.spec.ae.target.bot as trgbot
    return trgbot.Agent(prefix, ps_hint, runid, target)
