ignore = False

def factory (prefix, ps_hint=0, runid=-1, target='__none__'):
    import exo.spec.ae.transit.bot as trnbot
    return trnbot.Actor(prefix, ps_hint, runid, target)

