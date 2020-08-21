ignore = False

def analysis (prefix, ps_hint=0, runid=-1):
    import excalibur.transit.bot as trnbot
    return trnbot.Agent(prefix, ps_hint, runid)

def task (prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.transit.bot as trnbot
    return trnbot.Actor(prefix, ps_hint, runid, target)
