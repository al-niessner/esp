ignore = False

def analysis (prefix, ps_hint=0, runid=-1):
    import excalibur.ancillary.bot as ancbot
    return ancbot.Actor(prefix, ps_hint, runid)

def task (prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.ancillary.bot as ancbot
    return ancbot.Agent(prefix, ps_hint, runid, target)
