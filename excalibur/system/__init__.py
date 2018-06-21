ignore = False

def factory (prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.system.bot as sysbot
    return sysbot.Actor(prefix, ps_hint, runid, target)
