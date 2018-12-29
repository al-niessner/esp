ignore = True

def task (prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.cerberus.bot as crbbot
    return crbbot.Actor(prefix, ps_hint, runid, target)
