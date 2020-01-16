ignore = False

def task(prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.classifier.bot as clsbot
    return clsbot.Actor(prefix, ps_hint, runid, target)
