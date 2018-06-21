ignore = False

def factory (prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.data.bot as datbot
    return datbot.Actor(prefix, ps_hint, runid, target)
