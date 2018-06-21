ignore = True

def factory(prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.cerberus.bot
    return excalibur.cerberus.bot.Actor(prefix, ps_hint, runid, target)
