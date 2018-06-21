ignore = True

def factory(prefix, ps_hint=0, runid=-1, target='__none__'):
    import exo.spec.ae.cerberus.bot
    return exo.spec.ae.cerberus.bot.Actor(prefix, ps_hint, runid, target)
