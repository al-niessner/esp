ignore = False

def factory (prefix, ps_hint=0, runid=-1, target='__none__'):
    import exo.spec.ae.eclipse.bot as eclbot
    return eclbot.Actor(prefix, ps_hint, runid, target)

