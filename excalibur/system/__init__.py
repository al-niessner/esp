'''
SYSTEM manages the astrophysical parameters of the target observed
- VALIDATE parameters from target.autofill, report missing parameters
- FINALIZE parameters using target/edit.py function ppar()
'''
ignore = False

def factory (prefix, ps_hint=0, runid=-1, target='__none__'):
    import excalibur.system.bot as sysbot
    return sysbot.Actor(prefix, ps_hint, runid, target)
