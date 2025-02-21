'''util checksv ds'''

# -- IMPORTS -- ------------------------------------------------------


# -------------- -----------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    '''Checks for empty SV'''
    if sv['STATUS'][-1]:
        valid = True
        errstring = None
    else:
        valid = False
        errstring = sv.name() + ' IS EMPTY'
    return valid, errstring
