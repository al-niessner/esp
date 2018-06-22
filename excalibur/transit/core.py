# -- IMPORTS -- ------------------------------------------------------
import pdb

import exo.spec.ae.system.core as syscore

import numpy as np
import matplotlib.pyplot as plt
# ------------- ------------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- NORMALIZATION -- ------------------------------------------------
def norm(cal, tme, fin, out, selftype, verbose=False, debug=False):
    normed = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    spectra = cal['data']['SPECTRUM']
    wave = cal['data']['WAVE']
    vrange = cal['data']['VRANGE']
    if len(tme[selftype]) > 0:
        for p in tme['data'].keys():
            rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
            ignore = tme['data'][p]['ignore']
            orbits = tme['data'][p]['orbits']
            dvisits = tme['data'][p]['dvisits']
            visits = tme['data'][p]['visits']
            z = tme['data'][p]['z']
            for v in tme[selftype]:
                selv = (dvisits == v) & (~ignore)
                selvoot = selv & (abs(z) > 1e0 + rpors)
                vspec = [s for s,sv in zip(spectra, selv)
                         if sv == True]
                vwave = [w for w,sv in zip(wave, selv)
                         if sv == True]
                if len(vwave) > 0:                    
                    pass
                pass
            pass
        pass
    return normed
# ------------------- ------------------------------------------------
