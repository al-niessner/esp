# -- IMPORTS -- ------------------------------------------------------
import pdb

import excalibur.system.core as syscore

import numpy as np
import lmfit as lm
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
    spectra = np.array(cal['data']['SPECTRUM'])
    wave = np.array(cal['data']['WAVE'])
    time = np.array(cal['data']['TIME'])
    vrange = cal['data']['VRANGE']
    if len(tme[selftype]) > 0:
        for p in tme['data'].keys():
            out['data'][p] = {}
            rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
            ignore = tme['data'][p]['ignore']
            orbits = tme['data'][p]['orbits']
            dvisits = tme['data'][p]['dvisits']
            visits = tme['data'][p]['visits']
            z = tme['data'][p]['z']
            out['data'][p]['vrange'] = vrange
            out['data'][p]['visits'] = []
            out['data'][p]['nspec'] = []
            out['data'][p]['wave'] = []
            out['data'][p]['time'] = []
            for v in tme[selftype]:
                selv = (dvisits == v) & (~ignore)
                selvoot = selv & (abs(z) > (1e0 + rpors))
                if np.sum(selv) > 0:
                    out['data'][p]['visits'].append(v)
                    scores = [np.nansum(s)
                              for s,sv in zip(spectra, selvoot)
                              if sv == True]
                    thrit = np.nanpercentile(scores, 75,
                                             interpolation='higher')
                    it = scores.index(thrit)
                    template = spectra[selvoot][it]
                    nspectra = [s/template for s in spectra[selv]]
                    nspectra = np.array(nspectra)
                    out['data'][p]['nspec'].append(nspectra)
                    out['data'][p]['wave'].append(wave[selv])
                    out['data'][p]['time'].append(time[selv])
                    if verbose:
                        plt.figure()
                        for w,s in zip(wave[selv], nspectra):
                            select = ((w > np.min(vrange)) &
                                      (w < np.max(vrange)))
                            plt.plot(w[select], s[select], 'o')
                            pass
                        plt.ylabel('Normalized Spectra')
                        plt.xlabel('Wavelength [$\mu$m]')
                        plt.xlim(np.min(vrange), np.max(vrange))
                        plt.show()
                        pass
                    pass
                pass
            if len(out['data'][p]['visits']) > 0:
                normed = True
                out['STATUS'].append(True)
                pass
            pass
        pass
    return normed
# ------------------- ------------------------------------------------
