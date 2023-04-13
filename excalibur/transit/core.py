'''transit core ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

# pylint: disable=import-self
import excalibur.data.core as datcore
import excalibur.system.core as syscore
import excalibur.util.cerberus as crbutil
import excalibur.transit.core

import re
import io
import copy
import requests
import logging
import random
import lmfit as lm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator
from matplotlib.ticker import ScalarFormatter

# pylint: disable=import-error
from ultranest import ReactiveNestedSampler

import pymc3 as pm
log = logging.getLogger(__name__)
pymc3log = logging.getLogger('pymc3')
pymc3log.setLevel(logging.ERROR)

from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares, brentq, curve_fit
import scipy.constants as cst
from scipy import spatial
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

import theano.tensor as tt
import theano.compile.ops as tco

import numpy as np

try:
    import astropy.constants
    import astropy.units
    from astropy.modeling.models import BlackBody
except ImportError:
    from astropy.modeling.blackbody import blackbody_lambda as BlackBody


from collections import namedtuple
CONTEXT = namedtuple('CONTEXT', ['alt', 'ald', 'allz', 'orbp', 'commonoim', 'ecc',
                                 'g1', 'g2', 'g3', 'g4', 'ootoindex', 'ootorbits',
                                 'orbits', 'period', 'selectfit', 'smaors', 'time',
                                 'tmjd', 'ttv', 'valid', 'visits', 'aos', 'avi',
                                 'ginc', 'gttv'])
ctxt = CONTEXT(alt=None, ald=None, allz=None, orbp=None, commonoim=None, ecc=None,
               g1=None, g2=None, g3=None, g4=None, ootoindex=None, ootorbits=None,
               orbits=None, period=None, selectfit=None, smaors=None, time=None,
               tmjd=None, ttv=None, valid=None, visits=None, aos=None, avi=None,
               ginc=None, gttv=None)
def ctxtupdt(alt=None, ald=None, allz=None, orbp=None, commonoim=None, ecc=None,
             g1=None, g2=None, g3=None, g4=None, ootoindex=None, ootorbits=None,
             orbits=None, period=None, selectfit=None, smaors=None, time=None,
             tmjd=None, ttv=None, valid=None, visits=None, aos=None, avi=None,
             ginc=None, gttv=None):
    '''
G. ROUDIER: Update global context for pymc3 deterministics
    '''
    excalibur.transit.core.ctxt = CONTEXT(alt=alt, ald=ald, allz=allz, orbp=orbp,
                                          commonoim=commonoim, ecc=ecc, g1=g1, g2=g2,
                                          g3=g3, g4=g4, ootoindex=ootoindex,
                                          ootorbits=ootorbits, orbits=orbits,
                                          period=period, selectfit=selectfit,
                                          smaors=smaors, time=time, tmjd=tmjd, ttv=ttv,
                                          valid=valid, visits=visits, aos=aos, avi=avi,
                                          ginc=ginc, gttv=gttv)
    return

import ldtk
from ldtk import LDPSetCreator, BoxcarFilter
from ldtk.ldmodel import LinearModel, QuadraticModel, NonlinearModel
class LDPSet(ldtk.LDPSet):
    '''
    A. NIESSNER: INLINE HACK TO ldtk.LDPSet
    '''
    @staticmethod
    def is_mime():
        '''is_mime ds'''
        return True

    @property
    def profile_mu(self):
        '''profile_mu ds'''
        return self._mu
    pass
setattr(ldtk, 'LDPSet', LDPSet)
setattr(ldtk.ldtk, 'LDPSet', LDPSet)
# ------------- ------------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    '''
    G. ROUDIER: Tests for empty SV shell
    '''
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- NORMALIZATION -- ------------------------------------------------
def normversion():
    '''
    1.1.5: add hstbreath parameters to SV
    1.1.6: add timing parameter start model
    1.1.7: new sigma clip for spitzer
    1.1.8: added jwst filter
    '''
    return dawgie.VERSION(1,1,9)

def norm(cal, tme, fin, ext, out, selftype, verbose=False, debug=False):
    '''
    G. ROUDIER: Out of transit data normalization
    '''
    normed = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    spectra = cal['data']['SPECTRUM']
    wave = cal['data']['WAVE']
    time = np.array(cal['data']['TIME'])
    disp = np.array(cal['data']['DISPERSION'])
    scanlen = np.array(cal['data']['SCANLENGTH'])
    vrange = cal['data']['VRANGE']
    arcsec2pix = datcore.dps(ext)
    scanlen = np.floor(scanlen/arcsec2pix)
    events = [pnet for pnet in tme['data'].keys()
              if (pnet in priors.keys()) and tme['data'][pnet][selftype]]
    for p in events:
        log.warning('>-- Planet: %s', p)
        out['data'][p] = {}
        rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
        mttref = priors[p]['t0']
        if mttref > 2400000.5: mttref -= 2400000.5
        ignore = np.array(tme['data'][p]['ignore']) | np.array(cal['data']['IGNORED'])
        orbits = tme['data'][p]['orbits']
        dvisits = tme['data'][p]['dvisits']
        visits = tme['data'][p]['visits']
        phase = tme['data'][p]['phase']
        z = tme['data'][p]['z']
        zoot = z.copy()
        out['data'][p]['vrange'] = vrange
        out['data'][p]['visits'] = []
        out['data'][p]['dvisnum'] = []
        out['data'][p]['nspec'] = []
        out['data'][p]['wave'] = []
        out['data'][p]['time'] = []
        out['data'][p]['orbits'] = []
        out['data'][p]['dispersion'] = []
        out['data'][p]['z'] = []
        out['data'][p]['phase'] = []
        out['data'][p]['wavet'] = []
        out['data'][p]['photnoise'] = []
        out['data'][p]['trial'] = []
        out['data'][p]['vignore'] = []
        out['data'][p]['stdns'] = []
        out['data'][p]['hstbreath'] = []
        singlevisit = False
        svnkey = 'svn'+selftype
        if tme['data'][p][svnkey].__len__() == 1:
            singlevisit = True
            log.warning('--< Single Visit Observation')
            pass
        for v in tme['data'][p][svnkey]:  # SINGLE SCAN NUMBERING
            selv = (visits == v) & ~ignore
            if selftype in ['transit', 'phasecurve']:
                select = (phase[selv] > 0.25) | (phase[selv] < -0.25)
                vzoot = zoot[selv]
                vzoot[select] = np.nan
                zoot[selv] = vzoot
                pass
            if selftype in ['eclipse']:
                vzoot = zoot[selv]
                select = (phase[selv] < 0.25) & (phase[selv] > -0.25)
                vzoot[select] = np.nan
                zoot[selv] = vzoot
                pass
            selv = selv & np.isfinite(zoot)
            if True in selv: firstorb = int(np.min(orbits[selv]))
            else: firstorb = int(1)
            # ORBIT SELECTION FOR HST BREATHING MODEL ------------------------------------
            ootplus = []
            ootpv = []
            ootminus = []
            ootmv = []
            inorb = []
            for o in set(orbits[selv]):
                zorb = zoot[selv][orbits[selv] == o]
                medzorb = np.nanmedian(zorb)
                if (medzorb > 0) and (np.nanmin(zorb) > (1e0 + rpors)):
                    ootplus.append(o)
                    ootpv.append(medzorb)
                    pass
                elif medzorb < -(1e0 + rpors):
                    ootminus.append(o)
                    ootmv.append(medzorb)
                    pass
                if np.any(abs(zorb) < (1e0 + rpors)): inorb.append(int(o))
                pass
            ootplus = np.array(ootplus)
            ootpv = np.array(ootpv)
            selord = np.argsort(abs(ootplus - np.mean(inorb)))
            ootplus = ootplus[selord]
            ootpv = ootpv[selord]
            ootminus = np.array(ootminus)
            ootmv = np.array(ootmv)
            selord = np.argsort(abs(ootminus - np.mean(inorb)))
            ootminus = ootminus[selord]
            ootmv = ootmv[selord]
            trash = []
            pureoot = []
            keep = None
            for thisorb in ootplus:
                ckeep = keep is not None
                badcond = np.sum(orbits[selv] == thisorb) < 7
                if ckeep or badcond: trash.append(int(thisorb))
                else:
                    keep = thisorb
                    pureoot.append(thisorb)
                    pass
                pass
            keep = None
            for thisorb in ootminus:
                ckeep = keep is not None
                zorb = zoot[selv][orbits[selv] == thisorb]
                badcond = np.sum(zorb < -(1e0 + rpors)) < 7
                if thisorb not in inorb:
                    if ckeep or badcond or (thisorb in [firstorb]):
                        trash.append(int(thisorb))
                        pass
                    elif (thisorb not in inorb) and (thisorb not in [firstorb]):
                        keep = thisorb
                        pureoot.append(thisorb)
                        pass
                    pass
                pass
            # COMPENSATE UNBALANCED OOT DATA ---------------------------------------------
            innout = [int(o) for o in set(orbits[selv]) if o not in trash]
            pickmeup = [int(o) for o in trash if o in ootminus]
            if (np.sum(zoot[selv] > (1e0 + rpors)) < 3) and pickmeup:
                dist = list(abs(np.array(pickmeup) - np.mean(innout)))
                if pickmeup[dist.index(min(dist))] not in [firstorb]:
                    trash.pop(trash.index(pickmeup[dist.index(min(dist))]))
                    log.warning('--< Missing OOT+ data, adding orbit: %s',
                                str(int(pickmeup[dist.index(min(dist))])))
                    pass
                pass
            pickmeup = [int(o) for o in trash if o in ootplus]
            if (np.sum(zoot[selv] < -(1e0 + rpors)) < 3) and pickmeup:
                dist = list(abs(np.array(pickmeup) - np.mean(innout)))
                if pickmeup[dist.index(min(dist))] not in [firstorb]:
                    trash.pop(trash.index(pickmeup[dist.index(min(dist))]))
                    log.warning('--< Missing OOT- data, adding orbit: %s',
                                str(int(pickmeup[dist.index(min(dist))])))
                    pass
                pass
            log.warning('>-- Visit %s', str(int(v)))
            log.warning('>-- Orbit %s', str([int(o) for o in set(orbits[selv])]))
            log.warning('>-- Trash %s', str(trash))
            # UPDATE IGNORE FLAG WITH REJECTED ORBITS ------------------------------------
            if trash and (selftype in ['transit', 'eclipse']):
                for o in trash:
                    select = orbits[selv] == o
                    vignore = ignore[selv]
                    vignore[select] = True
                    ignore[selv] = vignore
                    pass
                pass
            # VISIT SELECTION ------------------------------------------------------------
            selv = selv & (~ignore)
            viss = list(np.array(spectra)[selv])
            visw = np.array(wave)[selv]
            cwave, _t = tplbuild(viss, visw, vrange, disp[selv]*1e-4, medest=True)
            # OUT OF TRANSIT ORBITS SELECTION --------------------------------------------
            selvoot = selv & np.array([(test in pureoot) for test in orbits])
            selvoot = selvoot & (abs(zoot) > (1e0 + rpors))
            voots = list(np.array(spectra)[selvoot])
            vootw = list(np.array(wave)[selvoot])
            ivoots = []
            for s, w in zip(voots, vootw):
                itps = np.interp(np.array(cwave), w, s, left=np.nan, right=np.nan)
                ivoots.append(itps)
                pass
            pureootext = pureoot.copy()
            if firstorb in orbits[selv]:
                fovoots = list(np.array(spectra)[selv])
                fovootw = list(np.array(wave)[selv])
                pureootext.append(firstorb)
                foivoots = []
                for s, w in zip(fovoots, fovootw):
                    itps = np.interp(np.array(cwave), w, s, left=np.nan, right=np.nan)
                    foivoots.append(itps)
                    pass
                pass
            if True in selvoot:
                # BREATHING CORRECTION ---------------------------------------------------
                hstbreath = {}
                for thisorb in pureootext:
                    alltfo = []
                    alldfo = []
                    allito = []
                    allslo = []
                    if thisorb in [firstorb]:
                        selorb = (orbits[selv] == thisorb)
                        fotime = time[selv][selorb]
                        zorb = zoot[selv][selorb]
                        specin = [s for s, ok in zip(foivoots, selorb) if ok]
                        wavein = [cwave]*len(specin)
                        _t, fos = tplbuild(specin, wavein, vrange,
                                           (disp[selv][selorb])*1e-4, superres=True)
                        pass
                    else:
                        selorb = (orbits[selvoot] == thisorb)
                        fotime = time[selvoot][selorb]
                        zorb = zoot[selvoot][selorb]
                        specin = [s for s, ok in zip(ivoots, selorb) if ok]
                        wavein = [cwave]*len(specin)
                        _t, fos = tplbuild(specin, wavein, vrange,
                                           (disp[selvoot][selorb])*1e-4, superres=True)
                        pass
                    for eachfos in fos:
                        mintscale = np.min(abs(np.diff(np.sort(fotime))))
                        maxtscale = np.max(fotime) - np.min(fotime)
                        minoscale = np.log10(mintscale*36e2*24)
                        maxoscale = np.log10(maxtscale*36e2*24)
                        maxdelay = np.log10(5e0*maxtscale*36e2*24)
                        maxrfos = np.nanmax(fos)/np.nanmedian(fos)
                        minrfos = np.nanmin(fos)/np.nanmedian(fos)
                        params = lm.Parameters()
                        params.add('oitcp', value=1e0, min=minrfos, max=maxrfos)
                        params.add('oslope', value=1e-4,
                                   min=(minrfos - maxrfos)/maxtscale,
                                   max=(maxrfos - minrfos)/maxtscale)
                        if 'HST' in ext:  # empirically fit start point model
                            coef = 0.3494939103124791
                            ologtau_start = maxoscale*coef + minoscale*(1-coef)
                        else:
                            ologtau_start = np.mean([minoscale, maxoscale])
                        params.add('ologtau', value=ologtau_start,
                                   min=minoscale, max=maxoscale)
                        if 'HST' in ext:  # empirically fit start point model
                            coef = 0.7272498969170312
                            ologdelay_start = maxdelay*coef + minoscale*(1-coef)
                        else:
                            ologdelay_start = np.mean([minoscale, maxdelay])
                        params.add('ologdelay', value=ologdelay_start,
                                   min=minoscale, max=maxdelay)
                        normcond = zorb > np.nanmedian(zorb)
                        if True in normcond:
                            normfordata = np.nanmedian(np.array(eachfos)[normcond])
                            ndata = np.array(eachfos)/normfordata
                            pass
                        else: ndata = np.array(eachfos)/np.nanmedian(eachfos)
                        if np.sum(np.isfinite(ndata)) > 4:
                            lmout = lm.minimize(hstramp, params,
                                                args=(fotime, ndata), method='cg')
                            alltfo.append(lmout.params['ologtau'].value)
                            alldfo.append(lmout.params['ologdelay'].value)
                            allito.append(lmout.params['oitcp'].value)
                            allslo.append(lmout.params['oslope'].value)
                            pass
                        if debug:
                            plt.figure()
                            plt.plot(fotime, ndata, 'o')
                            plt.plot(fotime, hstramp(lmout.params, fotime), '*')
                            plt.show()
                            pass
                        pass
                    params = lm.Parameters()
                    params.add('oitcp', value=np.nanmedian(allito))
                    params.add('oslope', value=np.nanmedian(allslo))
                    params.add('ologtau', value=np.nanmedian(alltfo))
                    params.add('ologdelay', value=np.nanmedian(alldfo))
                    if debug:
                        plt.figure()
                        plt.title('Breathing ramp orbit: ' + str(int(thisorb)))
                        plt.plot(fotime, hstramp(params, fotime), 'o')
                        plt.show()
                        pass
                    hstbreath[str(int(thisorb))] = params
                    if False in np.isfinite([np.nanmedian(alltfo), np.nanmedian(alldfo)]):
                        log.warning('--< No ramp for Visit :%s Orbit: %s',
                                    str(int(v)), str(int(thisorb)))
                        pass
                    pass
                viss = np.array(viss)
                for spec in viss.T:
                    for orb in set(orbits[selv]):
                        selorb = orbits[selv] == orb
                        if (orb in pureoot) or (orb in [firstorb]):
                            model = hstramp(hstbreath[str(int(orb))], time[selv][selorb])
                            pass
                        elif hstbreath:
                            choice = [int(key) for key in hstbreath]
                            if (firstorb in choice) and (choice.__len__() > 1):
                                choice.pop(choice.index(firstorb))
                                pass
                            diff = list(abs(np.array(choice) - orb))
                            closest = diff.index(min(diff))
                            model = hstramp(hstbreath[str(choice[closest])],
                                            time[selv][selorb])
                            pass
                        else: model = np.array([1e0]*np.sum(selorb))
                        if np.all(np.isfinite(model)): spec[selorb] /= model
                        pass
                    pass
                # PHOTON NOISE ESTIMATE --------------------------------------------------
                photnoise = np.sqrt(abs(viss.copy()))
                for phnoise in photnoise.T: phnoise /= np.sqrt(scanlen[selv])
                # DOUBLE SCAN CORRECTION -------------------------------------------------
                vlincorr = visits[selv]
                if set(dvisits[selv]).__len__() > 1: vlincorr = dvisits[selv]
                ordsetds = np.sort(list(set(vlincorr))).astype(int)
                ootinv = abs(zoot[selv]) > (1e0 + rpors)
                for dsscan in ordsetds:
                    # WAVELENGTH INTERPOLATION -------------------------------------------
                    thisscan = (vlincorr == dsscan) & ootinv
                    fos = []
                    wfos = []
                    allvissts = viss[thisscan].flatten()
                    allviswts = visw[thisscan].flatten()
                    dcwave = [np.diff(cwave)[0]]
                    dcwave.extend(np.diff(cwave))
                    for cw, dcw in zip(cwave, dcwave):
                        select = (allviswts > (cw - dcw/2e0))
                        select = select & (allviswts < (cw + dcw/2e0))
                        fos.append(allvissts[select])
                        wfos.append(allviswts[select])
                        pass
                    witp = {}
                    for eachwfos, eachfos, cw in zip(wfos, fos, cwave):
                        eachwfos = np.array(eachwfos)
                        eachfos = np.array(eachfos)
                        finval = np.isfinite(eachfos)
                        key = cwave.index(cw)
                        if 'G430' in ext: polyorder = 1
                        else: polyorder = 3
                        if np.sum(finval) > (polyorder + 1):
                            poly = np.poly1d(np.polyfit(eachwfos[finval], eachfos[finval],
                                                        polyorder))
                            witp[str(key)] = poly
                            if debug:
                                plt.figure()
                                plt.plot(eachwfos[finval], eachfos[finval], '+')
                                plt.plot(eachwfos[finval], poly(eachwfos[finval]), 'o')
                                plt.show()
                                pass
                            pass
                        pass
                    thisscan = vlincorr == dsscan
                    vtsdt = []
                    ptsdt = []
                    for spec, wspec, phnoise in zip(viss[thisscan], visw[thisscan],
                                                    photnoise[thisscan]):
                        wavecorr = []
                        for w in wspec:
                            if (w < np.min(vrange)) or (w > np.max(vrange)):
                                wavecorr.append(np.nan)
                                pass
                            else:
                                diff = list(abs(np.array(cwave) - w))
                                closest = diff.index(min(diff))
                                if str(closest) in witp:
                                    wavecorr.append(witp[str(closest)](w))
                                    pass
                                else: wavecorr.append(np.nan)
                                pass
                            pass
                        wavecorr = np.array(wavecorr)
                        vtsdt.append(spec/wavecorr)
                        ptsdt.append(phnoise/wavecorr)
                        pass
                    viss[thisscan] = np.array(vtsdt)
                    photnoise[thisscan] = np.array(ptsdt)
                    pass
                if debug:
                    plt.figure()
                    plt.plot(visw.T, viss.T, 'o')
                    plt.show()
                    pass
                # DATA CONSISTENCY AND QUALITY CHECK -------------------------------------
                wanted = []
                for w, s, pn, zv in zip(visw, viss, photnoise, zoot[selv]):
                    wselect = (w >= np.min(vrange)) & (w <= np.max(vrange))
                    if (True in wselect) and (abs(zv) > (1e0 + rpors)):
                        wanted.append(np.nanstd(s[wselect])/np.nanmedian(pn))
                        pass
                    pass
                nscale = np.round(np.percentile(wanted, 50))
                log.warning('--< Visit %s: Noise scale %s', str(int(v)), str(nscale))
                # FLAGGING THRESHOLD
                noisescalethr = 5e0
                if nscale <= noisescalethr: nscale = noisescalethr
                elif (nscale > noisescalethr) and (not singlevisit):
                    nscale = 2e0
                    log.warning('--< Visit %s: Noise scale above %s',
                                str(int(v)), str(int(noisescalethr)))
                    pass
                check = []
                for w, s, pn in zip(visw, viss, photnoise):
                    wselect = (w >= np.min(vrange)) & (w <= np.max(vrange))
                    if True in wselect:
                        crit = np.nanstd(s[wselect]) < (nscale*np.nanmedian(pn))
                        check.append(crit)
                        pass
                    else: check.append(False)
                    pass
                check = np.array(check)
                rejrate = check.size - np.sum(check)
                log.warning('--< Visit %s: Rejected %s/%s',
                            str(int(v)), str(rejrate), str(check.size))
                dataoot = viss[(abs(z[selv]) > (1e0 + rpors)) & check]
                waveoot = visw[(abs(z[selv]) > (1e0 + rpors)) & check]
                stdoot = []
                for s, w in zip(dataoot, waveoot):
                    wselect = (w >= np.min(vrange)) & (w <= np.max(vrange))
                    if True in wselect: stdoot.append(np.nanstd(s[wselect]))
                    pass
                thrz = np.nanmedian(stdoot) + 3e0*np.nanstd(stdoot)
                checkz = []
                for w, s, zv in zip(visw, viss, zoot[selv]):
                    wselect = (w >= np.min(vrange)) & (w <= np.max(vrange))
                    if True in wselect:
                        if abs(zv) > (1e0 + rpors):
                            critz = abs(1e0 - np.nanmean(s[wselect])) < thrz
                            pass
                        else: critz = np.nanmean(s[wselect]) - 1e0 < thrz
                        checkz.append(critz)
                        if not critz:
                            log.warning('--< Visit %s: Excluding averaged spectrum %s',
                                        str(int(v)), str(np.nanmean(s[wselect])))
                            pass
                        pass
                    else: checkz.append(True)
                    pass
                check = check & np.array(checkz)  # comment if G430L
                wellcondin = np.sum(abs(zoot[selv][check]) < (1e0 + rpors)) > 3
                if not((np.sum(check) > 9) and wellcondin) and singlevisit:
                    wellcondin = True
                    check = check | True
                    log.warning('--< Visit %s: %s', str(int(v)), 'Single visit exception')
                    pass
                if (np.sum(check) > 9) and wellcondin:
                    vnspec = np.array(viss)[check]
                    nphotn = np.array(photnoise)[check]
                    visw = np.array(visw)
                    eclphase = phase[selv][check]
                    if selftype in ['eclipse']:
                        eclphase[eclphase < 0] = eclphase[eclphase < 0] + 1e0
                        pass
                    out['data'][p]['visits'].append(int(v))
                    out['data'][p]['dvisnum'].append(set(visits[selv]))
                    out['data'][p]['nspec'].append(vnspec)
                    out['data'][p]['wavet'].append(cwave)
                    out['data'][p]['wave'].append(visw[check])
                    out['data'][p]['time'].append(time[selv][check])
                    out['data'][p]['orbits'].append(orbits[selv][check])
                    out['data'][p]['dispersion'].append(disp[selv][check])
                    out['data'][p]['z'].append(z[selv][check])
                    out['data'][p]['phase'].append(eclphase)
                    out['data'][p]['photnoise'].append(nphotn)
                    out['data'][p]['stdns'].append(np.nanstd(np.nanmedian(viss, axis=1)))
                    out['data'][p]['hstbreath'].append(hstbreath)
                    if verbose:
                        plt.figure()
                        for w,s in zip(visw[check], vnspec):
                            select = (w > np.min(vrange)) & (w < np.max(vrange))
                            plt.plot(w[select], s[select], 'o')
                            pass
                        plt.ylabel('Normalized Spectra')
                        plt.xlabel('Wavelength [$\\mu$m]')
                        plt.xlim(np.min(vrange), np.max(vrange))
                        plt.show()
                        pass
                    pass
                else:
                    if wellcondin:
                        out['data'][p]['trial'].append('Spectral Variance Excess')
                        pass
                    else: out['data'][p]['trial'].append('Missing IT Data')
                    out['data'][p]['vignore'].append(v)
                    if (len(tme[selftype]) - len(out['data'][p]['vignore'])) < 2:
                        singlevisit = True
                        pass
                    pass
                pass
            else:
                out['data'][p]['trial'].append('Not Enough OOT Data')
                out['data'][p]['vignore'].append(v)
                if (len(tme[selftype]) - len(out['data'][p]['vignore'])) < 2:
                    singlevisit = True
                    pass
                pass
            pass
        # VARIANCE EXCESS FROM LOST GUIDANCE ---------------------------------------------
        kickout = []
        if out['data'][p]['stdns'].__len__() > 2:
            stdns = np.array(out['data'][p]['stdns'])
            vthr = np.nanpercentile(stdns, 66, interpolation='nearest')
            ref = np.nanmean(stdns[stdns <= vthr])
            vesel = abs((stdns/ref - 1e0)*1e2) > 5e1
            kickout = list(np.array(out['data'][p]['visits'])[vesel])
            pass
        # TEST FOR SEPARATE VISITS -------------------------------------------------------
        # GJ 1132
        # kickout = [1, 2, 3, 4]
        if kickout:
            for v in kickout:
                i2pop = out['data'][p]['visits'].index(v)
                out['data'][p]['visits'].pop(i2pop)
                out['data'][p]['dvisnum'].pop(i2pop)
                out['data'][p]['wavet'].pop(i2pop)
                out['data'][p]['nspec'].pop(i2pop)
                out['data'][p]['wave'].pop(i2pop)
                out['data'][p]['time'].pop(i2pop)
                out['data'][p]['orbits'].pop(i2pop)
                out['data'][p]['dispersion'].pop(i2pop)
                out['data'][p]['z'].pop(i2pop)
                out['data'][p]['phase'].pop(i2pop)
                out['data'][p]['photnoise'].pop(i2pop)
                out['data'][p]['trial'].append('Lost Guidance Variance Excess')
                out['data'][p]['vignore'].append(v)
                pass
            pass
        for v, m in zip(out['data'][p]['vignore'], out['data'][p]['trial']):
            log.warning('--< Visit %s: %s', str(int(v)), str(m))
            pass
        if out['data'][p]['visits']:
            normed = True
            out['STATUS'].append(True)
            pass
        pass
    return normed
# ------------------- ------------------------------------------------
# -- TEMPLATE BUILDER -- ---------------------------------------------
def tplbuild(spectra, wave, vrange, disp, superres=False, medest=False):
    '''
    G. ROUDIER: Builds a spectrum template according to the peak in population density
    per wavelength bins
    '''
    allspec = []
    for s in spectra.copy(): allspec.extend(s)
    allwave = []
    for w in wave.copy(): allwave.extend(w)
    allspec = np.array(allspec)
    allwave = np.array(allwave)
    vdisp = np.mean(disp)
    wavet = []
    template = []
    guess = [np.min(vrange) - vdisp/2e0]
    while guess[-1] < (max(vrange) + vdisp/2e0):
        dist = abs(allwave - guess[-1])
        selwave = list(allwave[dist < vdisp])
        selspec = list(allspec[dist < vdisp])
        seldist = list(dist[dist < vdisp])
        if seldist:
            if np.min(seldist) < (vdisp/2e0):
                cluster = [selwave[seldist.index(min(seldist))]]
                cloud = [selspec[seldist.index(min(seldist))]]
                selwave.pop(seldist.index(min(seldist)))
                selspec.pop(seldist.index(min(seldist)))
                while (cluster.__len__() < disp.size) and selwave:
                    seldist = abs(np.array(selwave) - np.median(cluster))
                    if np.min(seldist) < (vdisp/2e0):
                        seldist = list(seldist)
                        cluster.append(selwave[seldist.index(min(seldist))])
                        cloud.append(selspec[seldist.index(min(seldist))])
                        selwave.pop(seldist.index(min(seldist)))
                        selspec.pop(seldist.index(min(seldist)))
                        pass
                    else:
                        seldist = list(seldist)
                        selwave.pop(seldist.index(min(seldist)))
                        selspec.pop(seldist.index(min(seldist)))
                        pass
                    pass
                if superres:
                    wavet.append(cluster)
                    template.append(cloud)
                    pass
                elif True in np.isfinite(cloud):
                    if medest:
                        arrcluster = np.array(cluster)
                        arrcloud = np.array(cloud)
                        wavet.append(np.median(arrcluster[np.isfinite(arrcloud)]))
                        template.append(np.median(arrcloud[np.isfinite(arrcloud)]))
                        pass
                    else:
                        arrcluster = np.array(cluster)
                        arrcloud = np.array(cloud)
                        wavet.append(np.mean(arrcluster[np.isfinite(arrcloud)]))
                        template.append(np.mean(arrcloud[np.isfinite(arrcloud)]))
                        pass
                    pass
                finiteloop = np.median(cluster) + vdisp
                pass
            finiteloop = guess[-1] + vdisp
            pass
        else: finiteloop = guess[-1] + vdisp
        while finiteloop in guess: finiteloop += vdisp
        guess.append(finiteloop)
        pass
    return wavet, template
# ---------------------- ---------------------------------------------
# -- WHITE LIGHT CURVE -- --------------------------------------------
def wlversion():
    '''
    G. ROUDIER: 1.2.0 includes a multi instrument orbital solution
    K. PEARSON: 1.2.2 new eclipse model + priors from transit
              : 1.2.3 new priors
    N. HUBERFE: 1.2.4 new priors
    K. PEARSON: 1.2.5 nested sampling for spitzer
    K. PEARSON: 1.2.6 jwst support
    K. PEARSON: 1.2.7 C-optimized for spitzer
    S. KANTAMNENI: 1.3.0 added data simulation
    '''
    return dawgie.VERSION(1,3,0)

def hstwhitelight(allnrm, fin, out, allext, selftype, chainlen=int(1e4), verbose=False):
    '''
    S. KANTAMNENI: Created key for simulated whitelight data
    G. ROUDIER: Combined orbital parameters recovery
    '''
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = []
    for nrm in allnrm:
        planetloop.extend([p for p in nrm['data'].keys() if
                           (nrm['data'][p]['visits']) and (p not in planetloop)])
        pass
    for p in planetloop:
        rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
        maxvis = 0
        visits = []
        orbits = []
        time = []
        wave = []
        nspec = []
        sep = []
        phase = []
        photnoise = []
        allfltrs = []
        allvisits = []
        pnrmlist = [nrm for nrm in allnrm if p in nrm['data']]
        pextlist = [thisext for nrm, thisext in zip(allnrm, allext) if p in nrm['data']]
        ext = ''
        for thisext in pextlist:
            if ext: ext = ext + '+' + thisext
            else: ext = thisext
            pass
        for nrm, fltr in zip(pnrmlist, pextlist):
            visits.extend(np.array(nrm['data'][p]['visits']) + maxvis)
            orbits.extend(np.array(nrm['data'][p]['orbits']))
            time.extend(np.array(nrm['data'][p]['time']))
            wave.extend(np.array(nrm['data'][p]['wave']))
            nspec.extend(np.array(nrm['data'][p]['nspec']))
            sep.extend(np.array(nrm['data'][p]['z']))
            phase.extend(np.array(nrm['data'][p]['phase']))
            photnoise.extend(np.array(nrm['data'][p]['photnoise']))
            maxvis = maxvis + np.max(visits)
            allfltrs.extend([fltr]*len(nrm['data'][p]['visits']))
            allvisits.extend(nrm['data'][p]['visits'])
            pass
        nspec = np.array(nspec)
        wave = np.array(wave)
        out['data'][p] = {}
        out['data'][p]['nspec'] = nspec
        out['data'][p]['wave'] = wave
        out['data'][p]['visits'] = visits
        out['data'][p]['orbits'] = orbits
        allwhite = []
        allerrwhite = []
        flatminww = []
        flatmaxww = []
        for index, _v in enumerate(visits):
            white = []
            errwhite = []
            for w, s, e in zip(wave[index], nspec[index], photnoise[index]):
                select = np.isfinite(s)
                if True in select:
                    white.append(np.nanmean(s[select]))
                    errwhite.append(np.nanmedian(e[select])/np.sqrt(np.nansum(select)))
                    pass
                else:
                    white.append(np.nan)
                    errwhite.append(np.nan)
                    pass
                flatminww.append(min(w[select]))
                flatmaxww.append(max(w[select]))
                pass
            allwhite.append(white)
            allerrwhite.append(errwhite)
            pass
        flaterrwhite = []
        for r in allerrwhite: flaterrwhite.extend(r)
        flaterrwhite = np.array(flaterrwhite)
        flatwhite = []
        for w in allwhite: flatwhite.extend(w)
        flatwhite = np.array(flatwhite)
        flatz = []
        for z in sep: flatz.extend(z)
        flatz = np.array(flatz)
        flatphase = []
        for ph in phase: flatphase.extend(ph)
        flatphase = np.array(flatphase)
        allwwmin = min(flatminww)
        allwwmax = max(flatmaxww)
        renorm = np.nanmean(flatwhite[abs(flatz) > 1e0 + rpors])
        flatwhite /= renorm
        flaterrwhite /= renorm
        allwhite = [np.array(aw)/renorm for aw in allwhite]
        allerrwhite = [np.array(aew)/renorm for aew in allerrwhite]
        out['data'][p]['allwhite'] = allwhite
        out['data'][p]['phase'] = phase
        out['data'][p]['flatphase'] = flatphase
        # LIMB DARKENING ---------------------------------------------
        if selftype in ['transit']:
            whiteld = createldgrid([allwwmin], [allwwmax], priors,
                                   segmentation=int(10), verbose=verbose)
            g1, g2, g3, g4 = whiteld['LD']
            pass
        else: g1, g2, g3, g4 = [[0], [0], [0], [0]]
        wlmod = tldlc(abs(flatz), rpors, g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0])
        out['data'][p]['whiteld'] = [g1[0], g2[0], g3[0], g4[0]]
        # PLOT -------------------------------------------------------
        if verbose:
            plt.figure(figsize=(10, 6))
            for v in visits:
                index = visits.index(v)
                plt.plot(phase[index], allwhite[index], 'o', label=allfltrs[index])
                pass
            if visits.__len__() > 14: ncol = 2
            else: ncol = 1
            plt.plot(flatphase, wlmod, '^', label='M')
            plt.xlabel('Orbital Phase')
            plt.ylabel('Normalized Raw White Light Curve')
            plt.legend(loc='best', ncol=ncol, mode='expand', numpoints=1,
                       borderaxespad=0., frameon=False)
            plt.tight_layout()
            plt.show()
            pass
        # TTV --------------------------------------------------------
        ttv = []
        for index, v in enumerate(visits):
            select = (abs(sep[index]) < (1e0+rpors)) & (abs(sep[index]) > (1e0-rpors))
            if True in select: ttv.append(v)
            pass
        allttvs = []
        allttvfltrs = []
        for index, v in enumerate(allvisits):
            select = (abs(sep[index]) < (1e0+rpors)) & (abs(sep[index]) > (1e0-rpors))
            if True in select:
                allttvs.append(v)
                allttvfltrs.append(allfltrs[index])
                pass
            pass
        # PRIORS -----------------------------------------------------
        tmjd = priors[p]['t0']
        if tmjd > 2400000.5: tmjd -= 2400000.5
        period = priors[p]['period']
        ecc = priors[p]['ecc']
        inc = priors[p]['inc']
        smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
        ootstd = np.nanstd(flatwhite[abs(flatz) > 1e0 + rpors])
        taurprs = 1e0/(rpors*1e-2)**2
        ttrdur = np.arcsin((1e0+rpors)/smaors)
        trdura = priors[p]['period']*ttrdur/np.pi
        mintscale = []
        maxtscale = []
        for i, tvs in enumerate(time):
            mintscale.append(np.nanmin(abs(np.diff(np.sort(tvs)))))
            for o in set(orbits[i]):
                maxtscale.append(np.nanmax(tvs[orbits[i] == o]) -
                                 np.nanmin(tvs[orbits[i] == o]))
                pass
            pass
        tautknot = 1e0/(3e0*np.nanmin(mintscale))**2
        tknotmin = tmjd - np.nanmax(maxtscale)/2e0
        tknotmax = tmjd + np.nanmax(maxtscale)/2e0
        if priors[p]['inc'] != 9e1:
            if priors[p]['inc'] > 9e1:
                lowinc = 9e1
                upinc = 9e1 + 18e1*np.arcsin(1e0/smaors)/np.pi
                pass
            if priors[p]['inc'] < 9e1:
                lowinc = 9e1 - 18e1*np.arcsin(1e0/smaors)/np.pi
                upinc = 9e1
                pass
            pass
        else:
            lowinc = 0e0
            upinc = 9e1
            pass
        tauinc = 1e0/(priors[p]['inc']*1e-2)**2
        # INSTRUMENT MODEL PRIORS --------------------------------------------------------
        tauvs = 1e0/((1e-2/trdura)**2)
        tauvi = 1e0/(ootstd**2)
        selectfit = np.isfinite(flatwhite)
        tauwhite = 1e0/((np.nanmedian(flaterrwhite))**2)
        if tauwhite == 0: tauwhite = 1e0/(ootstd**2)
        shapettv = 2
        if shapettv < len(ttv): shapettv = len(ttv)
        shapevis = 2
        if shapevis < len(visits): shapevis = len(visits)
        if priors[p]['inc'] != 9e1: fixedinc = False
        else: fixedinc = True
        if 'eclipse' in selftype: fixedinc = True
        if 'WFC3' not in ext: fixedinc = True
        nodes = []
        ctxtupdt(orbp=priors[p], ecc=ecc, g1=g1, g2=g2, g3=g3, g4=g4,
                 orbits=orbits, period=period,
                 selectfit=selectfit, smaors=smaors,
                 time=time, tmjd=tmjd, ttv=ttv, visits=visits)
        # PYMC3 --------------------------------------------------------------------------
        with pm.Model():
            rprs = pm.TruncatedNormal('rprs', mu=rpors, tau=taurprs,
                                      lower=rpors/2e0, upper=2e0*rpors)
            nodes.append(rprs)
            if 'WFC3' in ext:
                alltknot = pm.TruncatedNormal('dtk', mu=tmjd, tau=tautknot,
                                              lower=tknotmin, upper=tknotmax,
                                              shape=shapettv)
                nodes.append(alltknot)
                if fixedinc: inc = priors[p]['inc']
                else:
                    inc = pm.TruncatedNormal('inc', mu=priors[p]['inc'], tau=tauinc,
                                             lower=lowinc, upper=upinc)
                    nodes.append(inc)
                    pass
                pass
            allvslope = pm.TruncatedNormal('vslope', mu=0e0, tau=tauvs,
                                           lower=-3e-2/trdura,
                                           upper=3e-2/trdura, shape=shapevis)
            alloslope = pm.Normal('oslope', mu=0e0, tau=tauvs, shape=shapevis)
            alloitcp = pm.Normal('oitcp', mu=1e0, tau=tauvi, shape=shapevis)
            nodes.append(allvslope)
            nodes.append(alloslope)
            nodes.append(alloitcp)
            if 'WFC3' in ext:
                # TTV + FIXED OR VARIABLE INC
                if fixedinc:
                    _whitedata = pm.Normal('whitedata', mu=fiorbital(*nodes),
                                           tau=tauwhite, observed=flatwhite[selectfit])
                    pass
                else:
                    _whitedata = pm.Normal('whitedata', mu=orbital(*nodes),
                                           tau=tauwhite, observed=flatwhite[selectfit])
                    pass
                pass
            else:
                # NO TTV, FIXED INC
                _whitedata = pm.Normal('whitedata', mu=nottvfiorbital(*nodes),
                                       tau=tauwhite, observed=flatwhite[selectfit])
                pass
            log.warning('>-- MCMC nodes: %s', str([n.name for n in nodes]))
            trace = pm.sample(chainlen, cores=4, tune=int(chainlen/2),
                              compute_convergence_checks=False, step=pm.Metropolis(),
                              progressbar=verbose)
            # GMR: Should be able to find it... Joker
            # pylint: disable=no-member
            mcpost = pm.stats.summary(trace)
            pass
        mctrace = {}
        for key in mcpost['mean'].keys():
            if len(key.split('[')) > 1:  # change PyMC3.8 key format to previous
                pieces = key.split('[')
                key = f"{pieces[0]}__{pieces[1].strip(']')}"
            tracekeys = key.split('__')
            if tracekeys.__len__() > 1:
                mctrace[key] = trace[tracekeys[0]][:, int(tracekeys[1])]
                pass
            else: mctrace[key] = trace[tracekeys[0]]
            pass
        postlc = []
        postim = []
        postsep = []
        postphase = []
        postflatphase = []
        ttvindex = 0
        if 'WFC3' in ext:
            if fixedinc: inclination = priors[p]['inc']
            else: inclination = np.nanmedian(mctrace['inc'])
            for i, v in enumerate(visits):
                if v in ttv:
                    posttk = np.nanmedian(mctrace[f'dtk__{ttvindex}'])
                    ttvindex += 1
                    pass
                else: posttk = tmjd
                postz, postph = datcore.time2z(time[i], inclination,
                                               posttk, smaors, period, ecc)
                if selftype in ['eclipse']: postph[postph < 0] = postph[postph < 0] + 1e0
                postsep.extend(postz)
                postphase.append(postph)
                postflatphase.extend(postph)
                postlc.extend(tldlc(abs(postz), np.nanmedian(mctrace['rprs']),
                                    g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0]))
                postim.append(timlc(time[i], orbits[i],
                                    vslope=np.nanmedian(mctrace[f'vslope__{i}']),
                                    vitcp=1e0,
                                    oslope=np.nanmedian(mctrace[f'oslope__{i}']),
                                    oitcp=np.nanmedian(mctrace[f'oitcp__{i}'])))
                pass
            pass
        else:
            omtk = ctxt.tmjd
            inclination = ctxt.orbp['inc']
            for i, v in enumerate(visits):
                postz, postph = datcore.time2z(time[i], inclination,
                                               omtk, smaors, period, ecc)
                if selftype in ['eclipse']: postph[postph < 0] = postph[postph < 0] + 1e0
                postsep.extend(postz)
                postphase.append(postph)
                postflatphase.extend(postph)
                postlc.extend(tldlc(abs(postz), np.nanmedian(mctrace['rprs']),
                                    g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0]))
                postim.append(timlc(time[i], orbits[i],
                                    vslope=np.nanmedian(mctrace[f'vslope__{i}']),
                                    vitcp=1e0,
                                    oslope=np.nanmedian(mctrace[f'oslope__{i}']),
                                    oitcp=np.nanmedian(mctrace[f'oitcp__{i}'])))
                pass
            pass
        out['data'][p]['postlc'] = postlc
        out['data'][p]['postim'] = postim
        out['data'][p]['postsep'] = postsep
        out['data'][p]['postphase'] = postphase
        out['data'][p]['postflatphase'] = postflatphase
        out['data'][p]['mcpost'] = mcpost
        out['data'][p]['mctrace'] = mctrace
        out['data'][p]['allttvfltrs'] = allttvfltrs
        out['data'][p]['allfltrs'] = allfltrs
        out['data'][p]['allttvs'] = allttvs
        out['STATUS'].append(True)
        if verbose:
            plt.figure(figsize=(10, 6))
            for iv, v in enumerate(visits):
                vlabel = allfltrs[iv]
                plt.plot(phase[iv], allwhite[iv], 'k+')
                plt.plot(postphase[iv], allwhite[iv]/postim[iv], 'o', label=vlabel)
                pass
            if visits.__len__() > 14: ncol = 2
            else: ncol = 1
            plt.plot(postflatphase, postlc, '^', label='M')
            plt.xlabel('Orbital Phase')
            plt.ylabel('Normalized Post White Light Curve')
            plt.legend(loc='best', ncol=ncol,
                       mode='expand', numpoints=1, borderaxespad=0., frameon=False)
            plt.tight_layout()
            plt.show()
            pass
        pass
    return True

def whitelight(nrm, fin, out, ext, selftype, multiwl, chainlen=int(1e4),
               verbose=False, parentprior=False):
    '''
    G. ROUDIER: Orbital parameters recovery
    '''
    wl = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = [p for p in nrm['data'].keys() if nrm['data'][p]['visits']]
    for p in planetloop:
        rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
        visits = nrm['data'][p]['visits']
        orbits = nrm['data'][p]['orbits']
        time = nrm['data'][p]['time']
        wave = nrm['data'][p]['wave']
        nspec = nrm['data'][p]['nspec']
        sep = nrm['data'][p]['z']
        phase = nrm['data'][p]['phase']
        photnoise = nrm['data'][p]['photnoise']
        out['data'][p] = {}
        out['data'][p]['nspec'] = nspec
        out['data'][p]['wave'] = wave
        out['data'][p]['visits'] = visits
        out['data'][p]['orbits'] = orbits
        allwhite = []
        allerrwhite = []
        flatminww = []
        flatmaxww = []
        for index, _v in enumerate(visits):
            white = []
            errwhite = []
            for w, s, e in zip(wave[index], nspec[index], photnoise[index]):
                select = np.isfinite(s)
                if True in select:
                    white.append(np.nanmean(s[select]))
                    errwhite.append(np.nanmedian(e[select])/np.sqrt(np.nansum(select)))
                    pass
                else:
                    white.append(np.nan)
                    errwhite.append(np.nan)
                    pass
                flatminww.append(min(w[select]))
                flatmaxww.append(max(w[select]))
                pass
            allwhite.append(white)
            allerrwhite.append(errwhite)
            pass
        flaterrwhite = []
        for r in allerrwhite: flaterrwhite.extend(r)
        flaterrwhite = np.array(flaterrwhite)
        flatwhite = []
        for w in allwhite: flatwhite.extend(w)
        flatwhite = np.array(flatwhite)
        flatz = []
        for z in sep: flatz.extend(z)
        flatz = np.array(flatz)
        flatphase = []
        for ph in phase: flatphase.extend(ph)
        flatphase = np.array(flatphase)
        allwwmin = min(flatminww)
        allwwmax = max(flatmaxww)
        renorm = np.nanmean(flatwhite[abs(flatz) > 1e0 + rpors])
        flatwhite /= renorm
        flaterrwhite /= renorm
        allwhite = [np.array(aw)/renorm for aw in allwhite]
        allerrwhite = [np.array(aew)/renorm for aew in allerrwhite]
        out['data'][p]['allwhite'] = allwhite
        out['data'][p]['phase'] = phase
        out['data'][p]['flatphase'] = flatphase
        # LIMB DARKENING ---------------------------------------------
        if selftype in ['transit']:
            whiteld = createldgrid([allwwmin], [allwwmax], priors,
                                   segmentation=int(10), verbose=verbose)
            g1, g2, g3, g4 = whiteld['LD']
            pass
        else: g1, g2, g3, g4 = [[0], [0], [0], [0]]
        wlmod = tldlc(abs(flatz), rpors, g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0])
        out['data'][p]['whiteld'] = [g1[0], g2[0], g3[0], g4[0]]
        # PLOT -------------------------------------------------------
        if verbose:
            plt.figure(figsize=(10, 6))
            for v in visits:
                index = visits.index(v)
                plt.plot(phase[index], allwhite[index], 'o', label=str(v))
                pass
            if visits.__len__() > 14: ncol = 2
            else: ncol = 1
            plt.plot(flatphase, wlmod, '^', label='M')
            plt.xlabel('Orbital Phase')
            plt.ylabel('Normalized Raw White Light Curve')
            plt.legend(bbox_to_anchor=(1 + 0.1*(ncol - 0.5), 0.5),
                       loc=5, ncol=ncol, mode='expand', numpoints=1,
                       borderaxespad=0., frameon=False)
            plt.tight_layout(rect=[0,0,(1 - 0.1*ncol),1])
            plt.show()
            pass
        # TTV --------------------------------------------------------
        ttv = []
        for index, v in enumerate(visits):
            select = (abs(sep[index]) < (1e0+rpors)) & (abs(sep[index]) > (1e0-rpors))
            if True in select: ttv.append(v)
            pass
        # PRIORS -----------------------------------------------------
        tmjd = priors[p]['t0']
        if tmjd > 2400000.5: tmjd -= 2400000.5
        allttvfltrs = np.array(multiwl['data'][p]['allttvfltrs'])
        if ext in allttvfltrs:
            ttvmask = allttvfltrs == ext
            alltknot = [np.median(multiwl['data'][p]['mctrace']['dtk__'+str(i)])
                        for i, cond in enumerate(ttvmask) if cond]
            pass
        else: alltknot = []
        period = priors[p]['period']
        ecc = priors[p]['ecc']
        inc = priors[p]['inc']
        smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
        ootstd = np.nanstd(flatwhite[abs(flatz) > 1e0 + rpors])
        taurprs = 1e0/(rpors*1e-2)**2
        ttrdur = np.arcsin((1e0+rpors)/smaors)
        trdura = priors[p]['period']*ttrdur/np.pi
        mintscale = []
        maxtscale = []
        for i, tvs in enumerate(time):
            mintscale.append(np.nanmin(abs(np.diff(np.sort(tvs)))))
            for o in set(orbits[i]):
                maxtscale.append(np.nanmax(tvs[orbits[i] == o]) -
                                 np.nanmin(tvs[orbits[i] == o]))
                pass
            pass
        # INSTRUMENT MODEL PRIORS --------------------------------------------------------
        tauvs = 1e0/((1e-2/trdura)**2)
        tauvi = 1e0/(ootstd**2)
        selectfit = np.isfinite(flatwhite)
        tauwhite = 1e0/((np.nanmedian(flaterrwhite))**2)
        if tauwhite == 0: tauwhite = 1e0/(ootstd**2)
        shapettv = 2
        if shapettv < len(ttv): shapettv = len(ttv)
        shapevis = 2
        if shapevis < len(visits): shapevis = len(visits)
        if 'inc' in multiwl['data'][p]['mctrace']:
            inc = np.median(multiwl['data'][p]['mctrace']['inc'])
            pass
        else: inc = priors[p]['inc']
        nodes = []
        ctxtupdt(orbp=priors[p], ecc=ecc, g1=g1, g2=g2, g3=g3, g4=g4,
                 orbits=orbits, period=period,
                 selectfit=selectfit, smaors=smaors,
                 time=time, tmjd=tmjd, ttv=ttv, visits=visits, ginc=inc, gttv=alltknot)
        # Set up priors for if parentprior is true
        if selftype in ['transit'] and 'G141-SCAN' in ext:
            oslope_alpha = 0.004633620507894198; oslope_beta = 0.012556238027618398
            vslope_alpha = -0.0013980054382670398; vslope_beta = 0.0016336714834115414
            oitcp_alpha = 1.0000291019498646; oitcp_beta = 7.176342068341074e-05
        elif selftype in ['transit'] and 'G430L-STARE' in ext:
            oslope_alpha = 0.04587012155603797; oslope_beta = 0.03781489933244744
            vslope_alpha = -0.0006729851708645652; vslope_beta = 0.008957326101096843
            oitcp_alpha = 0.9999462758123321; oitcp_beta = 0.0001556495709041709
        elif selftype in ['transit'] and 'G750L-STARE' in ext:
            oslope_alpha = 0.027828748287645484; oslope_beta = 0.02158079144341918
            vslope_alpha = 0.0012904512219440258; vslope_beta = 0.004194712807907309
            oitcp_alpha = 1.0000037868438292; oitcp_beta = 4.845142445585787e-05
        else:  # Handle estimation for non-optimized instrumentation
            # Lorentzian beta parameter is not directly analogous
            # to standard deviation but is approximately so
            vslope_alpha = 0e0
            vslope_beta = (1/tauvs)**0.5
            oslope_alpha = 0e0
            oslope_beta = (1/tauvs)**0.5
            oitcp_alpha = 1e0
            oitcp_beta = (1/tauvi)**0.5
        # PYMC3 --------------------------------------------------------------------------
        with pm.Model():
            rprs = pm.TruncatedNormal('rprs', mu=rpors, tau=taurprs,
                                      lower=rpors/2e0, upper=2e0*rpors)
            nodes.append(rprs)
            if parentprior:
                # use parent distr fitted Lorentzians (also called Cauchy)
                allvslope = pm.Cauchy('vslope', alpha=vslope_alpha,
                                      beta=vslope_beta, shape=shapevis)
                alloslope = pm.Cauchy('oslope', alpha=oslope_alpha,
                                      beta=oslope_beta, shape=shapevis)
                alloitcp = pm.Cauchy('oitcp', alpha=oitcp_alpha,
                                     beta=oitcp_beta, shape=shapevis)
            else:
                allvslope = pm.TruncatedNormal('vslope', mu=0e0, tau=tauvs,
                                               lower=-3e-2/trdura,
                                               upper=3e-2/trdura, shape=shapevis)
                alloslope = pm.Normal('oslope', mu=0e0, tau=tauvs, shape=shapevis)
                alloitcp = pm.Normal('oitcp', mu=1e0, tau=tauvi, shape=shapevis)
            nodes.append(allvslope)
            nodes.append(alloslope)
            nodes.append(alloitcp)
            # FIXED ORBITAL SOLUTION
            _whitedata = pm.Normal('whitedata', mu=nottvfiorbital(*nodes),
                                   tau=tauwhite, observed=flatwhite[selectfit])
            log.warning('>-- MCMC nodes: %s', str([n.name for n in nodes]))
            trace = pm.sample(chainlen, cores=4, tune=int(chainlen/2),
                              compute_convergence_checks=False, step=pm.Metropolis(),
                              progressbar=verbose)
            # GMR: Should be able to find it... Joker
            # pylint: disable=no-member
            mcpost = pm.stats.summary(trace)
            pass
        mctrace = {}
        for key in mcpost['mean'].keys():
            if len(key.split('[')) > 1:  # change PyMC3.8 key format to previous
                pieces = key.split('[')
                key = f"{pieces[0]}__{pieces[1].strip(']')}"
            tracekeys = key.split('__')
            if tracekeys.__len__() > 1:
                mctrace[key] = trace[tracekeys[0]][:, int(tracekeys[1])]
                pass
            else: mctrace[key] = trace[tracekeys[0]]
            pass
        postlc = []
        postim = []
        postsep = []
        postphase = []
        postflatphase = []
        omtk = ctxt.tmjd
        inclination = ctxt.ginc
        for i, v in enumerate(visits):
            if v in ttv: omtk = float(alltknot[ttv.index(v)])
            else: omtk = tmjd
            postz, postph = datcore.time2z(time[i], inclination,
                                           omtk, smaors, period, ecc)
            if selftype in ['eclipse']: postph[postph < 0] = postph[postph < 0] + 1e0
            postsep.extend(postz)
            postphase.append(postph)
            postflatphase.extend(postph)
            postlc.extend(tldlc(abs(postz), np.nanmedian(mctrace['rprs']),
                                g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0]))
            postim.append(timlc(time[i], orbits[i],
                                vslope=np.nanmedian(mctrace[f'vslope__{i}']),
                                vitcp=1e0,
                                oslope=np.nanmedian(mctrace[f'oslope__{i}']),
                                oitcp=np.nanmedian(mctrace[f'oitcp__{i}'])))
            pass
        out['data'][p]['postlc'] = postlc
        out['data'][p]['postim'] = postim
        out['data'][p]['postsep'] = postsep
        out['data'][p]['postphase'] = postphase
        out['data'][p]['postflatphase'] = postflatphase
        out['data'][p]['mcpost'] = mcpost
        out['data'][p]['mctrace'] = mctrace
        out['data'][p]['tauwhite'] = tauwhite
        out['STATUS'].append(True)
        data = np.array(out['data'][p]['allwhite'])
        newdata = []
        for d in data: newdata.extend(d)
        newdata = np.array(newdata)
        residuals = newdata - postlc  # raw data - model

        def sample_dist(distribution,num_samples,bw_adjust=.35):
            interval = np.linspace(min(distribution),max(distribution),1000)
            fit = gaussian_kde(distribution,bw_method=bw_adjust)(interval)
            samples = random.choices(interval,fit,k=num_samples)
            return samples,interval,fit

        all_sims = []
        for i in range(100):
            samples,_,_ = sample_dist(residuals,len(newdata),bw_adjust=0.05)
            simulated_raw_data = np.array(postlc)+np.array(samples)
            all_sims.append(simulated_raw_data)
        out['data'][p]['simulated'] = all_sims  # certain targets the simulated data will be empty bc they're not gaussian
        wl = True
        if verbose:
            plt.figure(figsize=(10, 6))
            for iv, v in enumerate(visits):
                plt.plot(phase[iv], allwhite[iv], 'k+')
                plt.plot(postphase[iv], allwhite[iv]/postim[iv], 'o', label=str(v))
                pass
            if visits.__len__() > 14: ncol = 2
            else: ncol = 1
            plt.plot(postflatphase, postlc, '^', label='M')
            plt.xlabel('Orbital Phase')
            plt.ylabel('Normalized Post White Light Curve')
            plt.legend(bbox_to_anchor=(1 + 0.1*(ncol - 0.5), 0.5), loc=5, ncol=ncol,
                       mode='expand', numpoints=1, borderaxespad=0., frameon=False)
            plt.tight_layout(rect=[0,0,(1 - 0.1*ncol),1])
            plt.show()
            pass
        pass
    return wl
# ----------------------- --------------------------------------------
# -- TRANSIT LIMB DARKENED LIGHT CURVE -- ----------------------------
def tldlc(z, rprs, g1=0, g2=0, g3=0, g4=0, nint=int(8**2)):
    '''
    G. ROUDIER: Light curve model
    '''
    ldlc = np.zeros(z.size)
    xin = z.copy() - rprs
    xin[xin < 0e0] = 0e0
    xout = z.copy() + rprs
    xout[xout > 1e0] = 1e0
    select = xin > 1e0
    if True in select: ldlc[select] = 1e0
    inldlc = []
    xint = np.linspace(1e0, 0e0, nint)
    znot = z.copy()[~select]
    xinnot = np.arccos(xin[~select])
    xoutnot = np.arccos(xout[~select])
    xrs = np.array([xint]).T*(xinnot - xoutnot) + xoutnot
    xrs = np.cos(xrs)
    diffxrs = np.diff(xrs, axis=0)
    extxrs = np.zeros((xrs.shape[0]+1, xrs.shape[1]))
    extxrs[1:-1, :] = xrs[1:,:] - diffxrs/2.
    extxrs[0, :] = xrs[0, :] - diffxrs[0]/2.
    extxrs[-1, :] = xrs[-1, :] + diffxrs[-1]/2.
    occulted = vecoccs(znot, extxrs, rprs)
    diffocc = np.diff(occulted, axis=0)
    si = vecistar(xrs, g1, g2, g3, g4)
    drop = np.sum(diffocc*si, axis=0)
    inldlc = 1. - drop
    ldlc[~select] = np.array(inldlc)
    return ldlc
# --------------------------------------- ----------------------------
# -- STELLAR EXTINCTION LAW -- ---------------------------------------
def vecistar(xrs, g1, g2, g3, g4):
    '''
    G. ROUDIER: Stellar surface extinction model
    '''
    ldnorm = (-g1/10e0 - g2/6e0 - 3e0*g3/14e0 - g4/4e0 + 5e-1)*2e0*np.pi
    select = xrs < 1e0
    mu = np.zeros(xrs.shape)
    mu[select] = (1e0 - xrs[select]**2)**(1e0/4e0)
    s1 = g1*(1e0 - mu)
    s2 = g2*(1e0 - mu**2)
    s3 = g3*(1e0 - mu**3)
    s4 = g4*(1e0 - mu**4)
    outld = (1e0 - (s1+s2+s3+s4))/ldnorm
    return outld
# ---------------------------- ---------------------------------------
# -- STELLAR SURFACE OCCULTATION -- ----------------------------------
def vecoccs(z, xrs, rprs):
    '''
    G. ROUDIER: Stellar surface occulation model
    '''
    out = np.zeros(xrs.shape)
    vecxrs = xrs.copy()
    selx = vecxrs > 0e0
    veczsel = np.array([z.copy()]*xrs.shape[0])
    veczsel[veczsel < 0e0] = 0e0
    select1 = (vecxrs <= rprs - veczsel) & selx
    select2 = (vecxrs >= rprs + veczsel) & selx
    select = (~select1) & (~select2) & selx
    zzero = veczsel == 0e0
    if True in select1 & zzero:
        out[select1 & zzero] = np.pi*(np.square(vecxrs[select1 & zzero]))
        pass
    if True in select2 & zzero: out[select2 & zzero] = np.pi*(rprs**2)
    if True in select & zzero: out[select & zzero] = np.pi*(rprs**2)
    if True in select1 & ~zzero:
        out[select1 & ~zzero] = np.pi*(np.square(vecxrs[select1 & ~zzero]))
        pass
    if True in select2: out[select2 & ~zzero] = np.pi*(rprs**2)
    if True in select & ~zzero:
        redxrs = vecxrs[select & ~zzero]
        redz = veczsel[select & ~zzero]
        s1 = (np.square(redz) + np.square(redxrs) - rprs**2)/(2e0*redz*redxrs)
        s1[s1 > 1e0] = 1e0
        s2 = (np.square(redz) + rprs**2 - np.square(redxrs))/(2e0*redz*rprs)
        s2[s2 > 1e0] = 1e0
        fs3 = -redz + redxrs + rprs
        ss3 = redz + redxrs - rprs
        ts3 = redz - redxrs + rprs
        os3 = redz + redxrs + rprs
        s3 = fs3*ss3*ts3*os3
        zselect = s3 < 0e0
        if True in zselect: s3[zselect] = 0e0
        out[select & ~zzero] = (np.square(redxrs)*np.arccos(s1) +
                                (rprs**2)*np.arccos(s2) - (5e-1)*np.sqrt(s3))
        pass
    return out
# --------------------------------- ----------------------------------
# -- CREATE LD GRID -- -----------------------------------------------
def createldgrid(minmu, maxmu, orbp,
                 ldmodel='nonlinear', phoenixmin=1e-1,
                 segmentation=int(10), verbose=False):
    '''
    G. ROUDIER: Wrapper around LDTK downloading tools
    LDTK: Parviainen et al. https://github.com/hpparvi/ldtk
    '''
    tstar = orbp['T*']
    terr = np.sqrt(abs(orbp['T*_uperr']*orbp['T*_lowerr']))
    fehstar = orbp['FEH*']
    feherr = np.sqrt(abs(orbp['FEH*_uperr']*orbp['FEH*_lowerr']))
    loggstar = orbp['LOGG*']
    loggerr = np.sqrt(abs(orbp['LOGG*_uperr']*orbp['LOGG*_lowerr']))
    log.warning('>-- Temperature: %s +/- %s', str(tstar), str(terr))
    log.warning('>-- Metallicity: %s +/- %s', str(fehstar), str(feherr))
    log.warning('>-- Surface Gravity: %s +/- %s', str(loggstar), str(loggerr))
    niter = int(len(minmu)/segmentation) + 1
    allcl = None
    allel = None
    out = {}
    avmu = [np.mean([mm, xm]) for mm, xm in zip(minmu, maxmu)]
    for i in np.arange(niter):
        loweri = i*segmentation
        upperi = (i+1)*segmentation
        if i == (niter-1): upperi = len(avmu)
        munm = 1e3*np.array(avmu[loweri:upperi])
        munmmin = 1e3*np.array(minmu[loweri:upperi])
        munmmax = 1e3*np.array(maxmu[loweri:upperi])
        filters = [BoxcarFilter(str(mue), mun, mux)
                   for mue, mun, mux in zip(munm, munmmin, munmmax)]
        sc = LDPSetCreator(teff=(tstar, terr), logg=(loggstar, loggerr),
                           z=(fehstar, feherr), filters=filters)
        ps = sc.create_profiles(nsamples=int(1e4))
        itpfail = False
        for testprof in ps.profile_averages:
            if np.all(~np.isfinite(testprof)): itpfail = True
            pass
        nfail = 1e0
        while itpfail:
            nfail *= 2
            sc = LDPSetCreator(teff=(tstar, nfail*terr), logg=(loggstar, loggerr),
                               z=(fehstar, feherr), filters=filters)
            ps = sc.create_profiles(nsamples=int(1e4))
            itpfail = False
            for testprof in ps.profile_averages:
                if np.all(~np.isfinite(testprof)): itpfail = True
                pass
            pass
        cl, el = ldx(ps.profile_mu, ps.profile_averages, ps.profile_uncertainties,
                     mumin=phoenixmin, debug=verbose, model=ldmodel)
        if allcl is None: allcl = cl
        else: allcl = np.concatenate((allcl, cl), axis=0)
        if allel is None: allel = el
        else: allel = np.concatenate((allel, el), axis=0)
        pass
    allel[allel > 1.] = 0.
    allel[~np.isfinite(allel)] = 0.
    out['MU'] = avmu
    out['LD'] = allcl.T
    out['ERR'] = allel.T
    for i, _m in enumerate(allcl.T):
        log.warning('>-- LD%s: %s +/- %s',
                    str(int(i)), str(float(allcl.T[i])), str(float(allel.T[i])))
        pass
    return out
# -------------------- -----------------------------------------------
# -- LDX -- ----------------------------------------------------------
def ldx(psmu, psmean, psstd, mumin=1e-1, debug=False, model='nonlinear'):
    '''
    G. ROUDIER: Limb darkening coefficient retrievial on PHOENIX GRID models,
    OPTIONAL mumin set up on HAT-P-41 stellar class
    '''
    mup=np.array(psmu).copy()
    prfs=np.array(psmean).copy()
    sprfs=np.array(psstd).copy()
    nwave=prfs.shape[0]
    select=(mup > mumin)
    fitmup=mup[select]
    fitprfs=prfs[:, select]
    fitsprfs=sprfs[:, select]
    cl=[]
    el=[]
    params=lm.Parameters()
    params.add('gamma1', value=1e-1)
    params.add('gamma2', value=5e-1)
    params.add('gamma3', value=1e-1)
    params.add('gamma4', expr='1 - gamma1 - gamma2 - gamma3')
    if debug: plt.figure()
    for iwave in np.arange(nwave):
        select = fitsprfs[iwave] == 0e0
        if True in select: fitsprfs[iwave][select] = 1e-10
        if model == 'linear':
            params['gamma1'].value = 0
            params['gamma1'].vary = False
            params['gamma3'].value = 0
            params['gamma3'].vary = False
            params['gamma4'].value = 0
            params['gamma4'].vary = False
            out=lm.minimize(lnldx, params,
                            args=(fitmup, fitprfs[iwave], fitsprfs[iwave]))
            cl.append([out.params['gamma1'].value])
            el.append([out.params['gamma1'].stderr])
            pass
        if model == 'quadratic':
            params['gamma1'].value = 0
            params['gamma1'].vary = False
            params['gamma3'].value = 0
            params['gamma3'].vary = False
            out=lm.minimize(qdldx, params,
                            args=(mup, prfs[iwave], sprfs[iwave]))
            cl.append([out.params['gamma1'].value, out.params['gamma2'].value])
            el.append([out.params['gamma1'].stderr, out.params['gamma2'].stderr])
            pass
        if model == 'nonlinear':
            out = lm.minimize(nlldx, params,
                              args=(fitmup, fitprfs[iwave], fitsprfs[iwave]))
            cl.append([out.params['gamma1'].value, out.params['gamma2'].value,
                       out.params['gamma3'].value, out.params['gamma4'].value])
            el.append([out.params['gamma1'].stderr, out.params['gamma2'].stderr,
                       out.params['gamma3'].stderr, out.params['gamma4'].stderr])
            pass
        if debug:
            plt.plot(mup, prfs[iwave], 'k^')
            plt.errorbar(fitmup, fitprfs[iwave], yerr=fitsprfs[iwave], ls='None')
            if model == 'linear': plt.plot(fitmup, lnldx(out.params, fitmup))
            if model == 'quadratic': plt.plot(fitmup, qdldx(out.params, fitmup))
            if model == 'nonlinear': plt.plot(fitmup, nlldx(out.params, fitmup))
            pass
        pass
    if debug:
        plt.ylabel('$I(\\mu)$')
        plt.xlabel('$\\mu$')
        plt.title(model)
        plt.show()
        pass
    return np.array(cl), np.array(el)
# --------- ----------------------------------------------------------
# -- LNLDX -- --------------------------------------------------------
def lnldx(params, x, data=None, weights=None):
    '''
    G. ROUDIER: Linear law
    '''
    gamma1=params['gamma1'].value
    model=LinearModel.evaluate(x, [gamma1])
    if data is None: return model
    if weights is None: return data - model
    return (data - model)/weights
# ----------- --------------------------------------------------------
# -- QDLDX -- --------------------------------------------------------
def qdldx(params, x, data=None, weights=None):
    '''
    G. ROUDIER: Quadratic law
    '''
    gamma1 = params['gamma1'].value
    gamma2 = params['gamma2'].value
    model = QuadraticModel.evaluate(x, [gamma1, gamma2])
    if data is None: return model
    if weights is None: return data - model
    return (data - model)/weights
# ----------- --------------------------------------------------------
# -- NLLDX -- --------------------------------------------------------
def nlldx(params, x, data=None, weights=None):
    '''
    G. ROUDIER: Non Linear law
    '''
    gamma1 = params['gamma1'].value
    gamma2 = params['gamma2'].value
    gamma3 = params['gamma3'].value
    gamma4 = params['gamma4'].value
    model = NonlinearModel.evaluate(x, [gamma1, gamma2, gamma3, gamma4])
    if data is None: return model
    if weights is None: return data - model
    return (data - model)/weights
# ----------- --------------------------------------------------------
# -- INSTRUMENT MODEL -- ---------------------------------------------
def timlc(vtime, orbits, vslope=0, vitcp=1e0, oslope=0, oitcp=1e0):
    '''
    G. ROUDIER: WFC3 intrument model
    '''
    xout = np.array(vtime) - np.mean(vtime)
    vout = vslope*xout + vitcp
    oout = np.ones(vout.size)
    for o in set(np.sort(orbits)):
        select = orbits == o
        otime = xout[select] - np.mean(xout[select])
        olin = oslope*otime + oitcp
        oout[select] = olin
        pass
    return vout*oout
# -- RAMP MODEL -- ---------------------------------------------------
def hstramp(params, rtime, data=None):
    '''
    G. ROUDIER: HST breathing model
    '''
    louttime = np.array(rtime) - np.mean(rtime)
    ramptime = (louttime - np.min(louttime))*(36e2)*(24e0)  # SECONDS
    lout = params['oslope'].value*louttime + params['oitcp'].value
    ramp = 1e0 - np.exp(-(ramptime + (1e1)**params['ologdelay'].value)/
                        ((1e1)**params['ologtau'].value))
    if data is None: out = ramp*lout
    else:
        select = np.isfinite(data)
        if True in select: out = data[select] - (ramp[select]*lout[select])
        pass
    return out
# ---------------- ---------------------------------------------------
# -- SPECTRUM -- -----------------------------------------------------
def spectrumversion():
    '''
    G. ROUDIER: Neutral outlier rej/inpaint
    Whitelight +/- 5Hs instead of Previous +/- 1PN
    LDX robust to infinitely small errors + spectral binning boost
    R. Estrela: 1.2.0 lowing down the resolution of G430L
    N. Huber-Feely: 1.2.1 Add saving of trace to SV
    K. PEARSON: 1.2.2 JWST NIRISS
    R ESTRELA: 1.3.0 Merged Spectra Capability
    '''
    return dawgie.VERSION(1,3,2)

def spectrum(fin, nrm, wht, out, ext, selftype,
             chainlen=int(1e4), verbose=False, lcplot=False):
    '''
    G. ROUDIER: Exoplanet spectrum recovery
    '''
    exospec = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = [p for p in nrm['data'].keys() if nrm['data'][p]['visits']]
    for p in planetloop:
        out['data'][p] = {'LD':[]}
        rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
        smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
        ttrdur = np.arcsin((1e0+rpors)/smaors)
        trdura = priors[p]['period']*ttrdur/np.pi
        vrange = nrm['data'][p]['vrange']
        wave = nrm['data'][p]['wavet']
        waves = nrm['data'][p]['wave']
        nspec = nrm['data'][p]['nspec']
        photnoise = nrm['data'][p]['photnoise']
        if 'G750' in ext:
            wave, _trash = binnagem(wave, 105)  # 150
            wave = np.resize(wave,(1,105))
            pass
        if 'G430' in ext:
            wave, _trash = binnagem(wave, 121)  # 182
            wave = np.resize(wave,(1,121))
            pass
        time = nrm['data'][p]['time']
        visits = nrm['data'][p]['visits']
        orbits = nrm['data'][p]['orbits']
        disp = nrm['data'][p]['dispersion']
        im = wht['data'][p]['postim']
        allz = wht['data'][p]['postsep']
        allphase = np.array(wht['data'][p]['postflatphase'])
        whiterprs = np.nanmedian(wht['data'][p]['mctrace']['rprs'])
        allwave = []
        allspec = []
        allim = []
        allpnoise = []
        alldisp = []
        for w, s, i, n, d in zip(waves, nspec, im, photnoise, disp):
            allwave.extend(w)
            allspec.extend(s)
            allim.extend(i)
            allpnoise.extend(n)
            alldisp.extend(d)
            pass
        alldisp = np.array(alldisp)
        allim = np.array(allim)
        allz = np.array(allz)
        if 'STIS' in ext:
            disp = np.median([np.median(np.diff(w)) for w in wave])
            nbin = np.min([len(w) for w in wave])
            wavel = [np.min(w) for w in wave]
            wavec = np.arange(nbin)*disp + np.mean([np.max(wavel), np.min(wavel)])
            lwavec = wavec - disp/2e0
            hwavec = wavec + disp/2e0
            pass
        # MULTI VISITS COMMON WAVELENGTH GRID --------------------------------------------
        if 'WFC3' in ext:
            wavec, _t = tplbuild(allspec, allwave, vrange, alldisp*1e-4, medest=True)
            wavec = np.array(wavec)
            temp = [np.diff(wavec)[0]]
            temp.extend(np.diff(wavec))
            lwavec = wavec - np.array(temp)/2e0
            temp = list(np.diff(wavec))
            temp.append(np.diff(wavec)[-1])
            hwavec = wavec + np.array(temp)/2e0
            pass
        # EXCLUDE PARTIAL LIGHT CURVES AT THE EDGES --------------------------------------
        wavec = wavec[1:-2]
        lwavec = lwavec[1:-2]
        hwavec = hwavec[1:-2]
        # EXCLUDE ALL NAN CHANNELS -------------------------------------------------------
        allnanc = []
#         for wl, wh in zip(lwavec[0:6], hwavec[0:6]):
        for wl, wh in zip(lwavec, hwavec):
            select = [(w > wl) & (w < wh) for w in allwave]
            if 'STIS' in ext:
                data = np.array([np.nanmean(d[s]) for d, s in zip(allspec, select)])
                pass
            else: data = np.array([np.median(d[s]) for d, s in zip(allspec, select)])
            if np.all(~np.isfinite(data)): allnanc.append(True)
            else: allnanc.append(False)
            pass
        lwavec = [lwv for lwv, lln in zip(lwavec, allnanc) if not lln]
        hwavec = [hwv for hwv, lln in zip(hwavec, allnanc) if not lln]
        # LOOP OVER WAVELENGTH BINS ------------------------------------------------------
        out['data'][p]['WB'] = []
        out['data'][p]['WBlow'] = []
        out['data'][p]['WBup'] = []
        out['data'][p]['ES'] = []
        out['data'][p]['ESerr'] = []
        out['data'][p]['LD'] = []
        out['data'][p]['MCPOST'] = []
        out['data'][p]['MCTRACE'] = []
        out['data'][p]['LCFIT'] = []
        out['data'][p]['RSTAR'] = []
        out['data'][p]['rp0hs'] = []
        out['data'][p]['Hs'] = []
        startflag = True
        for wl, wh in zip(lwavec, hwavec):
            select = [(w > wl) & (w < wh) for w in allwave]
            if 'STIS' in ext:
                data = np.array([np.nanmean(d[s]) for d, s in zip(allspec, select)])
                dnoise = np.array([(1e0/np.sum(s))*np.sqrt(np.nansum((n[s])**2))
                                   for n, s in zip(allpnoise, select)])
                pass
            else:
                data = np.array([np.nanmean(d[s])
                                 for d, s in zip(allspec, select)])
                dnoise = np.array([np.nanmedian(n[s])/np.sqrt(np.nansum(s))
                                   for n, s in zip(allpnoise, select)])
                pass
            valid = np.isfinite(data)
            if selftype in ['transit']:
                try:
                    bld = createldgrid([wl], [wh], priors, segmentation=int(10))
                    pass
                except TypeError:
                    log.warning('>-- INCREASED BIN SIZE')
                    increment = 1e2*abs(wh - wl)
                    bld = createldgrid([wl - increment], [wh + increment], priors,
                                       segmentation=int(10))
                    pass
                g1, g2, g3, g4 = bld['LD']
                pass
            else: g1, g2, g3, g4 = [[0], [0], [0], [0]]
            out['data'][p]['LD'].append([g1[0], g2[0], g3[0], g4[0]])
            model = tldlc(abs(allz), whiterprs, g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0])

            if lcplot:
                plt.figure()
                plt.title(str(int(1e3*np.mean([wl, wh])))+' nm')
                plt.plot(allphase[valid], data[valid]/allim[valid], 'o')
                plt.plot(allphase[valid], model[valid], '^')
                plt.xlabel('Orbital phase')
                plt.show()
                pass
            # PRIORS ---------------------------------------------------------------------
            sscmks = syscore.ssconstants(mks=True)
            eqtemp = priors['T*']*np.sqrt(priors['R*']*sscmks['Rsun/AU']/
                                          (2.*priors[p]['sma']))
            pgrid = np.arange(np.log(10.)-15., np.log(10.)+15./100, 15./99)
            pgrid = np.exp(pgrid)
            pressure = pgrid[::-1]
            mixratio, fH2, fHe = crbutil.crbce(pressure, eqtemp)
            mmw, fH2, fHe = crbutil.getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
            mmw = mmw*cst.m_p  # [kg]
            Hs = cst.Boltzmann*eqtemp/(mmw*1e-2*(10.**priors[p]['logg']))  # [m]
            Hs = Hs/(priors['R*']*sscmks['Rsun'])
            tauvs = 1e0/((1e-2/trdura)**2)
            ootstd = np.nanstd(data[abs(allz) > (1e0 + whiterprs)])
            tauvi = 1e0/(ootstd**2)
            nodes = []
            tauwbdata = 1e0/dnoise**2
            # PRIOR WIDTH ----------------------------------------------------------------
            # noot = np.sum(abs(allz) > (1e0 + whiterprs))
            # nit = allz.size - noot
            # Noise propagation forecast on transit depth
            # propphn = np.nanmedian(dnoise)*(1e0 - whiterprs**2)
            # *np.sqrt(1e0/nit + 1e0/noot)
            # dirtypn = np.sqrt(propphn + whiterprs**2) - whiterprs
            prwidth = 2e0*Hs
            # PRIOR CENTER ---------------------------------------------------------------
            prcenter = whiterprs
            # UPDATE GLOBALS -------------------------------------------------------------
            shapevis = 2
            if shapevis < len(visits): shapevis = len(visits)
            ctxtupdt(allz=allz, g1=g1, g2=g2, g3=g3, g4=g4,
                     orbits=orbits, smaors=smaors, time=time, valid=valid, visits=visits)
            # PYMC3 ----------------------------------------------------------------------
            with pm.Model():
                if startflag:
                    lowstart = whiterprs - 5e0*Hs
                    lowstart = max(lowstart, 0)
                    upstart = whiterprs + 5e0*Hs
                    rprs = pm.Uniform('rprs', lower=lowstart, upper=upstart)
                    pass
                else:
                    rprs = pm.Normal('rprs', mu=prcenter, tau=1e0/(prwidth**2))
#                     lowstart = whiterprs - 5e0*Hs
#                     if lowstart < 0: lowstart = 0
#                     upstart = whiterprs + 5e0*Hs
#                     rprs = pm.Uniform('rprs', lower=lowstart, upper=upstart)
#                     pass
                allvslope = pm.TruncatedNormal('vslope', mu=0e0, tau=tauvs,
                                               lower=-3e-2/trdura,
                                               upper=3e-2/trdura, shape=shapevis)
                alloslope = pm.Normal('oslope', mu=0, tau=tauvs, shape=shapevis)
                alloitcp = pm.Normal('oitcp', mu=1e0, tau=tauvi, shape=shapevis)
                nodes.append(rprs)
                nodes.append(allvslope)
                nodes.append(alloslope)
                nodes.append(alloitcp)
                _wbdata = pm.Normal('wbdata', mu=lcmodel(*nodes),
                                    tau=np.nanmedian(tauwbdata[valid]),
                                    observed=data[valid])
                trace = pm.sample(chainlen, cores=4, tune=int(chainlen/2),
                                  compute_convergence_checks=False, step=pm.Metropolis(),
                                  progressbar=verbose)
                # GMR: Should be able to find it... Joker
                # pylint: disable=no-member
                mcpost = pm.stats.summary(trace)
                pass
            # Exclude first channel with Uniform prior
            if not startflag:
                # save MCMC samples in SV
                mctrace = {}
                mcests = {}
                for key in mcpost['mean'].keys():
                    if len(key.split('[')) > 1:  # change PyMC3.8 key format to previous
                        pieces = key.split('[')
                        key = f"{pieces[0]}__{pieces[1].strip(']')}"
                    tracekeys = key.split('__')
                    if tracekeys.__len__() > 1:
                        mctrace[key] = trace[tracekeys[0]][:, int(tracekeys[1])]
                        mcests[key] = np.nanmedian(mctrace[key])
                        pass
                    else:
                        mctrace[key] = trace[tracekeys[0]]
                        mcests[key] = np.nanmedian(mctrace[key])
                    pass
                # save rprs
                clspvl = np.nanmedian(trace['rprs'])
                # now produce fitted estimates

                # GMR: Huh I dont remember coding that one I m blind
                # Pulling out a joker
                def get_ests(n, v):
                    '''for param get all visit param values as list'''
                    # pylint: disable=cell-var-from-loop
                    return [mcests[f'{n}__{i}'] for i in range(len(v))]

                specparams = (mcests['rprs'], get_ests('vslope', visits),
                              get_ests('oslope', visits), get_ests('oitcp', visits))
                _r, avs, aos, aoi = specparams
                allimout = []
                for iv in range(len(visits)):
                    imout = timlc(time[iv], orbits[iv],
                                  vslope=float(avs[iv]), vitcp=1e0,
                                  oslope=float(aos[iv]), oitcp=float(aoi[iv]))
                    allimout.extend(imout)
                    pass
                allimout = np.array(allimout)
                lout = tldlc(abs(allz), clspvl, g1=g1[0], g2=g2[0], g3=g3[0],
                             g4=g4[0])
                lout = lout*np.array(allimout)
                lcfit = {'expected': lout[valid], 'observed': data[valid],
                         'im': allimout[valid], 'phase': allphase[valid],
                         'dnoise': np.nanmedian(dnoise[valid]),
                         'residuals': data[valid]-lout[valid]}
                # Spectrum outlier rejection + inpaint with np.nan
                if abs(clspvl - whiterprs) > 5e0*Hs: clspvl = np.nan
                out['data'][p]['ES'].append(clspvl)
                out['data'][p]['ESerr'].append(np.nanstd(trace['rprs']))
                out['data'][p]['MCPOST'].append(mcpost)
                out['data'][p]['MCTRACE'].append(mctrace)
                out['data'][p]['WBlow'].append(wl)
                out['data'][p]['WBup'].append(wh)
                out['data'][p]['WB'].append(np.mean([wl, wh]))
                out['data'][p]['LCFIT'].append(lcfit)
                pass
            else: startflag = False
            pass
        out['data'][p]['RSTAR'].append(priors['R*']*sscmks['Rsun'])
        out['data'][p]['Hs'].append(Hs)
        out['data'][p]['Teq'] = eqtemp
        # Wavelength re-ordering for Cerberus
        orderme = np.argsort(out['data'][p]['WB'])
        for keytoord in ['ES', 'ESerr', 'WBlow', 'WBup', 'WB']:
            temparr = np.array(out['data'][p][keytoord])
            out['data'][p][keytoord] = temparr[orderme]
            pass
        exospec = True
        out['STATUS'].append(True)
        pass
    if verbose:
        for p in out['data'].keys():
            if 'Teq' in out['data'][p]:
                Teq = str(int(out['data'][p]['Teq']))
                pass
            else: Teq = ''
            vspectrum = np.array(out['data'][p]['ES'])
            specerr = np.array(out['data'][p]['ESerr'])
            specwave = np.array(out['data'][p]['WB'])
            specerr = abs(vspectrum**2 - (vspectrum + specerr)**2)
            vspectrum = vspectrum**2
            Rstar = priors['R*']*sscmks['Rsun']
            Rp = priors[p]['rp']*7.14E7  # m
            Hs = cst.Boltzmann*eqtemp/(mmw*1e-2*(10.**priors[p]['logg']))  # m
            noatm = Rp**2/(Rstar)**2
            rp0hs = np.sqrt(noatm*(Rstar)**2)
            # Smooth spectrum
            binsize = 4
            nspec = int(specwave.size/binsize)
            minspec = np.nanmin(specwave)
            maxspec = np.nanmax(specwave)
            scale = (maxspec - minspec)/(1e0*nspec)
            wavebin = scale*np.arange(nspec) + minspec
            deltabin = np.diff(wavebin)[0]
            cbin = wavebin + deltabin/2e0
            specbin = []
            errbin = []
            for eachbin in cbin:
                select = specwave < (eachbin + deltabin/2e0)
                select = select & (specwave >= (eachbin - deltabin/2e0))
                select = select & np.isfinite(vspectrum)
                if np.sum(np.isfinite(vspectrum[select])) > 0:
                    specbin.append(np.nansum(vspectrum[select]/(specerr[select]**2))/
                                   np.nansum(1./(specerr[select]**2)))
                    errbin.append(np.nanmedian((specerr[select]))/
                                  np.sqrt(np.sum(select)))
                    pass
                else:
                    specbin.append(np.nan)
                    errbin.append(np.nan)
                    pass
                pass
            waveb = np.array(cbin)
            specb = np.array(specbin)
            errb = np.array(errbin)
            myfig, ax0 = plt.subplots(figsize=(8,6))
            plt.title(p+' '+Teq)
            ax0.errorbar(specwave, 1e2*vspectrum,
                         fmt='.', yerr=1e2*specerr, color='lightgray')
            ax0.errorbar(waveb, 1e2*specb,
                         fmt='^', yerr=1e2*errb, color='blue')
            ax0.set_xlabel(str('Wavelength [$\\mu m$]'))
            ax0.set_ylabel(str('$(R_p/R_*)^2$ [%]'))
            if ('Hs' in out['data'][p]) and ('RSTAR' in out['data'][p]):
                rp0hs = np.sqrt(np.nanmedian(vspectrum))
                Hs = out['data'][p]['Hs'][0]
                # Retro compatibility for Hs in [m]
                if Hs > 1: Hs = Hs/(out['data'][p]['RSTAR'][0])
                ax2 = ax0.twinx()
                ax2.set_ylabel('$\\Delta$ [Hs]')
                axmin, axmax = ax0.get_ylim()
                ax2.set_ylim((np.sqrt(1e-2*axmin) - rp0hs)/Hs,
                             (np.sqrt(1e-2*axmax) - rp0hs)/Hs)
                myfig.tight_layout()
                pass
            plt.show()
            pass
        pass
    return exospec
# -------------- -----------------------------------------------------
# -- PYMC3 DETERMINISTIC FUNCTIONS -- --------------------------------
@tco.as_op(itypes=[tt.dscalar, tt.dvector, tt.dscalar,
                   tt.dvector, tt.dvector, tt.dvector], otypes=[tt.dvector])
def orbital(*whiteparams):
    '''
    G. ROUDIER: Orbital model
    '''
    r, atk, icln, avs, aos, aoi = whiteparams
    if ctxt.orbp['inc'] == 9e1: inclination = 9e1
    else: inclination = float(icln)
    out = []
    for i, v in enumerate(ctxt.visits):
        omt = ctxt.time[i]
        if v in ctxt.ttv: omtk = float(atk[ctxt.ttv.index(v)])
        else: omtk = ctxt.tmjd
        omz, _pmph = datcore.time2z(omt, inclination, omtk,
                                    ctxt.smaors, ctxt.period, ctxt.ecc)
        lcout = tldlc(abs(omz), float(r),
                      g1=ctxt.g1[0], g2=ctxt.g2[0], g3=ctxt.g3[0], g4=ctxt.g4[0])
        imout = timlc(omt, ctxt.orbits[i],
                      vslope=float(avs[i]), vitcp=1e0,
                      oslope=float(aos[i]), oitcp=float(aoi[i]))
        out.extend(lcout*imout)
        pass
    return np.array(out)[ctxt.selectfit]

@tco.as_op(itypes=[tt.dscalar,
                   tt.dvector, tt.dvector, tt.dvector], otypes=[tt.dvector])
def nottvfiorbital(*whiteparams):
    '''
    R. ESTRELA: Fixed orbital solution
    '''
    r, avs, aos, aoi = whiteparams
    inclination = ctxt.ginc
    out = []
    for i, v in enumerate(ctxt.visits):
        omt = ctxt.time[i]
        if v in ctxt.ttv: omtk = float(ctxt.gttv[ctxt.ttv.index(v)])
        else: omtk = ctxt.tmjd
        omz, _pmph = datcore.time2z(omt, inclination, omtk,
                                    ctxt.smaors, ctxt.period, ctxt.ecc)
        lcout = tldlc(abs(omz), float(r),
                      g1=ctxt.g1[0], g2=ctxt.g2[0], g3=ctxt.g3[0], g4=ctxt.g4[0])
        imout = timlc(omt, ctxt.orbits[i],
                      vslope=float(avs[i]), vitcp=1e0,
                      oslope=float(aos[i]), oitcp=float(aoi[i]))
        out.extend(lcout*imout)
        pass
    return np.array(out)[ctxt.selectfit]

@tco.as_op(itypes=[tt.dscalar, tt.dvector,
                   tt.dvector, tt.dvector, tt.dvector], otypes=[tt.dvector])
def fiorbital(*whiteparams):
    '''
    G. ROUDIER: Orbital model with fixed inclination
    '''
    r, atk, avs, aos, aoi = whiteparams
    inclination = ctxt.orbp['inc']
    out = []
    for i, v in enumerate(ctxt.visits):
        omt = ctxt.time[i]
        if v in ctxt.ttv: omtk = float(atk[ctxt.ttv.index(v)])
        else: omtk = ctxt.tmjd
        omz, _pmph = datcore.time2z(omt, inclination, omtk,
                                    ctxt.smaors, ctxt.period, ctxt.ecc)
        lcout = tldlc(abs(omz), float(r),
                      g1=ctxt.g1[0], g2=ctxt.g2[0], g3=ctxt.g3[0], g4=ctxt.g4[0])
        imout = timlc(omt, ctxt.orbits[i],
                      vslope=float(avs[i]), vitcp=1e0,
                      oslope=float(aos[i]), oitcp=float(aoi[i]))
        out.extend(lcout*imout)
        pass
    return np.array(out)[ctxt.selectfit]

@tco.as_op(itypes=[tt.dscalar, tt.dvector, tt.dvector, tt.dvector],
           otypes=[tt.dvector])
def lcmodel(*specparams):
    '''
    G. ROUDIER: Spectral light curve model
    '''
    r, avs, aos, aoi = specparams
    allimout = []
    for iv in range(len(ctxt.visits)):
        imout = timlc(ctxt.time[iv], ctxt.orbits[iv],
                      vslope=float(avs[iv]), vitcp=1e0,
                      oslope=float(aos[iv]), oitcp=float(aoi[iv]))
        allimout.extend(imout)
        pass
    out = tldlc(abs(ctxt.allz), float(r), g1=float(ctxt.g1[0]), g2=float(ctxt.g2[0]),
                g3=float(ctxt.g3[0]), g4=float(ctxt.g4[0]))
    out = out*np.array(allimout)
    return out[ctxt.valid]
# ----------------------------------- --------------------------------
# -- BINNING FUNCTION -- ---------------------------------------------
def binnagem(t, nbins):
    '''
    R. ESTRELA: Binning the wavelength template
    '''
    tmax = t[0][-1]
    tmin = t[0][0]
    tbin = (tmax-tmin)*np.arange(nbins+1)/nbins
    tbin = tbin + tmin
    lower = np.resize(tbin, len(tbin)-1)
    tmid = lower + 0.5*np.diff(tbin)
    return tmid, lower
# ---------------------- ---------------------------------------------
# -- FAST SPECTRUM -- ------------------------------------------------
def fastspec(fin, nrm, wht, ext, selftype,
             chainlen=int(1e4), p=None, verbose=False):
    '''
    G. ROUDIER: Exoplanet spectrum fast recovery for prior setup
    '''
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
    smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
    ttrdur = np.arcsin((1e0+rpors)/smaors)
    trdura = priors[p]['period']*ttrdur/np.pi
    vrange = nrm['data'][p]['vrange']
    wave = nrm['data'][p]['wavet']
    waves = nrm['data'][p]['wave']
    nspec = nrm['data'][p]['nspec']
    photnoise = nrm['data'][p]['photnoise']
    if 'G750' in ext:
        wave, _trash = binnagem(wave, 100)
        wave = np.resize(wave,(1,250))
        pass
    if 'G430' in ext:
        wave, _trash = binnagem(wave, 25)
        wave = np.resize(wave,(1,121))
        pass
    time = nrm['data'][p]['time']
    visits = nrm['data'][p]['visits']
    orbits = nrm['data'][p]['orbits']
    disp = nrm['data'][p]['dispersion']
    im = wht['data'][p]['postim']
    allz = wht['data'][p]['postsep']
    whiterprs = np.nanmedian(wht['data'][p]['mctrace']['rprs'])
    allwave = []
    allspec = []
    allim = []
    allpnoise = []
    alldisp = []
    for w, s, i, n, d in zip(waves, nspec, im, photnoise, disp):
        allwave.extend(w)
        allspec.extend(s)
        allim.extend(i)
        allpnoise.extend(n)
        alldisp.extend(d)
        pass
    alldisp = np.array(alldisp)
    allim = np.array(allim)
    allz = np.array(allz)
    if 'STIS' in ext:
        disp = np.median([np.median(np.diff(w)) for w in wave])
        nbin = np.min([len(w) for w in wave])
        wavel = [np.min(w) for w in wave]
        wavec = np.arange(nbin)*disp + np.mean([np.max(wavel), np.min(wavel)])
        lwavec = wavec - disp/2e0
        hwavec = wavec + disp/2e0
        pass
    # MULTI VISITS COMMON WAVELENGTH GRID ------------------------------------------------
    if 'WFC3' in ext:
        wavec, _t = tplbuild(allspec, allwave, vrange, alldisp*1e-4, medest=True)
        wavec = np.array(wavec)
        temp = [np.diff(wavec)[0]]
        temp.extend(np.diff(wavec))
        lwavec = wavec - np.array(temp)/2e0
        temp = list(np.diff(wavec))
        temp.append(np.diff(wavec)[-1])
        hwavec = wavec + np.array(temp)/2e0
        pass
    # EXCLUDE PARTIAL LIGHT CURVES AT THE EDGES ------------------------------------------
    wavec = wavec[1:-2]
    lwavec = lwavec[1:-2]
    hwavec = hwavec[1:-2]
    # EXCLUDE ALL NAN CHANNELS -----------------------------------------------------------
    allnanc = []
    for wl, wh in zip(lwavec, hwavec):
        select = [(w > wl) & (w < wh) for w in allwave]
        if 'STIS' in ext:
            data = np.array([np.nanmean(d[s]) for d, s in zip(allspec, select)])
            pass
        else: data = np.array([np.median(d[s]) for d, s in zip(allspec, select)])
        if np.all(~np.isfinite(data)): allnanc.append(True)
        else: allnanc.append(False)
        pass
    lwavec = [lwv for lwv, lln in zip(lwavec, allnanc) if not lln]
    hwavec = [hwv for hwv, lln in zip(hwavec, allnanc) if not lln]
    # LOOP OVER WAVELENGTH BINS ----------------------------------------------------------
    ES = []
    ESerr = []
    WB = []
    for wl, wh in zip(lwavec, hwavec):
        select = [(w > wl) & (w < wh) for w in allwave]
        if 'STIS' in ext:
            data = np.array([np.nanmean(d[s]) for d, s in zip(allspec, select)])
            dnoise = np.array([(1e0/np.sum(s))*np.sqrt(np.nansum((n[s])**2))
                               for n, s in zip(allpnoise, select)])
            pass
        else:
            data = np.array([np.nanmean(d[s])
                             for d, s in zip(allspec, select)])
            dnoise = np.array([np.nanmedian(n[s])/np.sqrt(np.nansum(s))
                               for n, s in zip(allpnoise, select)])
            pass
        valid = np.isfinite(data)
        if selftype in ['transit']:
            bld = createldgrid([wl], [wh], priors, segmentation=int(10))
            g1, g2, g3, g4 = bld['LD']
            pass
        else: g1, g2, g3, g4 = [[0], [0], [0], [0]]
        # renorm = np.nanmean(data[abs(allz) > (1e0 + whiterprs)])
        # data /= renorm
        # dnoise /= renorm
        # PRIORS -------------------------------------------------------------------------
        sscmks = syscore.ssconstants(mks=True)
        eqtemp = priors['T*']*np.sqrt(priors['R*']*sscmks['Rsun/AU']/
                                      (2.*priors[p]['sma']))
        pgrid = np.arange(np.log(10.)-15., np.log(10.)+15./100, 15./99)
        pgrid = np.exp(pgrid)
        pressure = pgrid[::-1]
        mixratio, fH2, fHe = crbutil.crbce(pressure, eqtemp)
        mmw, fH2, fHe = crbutil.getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
        mmw = mmw*cst.m_p  # [kg]
        Hs = cst.Boltzmann*eqtemp/(mmw*1e-2*(10.**priors[p]['logg']))  # [m]
        Hs = Hs/(priors['R*']*sscmks['Rsun'])
        tauvs = 1e0/((1e-2/trdura)**2)
        ootstd = np.nanstd(data[abs(allz) > (1e0 + whiterprs)])
        tauvi = 1e0/(ootstd**2)
        nodes = []
        tauwbdata = 1e0/dnoise**2
        # UPDATE GLOBALS -----------------------------------------------------------------
        shapevis = 2
        if shapevis < len(visits): shapevis = len(visits)
        ctxtupdt(allz=allz, g1=g1, g2=g2, g3=g3, g4=g4,
                 orbits=orbits, smaors=smaors, time=time, valid=valid, visits=visits)
        # PYMC3 --------------------------------------------------------------------------
        with pm.Model():
            lowstart = whiterprs - 5e0*Hs
            lowstart = max(lowstart, 0)
            upstart = whiterprs + 5e0*Hs
            rprs = pm.Uniform('rprs', lower=lowstart, upper=upstart)
            allvslope = pm.TruncatedNormal('vslope', mu=0e0, tau=tauvs,
                                           lower=-3e-2/trdura,
                                           upper=3e-2/trdura, shape=shapevis)
            alloslope = pm.Normal('oslope', mu=0, tau=tauvs, shape=shapevis)
            alloitcp = pm.Normal('oitcp', mu=1e0, tau=tauvi, shape=shapevis)
            nodes.append(rprs)
            nodes.append(allvslope)
            nodes.append(alloslope)
            nodes.append(alloitcp)
            _wbdata = pm.Normal('wbdata', mu=lcmodel(*nodes),
                                tau=np.nanmedian(tauwbdata[valid]),
                                observed=data[valid])
            trace = pm.sample(chainlen, cores=4, tune=int(chainlen/3),
                              compute_convergence_checks=False, step=pm.Metropolis(),
                              progressbar=verbose)
            pass
        ES.append(np.nanmedian(trace['rprs']))
        ESerr.append(np.nanstd(trace['rprs']))
        WB.append(np.mean([wl, wh]))
        pass
    ES = np.array(ES)
    ESerr = np.array(ESerr)
    WB = np.array(WB)
    if verbose:
        vspectrum = ES.copy()
        specerr = ESerr.copy()
        specwave = WB.copy()
        specerr = abs(vspectrum**2 - (vspectrum + specerr)**2)
        vspectrum = vspectrum**2
        Rstar = priors['R*']*sscmks['Rsun']
        Rp = priors[p]['rp']*7.14E7  # m
        Hs = cst.Boltzmann*eqtemp/(mmw*1e-2*(10.**priors[p]['logg']))  # m
        noatm = Rp**2/(Rstar)**2
        rp0hs = np.sqrt(noatm*(Rstar)**2)
        _fig, ax0 = plt.subplots(figsize=(10,6))
        ax0.errorbar(specwave, 1e2*vspectrum, fmt='.', yerr=1e2*specerr)
        ax0.set_xlabel(str('Wavelength [$\\mu m$]'))
        ax0.set_ylabel(str('$(R_p/R_*)^2$ [%]'))
        ax1 = ax0.twinx()
        yaxmin, yaxmax = ax0.get_ylim()
        ax2min = (np.sqrt(1e-2*yaxmin)*Rstar - rp0hs)/Hs
        ax2max = (np.sqrt(1e-2*yaxmax)*Rstar - rp0hs)/Hs
        ax1.set_ylabel('Transit Depth Modulation [Hs]')
        ax1.set_ylim(ax2min, ax2max)
        plt.show()
        pass
    priorspec = ES
    # alpha > 1: Increase width, alpha < 1: Decrease width
    # decrease width by half if no modulation detected
    alphanum = np.nanmedian(np.diff(ES))
    if alphanum < np.nanmedian(ESerr)/2e0: alphanum = np.nanmedian(ESerr)/2e0
    alpha = alphanum/np.nanmedian(ESerr)
    return priorspec, alpha
# ------------------- ------------------------------------------------
# ----------------- --------------------------------------------------
def corner(xs, bins=20, arange=None, weights=None, color="k", hist_bin_factor=1,
           smooth=None, smooth1d=None, levels=None,
           labels=None, label_kwargs=None,
           titles=None, title_kwargs=None,
           truths=None, truth_color="#4682b4",
           scale_hist=False, quantiles=None, verbose=False, fig=None,
           max_n_ticks=5, top_ticks=False, use_math_text=False, reverse=False,
           hist_kwargs=None, **hist2d_kwargs):
    '''
    A fork of corner.py but altered to support a scatter function when plotting
    this allowing the data points to be color coded to a likehood or chi-square value.
    '''
    if quantiles is None: quantiles = []
    if title_kwargs is None: title_kwargs = {}
    if label_kwargs is None: label_kwargs = {}
    if titles is None: titles = []
    if levels is None: levels = [1]
    # Try filling in labels from pandas.DataFrame columns.
    if labels is None:
        try:
            labels = xs.columns
        except AttributeError:
            pass
        pass

    # Deal with 1D sample lists.
    xs = np.atleast_1d(xs)
    if len(xs.shape) == 1:
        xs = np.atleast_2d(xs)
    else:
        assert len(xs.shape) == 2, "The input sample array must be 1- or 2-D."
        xs = xs.T
    assert xs.shape[0] <= xs.shape[1], "I don't believe that you want more " \
                                       "dimensions than samples!"

    # Parse the weight array.
    if weights is not None:
        weights = np.asarray(weights)
        if weights.ndim != 1:
            raise ValueError("Weights must be 1-D")
        if xs.shape[1] != weights.shape[0]:
            raise ValueError("Lengths of weights must match number of samples")

    # Parse the parameter ranges.
    if arange is None:
        if "extents" in hist2d_kwargs:
            logging.info("Deprecated keyword argument 'extents'. "
                         "Use 'range' instead.")
            arange = hist2d_kwargs.pop("extents")
        else:
            arange = [[x.min(), x.max()] for x in xs]
            # Check for parameters that never change.
            m = np.array([e[0] == e[1] for e in arange], dtype=bool)
            if np.any(m):
                # GMR: I m not touching this. Where does this code come from?
                # pylint: disable=consider-using-f-string
                raise ValueError(("It looks like the parameter(s) in "
                                  "column(s) {0} have no dynamic range. "
                                  "Please provide a `range` argument.")
                                 .format(", ".join(map(
                                     "{0}".format, np.arange(len(m))[m]))))

    else:
        # If any of the extents are percentiles, convert them to ranges.
        # Also make sure it's a normal list.
        arange = list(arange)
        for i, _ in enumerate(arange):
            try:
                _, _ = arange[i]
            except TypeError:
                q = [0.5 - 0.5*arange[i], 0.5 + 0.5*arange[i]]
                arange[i] = quantile(xs[i], q, weights=weights)

    if len(arange) != xs.shape[0]:
        raise ValueError("Dimension mismatch between samples and range")

    # Parse the bin specifications.
    try:
        bins = [int(bins) for _ in arange]
    except TypeError as dood:
        if len(bins) != len(arange):
            raise ValueError("Dimension mismatch between bins and arange") from dood
        pass
    try:
        hist_bin_factor = [float(hist_bin_factor) for _ in arange]
    except TypeError as dood:
        if len(hist_bin_factor) != len(range):
            raise ValueError("Dimension mismatch between hist_bin_factor and "
                             "range") from dood

    # Some magic numbers for pretty axis layout.
    K = len(xs)
    factor = 2.0           # size of one side of one panel
    if reverse:
        lbdim = 0.2 * factor   # size of left/bottom margin
        trdim = 0.5 * factor   # size of top/right margin
    else:
        lbdim = 0.5 * factor   # size of left/bottom margin
        trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim

    # Create a new figure if one wasn't provided.
    if fig is None:
        fig, axes = plt.subplots(K, K, figsize=(dim, dim))
    else:
        try: axes = np.array(fig.axes).reshape((K, K))
        except:
            # GMR: Wildcard except nothing I can or want do for this
            # pylint: disable=raise-missing-from
            raise ValueError(f"""Provided figure has {len(fig.axes)} axes, but data has
                             dimensions K={K}""")
        pass

    # Format the figure.
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    # Set up the default histogram keywords.
    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs["color"] = hist_kwargs.get("color", color)
    if smooth1d is None:
        hist_kwargs["histtype"] = hist_kwargs.get("histtype", "step")

    for i, x in enumerate(xs):
        # Deal with masked arrays.
        if hasattr(x, "compressed"):
            x = x.compressed()

        if np.shape(xs)[0] == 1:
            ax = axes
        else:
            if reverse:
                ax = axes[K-i-1, K-i-1]
            else:
                ax = axes[i, i]
        # Plot the histograms.
        if smooth1d is None:
            bins_1d = int(max(1, np.round(hist_bin_factor[i] * bins[i])))
            n, _, _ = ax.hist(x, bins=bins_1d, weights=weights,
                              range=np.sort(arange[i]), **hist_kwargs)
        else:
            if gaussian_filter is None:
                raise ImportError("Please install scipy for smoothing")
            n, b = np.histogram(x, bins=bins[i], weights=weights,
                                range=np.sort(arange[i]))
            n = gaussian_filter(n, smooth1d)
            x0 = np.array(list(zip(b[:-1], b[1:]))).flatten()
            y0 = np.array(list(zip(n, n))).flatten()
            ax.plot(x0, y0, **hist_kwargs)

        if truths is not None and truths[i] is not None:
            ax.axvline(truths[i], color=truth_color)

        # Plot quantiles if wanted.
        qlen = len(quantiles)
        if qlen > 0:
            qvalues = quantile(x, quantiles, weights=weights)
            for q in qvalues:
                ax.axvline(q, ls="dashed", color=color)

            if verbose:
                print("Quantiles:")
                # print([item for item in zip(quantiles, qvalues)])
                print(list(zip(quantiles, qvalues)))

        # pylint: disable=len-as-condition
        if len(titles):
            ax.set_title(titles[i], **title_kwargs)

        # Set up the axes.
        ax.set_xlim(arange[i])
        if scale_hist:
            maxn = np.max(n)
            ax.set_ylim(-0.1 * maxn, 1.1 * maxn)
        else:
            ax.set_ylim(0, 1.1 * np.max(n))
        ax.set_yticklabels([])
        if max_n_ticks == 0:
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
        else:
            ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune="lower"))
            ax.yaxis.set_major_locator(NullLocator())

        if i < K - 1:
            if top_ticks:
                ax.xaxis.set_ticks_position("top")
                _ = [l.set_rotation(45) for l in ax.get_xticklabels()]
            else:
                ax.set_xticklabels([])
        else:
            if reverse:
                ax.xaxis.tick_top()
            _ = [l.set_rotation(45) for l in ax.get_xticklabels()]
            if labels is not None:
                if reverse:
                    ax.set_title(labels[i], y=1.25, **label_kwargs)
                else:
                    ax.set_xlabel(labels[i], **label_kwargs)

            # use MathText for axes ticks
            ax.xaxis.set_major_formatter(
                ScalarFormatter(useMathText=use_math_text))

        for j, y in enumerate(xs):
            if np.shape(xs)[0] == 1:
                ax = axes
            else:
                if reverse:
                    ax = axes[K-i-1, K-j-1]
                else:
                    ax = axes[i, j]
            if j > i:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            if j == i:
                continue

            # Deal with masked arrays.
            if hasattr(y, "compressed"):
                y = y.compressed()

            # pylint: disable=unexpected-keyword-arg
            hist2d(y, x, ax=ax, range=[arange[j], arange[i]], weights=weights,
                   smooth=smooth, bins=[bins[j], bins[i]], levels=levels,
                   **hist2d_kwargs)

            if truths is not None:
                if truths[i] is not None and truths[j] is not None:
                    ax.plot(truths[j], truths[i], "s", color=truth_color)
                if truths[j] is not None:
                    ax.axvline(truths[j], color=truth_color)
                if truths[i] is not None:
                    ax.axhline(truths[i], color=truth_color)

            if max_n_ticks == 0:
                ax.xaxis.set_major_locator(NullLocator())
                ax.yaxis.set_major_locator(NullLocator())
            else:
                ax.xaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))
                ax.yaxis.set_major_locator(MaxNLocator(max_n_ticks,
                                                       prune="lower"))

            if i < K - 1:
                ax.set_xticklabels([])
            else:
                if reverse:
                    ax.xaxis.tick_top()
                for l in ax.get_xticklabels():
                    l.set_rotation(45)
                if labels is not None:
                    ax.set_xlabel(labels[j], **label_kwargs)
                    if reverse:
                        ax.xaxis.set_label_coords(0.5, 1.4)
                    else:
                        ax.xaxis.set_label_coords(0.5, -0.3)

                # use MathText for axes ticks
                ax.xaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

            if j > 0:
                ax.set_yticklabels([])
            else:
                if reverse:
                    ax.yaxis.tick_right()
                for l in ax.get_yticklabels():
                    l.set_rotation(45)
                if labels is not None:
                    if reverse:
                        ax.set_ylabel(labels[i], rotation=-90, **label_kwargs)
                        ax.yaxis.set_label_coords(1.3, 0.5)
                    else:
                        ax.set_ylabel(labels[i], **label_kwargs)
                        ax.yaxis.set_label_coords(-0.3, 0.5)

                # use MathText for axes ticks
                ax.yaxis.set_major_formatter(
                    ScalarFormatter(useMathText=use_math_text))

    return fig

def quantile(x, q, weights=None):
    """
    Compute sample quantiles with support for weighted samples.

    Note
    ----
    When ``weights`` is ``None``, this method simply calls numpy's percentile
    function with the values of ``q`` multiplied by 100.

    Parameters
    ----------
    x : array_like[nsamples,]
       The samples.

    q : array_like[nquantiles,]
       The list of quantiles to compute. These should all be in the range
       ``[0, 1]``.

    weights : Optional[array_like[nsamples,]]
        An optional weight corresponding to each sample. These

    Returns
    -------
    quantiles : array_like[nquantiles,]
        The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError
        For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch
        between ``x`` and ``weights``.

    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None: return np.percentile(x, list(100.0 * q))

    weights = np.atleast_1d(weights)
    if len(x) != len(weights):
        raise ValueError("Dimension mismatch: len(weights) != len(x)")
    idx = np.argsort(x)
    sw = weights[idx]
    cdf = np.cumsum(sw)[:-1]
    cdf /= cdf[-1]
    cdf = np.append(0, cdf)
    return np.interp(q, cdf, x[idx]).tolist()

# pylint: disable=unused-argument
def hist2d(x, y, arange=None, levels=(2,),
           ax=None, plot_datapoints=True, plot_contours=True,
           contour_kwargs=None, data_kwargs=None, **kwargs):
    '''hist2d ds'''
    if ax is None:
        ax = plt.gca()

    if plot_datapoints:
        if data_kwargs is None: data_kwargs = {}
        data_kwargs["s"] = data_kwargs.get("s", 2.0)
        data_kwargs["alpha"] = data_kwargs.get("alpha", 0.2)
        ax.scatter(x, y, marker="o", zorder=-1, rasterized=True, **data_kwargs)

    # Plot the contour edge colors.
    if plot_contours:
        if contour_kwargs is None: contour_kwargs = {}

        # mask data in range + chi2
        maskx = (x > arange[0][0]) & (x < arange[0][1])
        masky = (y > arange[1][0]) & (y < arange[1][1])
        mask = maskx & masky & (data_kwargs['c'] < data_kwargs['vmax']*1.2)

        # approx posterior + smooth
        xg, yg = np.meshgrid(np.linspace(x[mask].min(),x[mask].max(),50), np.linspace(y[mask].min(),y[mask].max(),50))
        cg = griddata(np.vstack([x[mask],y[mask]]).T, data_kwargs['c'][mask], (xg,yg), method='nearest', rescale=True)
        scg = gaussian_filter(cg,sigma=2)

        # plot
        ax.contour(xg, yg, scg*np.nanmin(cg)/np.nanmin(scg), levels, **contour_kwargs, vmin=data_kwargs['vmin'], vmax=data_kwargs['vmax'])
    ax.set_xlim(arange[0])
    ax.set_ylim(arange[1])

#######################################################
def eclipse(times, values):
    '''eclipse ds'''
    tme = eclipse_mid_time(values['per'], values['ars'], values['ecc'], values['inc'], values['omega'], values['tmid'])
    model = pytransit([0,0,0,0],
                      values['rprs']*values['fpfs']**0.5, values['per'], values['ars'],
                      values['ecc'], values['inc'], values['omega']+180,
                      tme, times, method='claret', precision=3)
    return model

def brightness(time, values):
    '''brightness ds'''
    # compute mid-eclipse time
    tme = eclipse_mid_time(values['per'], values['ars'], values['ecc'], values['inc'], values['omega'], values['tmid'])

    # compute phase based on mid-eclipse
    phase = (time - tme)/values['per']

    # brightness amplitude variation
    bdata = values['c1']*np.cos(2*np.pi*phase) + values['c2']*np.sin(2*np.pi*phase) + values['c3']*np.cos(4*np.pi*phase) + values['c4']*np.sin(4*np.pi*phase)

    # offset so eclipse is around 1 in norm flux
    c0 = values['fpfs']*values['rprs']**2 - (values['c1'] + values['c3'])
    return 1+c0+bdata

def phasecurve(time, values):
    '''phasecurve ds'''
    # transit time series
    tdata = transit(time, values)

    # eclipse (similar to transit but with 0 Ld and 180+omega)
    edata = eclipse(time, values)

    # brightness variation
    bdata = brightness(time, values)

    # combine models
    pdata = tdata*bdata*edata

    # mask in-eclipse data
    emask = ~np.floor(edata).astype(bool)

    # offset data to be at 1 (TODO inspect this line)
    pdata[emask] = edata[emask]+values['fpfs']*values['rprs']**2
    return pdata

def time_bin(time, flux, dt=1./(60*24)):
    '''average data into bins of dt from start to finish'''
    bins = int(np.floor((max(time) - min(time))/dt))
    bflux = np.zeros(bins)
    btime = np.zeros(bins)
    bstds = np.zeros(bins)
    for i in range(bins):
        mask = (time >= (min(time)+i*dt)) & (time < (min(time)+(i+1)*dt))
        if mask.sum() > 0:
            bflux[i] = np.nanmean(flux[mask])
            btime[i] = np.nanmean(time[mask])
            bstds[i] = np.nanstd(flux[mask])/(1+mask.sum())**0.5
    zmask = (bflux==0) | (btime==0) | np.isnan(bflux) | np.isnan(btime)
    return btime[~zmask], bflux[~zmask], bstds[~zmask]

##########################################################
# phasecurve, eclipse, and transit fitting algorithm with
# nearest neighbor detrending
class pc_fitter():
    '''pc_fitter'''
    # pylint: disable=too-many-instance-attributes
    def __init__(self, time, data, dataerr, prior, bounds, syspars, neighbors=100, mode='ns', verbose=False):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = copy.deepcopy(prior)
        self.bounds = bounds
        self.syspars = syspars
        self.neighbors = neighbors
        self.verbose = verbose

        if mode == 'ns':
            self.fit_nested()
        else:
            self.fit_lm()

    def fit_lm(self):
        '''fit_lm ds'''
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])

        # trim data around predicted transit/eclipse time
        self.gw, self.nearest = gaussian_weights(self.syspars, neighbors=self.neighbors)

        def lc2min(pars):
            # pylint: disable=invalid-unary-operand-type
            for i, _ in enumerate(pars):
                self.prior[freekeys[i]] = pars[i]
            lightcurve = phasecurve(self.time, self.prior)
            detrended = self.data/lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            return ((self.data-model)/self.dataerr)**2

        res = least_squares(lc2min, x0=[self.prior[k] for k in freekeys],
                            bounds=[boundarray[:,0], boundarray[:,1]], jac='3-point',
                            loss='linear', method='dogbox', xtol=None, ftol=1e-5,
                            tr_options='exact', verbose=True)

        self.parameters = copy.deepcopy(self.prior)
        self.errors = {}

        for i,k in enumerate(freekeys):
            self.parameters[k] = res.x[i]
            self.errors[k] = 0

        self.create_fit_variables()

    def fit_nested(self):
        '''fit_nested ds'''
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)

        # trim data around predicted transit/eclipse time
        self.gw, self.nearest = gaussian_weights(self.syspars, neighbors=self.neighbors)

        def lc2min_transit(pars):
            '''lc2min_transit ds'''
            # pylint: disable=invalid-unary-operand-type
            for i, _ in enumerate(pars):
                self.prior[freekeys[i]] = pars[i]
            lightcurve = transit(self.time, self.prior)
            detrended = self.data/lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            return -np.sum(((self.data-model)/self.dataerr)**2)

        def lc2min_phasecurve(pars):
            '''lc2min_phasecurve ds'''
            # pylint: disable=invalid-unary-operand-type
            for i, _ in enumerate(pars):
                self.prior[freekeys[i]] = pars[i]
            lightcurve = phasecurve(self.time, self.prior)
            detrended = self.data/lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            return -np.sum(((self.data-model)/self.dataerr)**2)

        def prior_transform_basic(upars):
            '''prior_transform_basic ds'''
            return (boundarray[:,0] + bounddiff*upars)

        def prior_transform_phasecurve(upars):
            '''prior_transform_phasecurve ds'''
            vals = (boundarray[:,0] + bounddiff*upars)

            # set limits of phase amplitude to be less than eclipse depth or user bound
            edepth = vals[freekeys.index('rprs')]**2 * vals[freekeys.index('fpfs')]
            for k in ['c1','c2']:
                if k in freekeys:
                    # conditional prior needed to conserve energy
                    if k == 'c1':
                        ki = freekeys.index(k)
                        vals[ki] = upars[ki]*0.4*edepth+0.1*edepth
                    if k == 'c2':
                        ki = freekeys.index(k)
                        vals[ki] = upars[ki]*0.25*edepth - 0.125*edepth
            return vals

        if self.verbose:
            if 'fpfs' in freekeys:
                self.results = ReactiveNestedSampler(freekeys, lc2min_phasecurve, prior_transform_phasecurve).run(max_ncalls=5e5)
            else:
                self.results = ReactiveNestedSampler(freekeys, lc2min_transit, prior_transform_basic).run(max_ncalls=5e5)
        else:
            if 'fpfs' in freekeys:
                self.results = ReactiveNestedSampler(freekeys, lc2min_phasecurve, prior_transform_phasecurve).run(max_ncalls=5e5, show_status=self.verbose, viz_callback=self.verbose)
            else:
                self.results = ReactiveNestedSampler(freekeys, lc2min_transit, prior_transform_basic).run(max_ncalls=5e5, show_status=self.verbose, viz_callback=self.verbose)

        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)

        for i, key in enumerate(freekeys):

            self.parameters[key] = self.results['maximum_likelihood']['point'][i]
            self.errors[key] = self.results['posterior']['stdev'][i]
            self.quantiles[key] = [
                self.results['posterior']['errlo'][i],
                self.results['posterior']['errup'][i]]

        # self.results['maximum_likelihood']
        self.create_fit_variables()

    def create_fit_variables(self):
        '''create_fit_variables ds'''
        # pylint: disable=attribute-defined-outside-init
        self.phase = (self.time - self.parameters['tmid']) / self.parameters['per']
        self.transit = phasecurve(self.time, self.parameters)
        detrended = self.data / self.transit
        self.wf = weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.detrended = self.data/self.wf
        self.detrendederr = self.dataerr
        self.residuals = self.data - self.model
        self.chi2 = np.sum(self.residuals**2/self.dataerr**2)
        self.bic = len(self.bounds) * np.log(len(self.time)) - 2*np.log(self.chi2)

    def plot_bestfit(self, bin_dt=10./(60*24), zoom=False, phase=True):
        '''plot_bestfit ds'''
        f = plt.figure(figsize=(12,7))
        # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        bt, bf, _ = time_bin(self.time, self.detrended, bin_dt)
        bp = (bt-self.parameters['tmid'])/self.parameters['per']

        if phase:
            axs[0].plot(bp,bf,'co',alpha=0.5,zorder=2)
            axs[0].plot(self.phase, self.transit, 'r-', zorder=3)
            axs[0].set_xlim([min(self.phase), max(self.phase)])
            axs[0].set_xlabel("Phase ")
        else:
            axs[0].plot(bt,bf,'co',alpha=0.5,zorder=2)
            axs[0].plot(self.time, self.transit, 'r-', zorder=3)
            axs[0].set_xlim([min(self.time), max(self.time)])
            axs[0].set_xlabel("Time [day]")

        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')

        if zoom:
            axs[0].set_ylim([1-1.25*self.parameters['rprs']**2, 1+0.5*self.parameters['rprs']**2])
        else:
            if phase:
                axs[0].errorbar(self.phase, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.025)
            else:
                axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.025)

        bt, br, _ = time_bin(self.time, self.residuals/np.median(self.data)*1e6, bin_dt)
        bp = (bt-self.parameters['tmid'])/self.parameters['per']

        if phase:
            axs[1].plot(self.phase, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.15, label=fr'$\sigma$ = {np.std(self.residuals/np.median(self.data)*1e6):.0f} ppm')
            axs[1].plot(bp,br,'c.',alpha=0.5,zorder=2,
                        label=fr'$\sigma$ = {np.std(br):.0f} ppm')
            axs[1].set_xlim([min(self.phase), max(self.phase)])
            axs[1].set_xlabel("Phase")
        else:
            axs[1].plot(self.time, self.residuals/np.median(self.data)*1e6, 'k.',
                        alpha=0.15,
                        label=fr'$\sigma$ = {np.std(self.residuals/np.median(self.data)*1e6):.0f} ppm')
            axs[1].plot(bt,br,'c.',alpha=0.5,zorder=2,
                        label=fr'$\sigma$ = {np.std(br):.0f} ppm')
            axs[1].set_xlim([min(self.time), max(self.time)])
            axs[1].set_xlabel("Time [day]")

        axs[1].legend(loc='best')
        axs[1].set_ylabel("Residuals [ppm]")
        axs[1].grid(True,ls='--')
        plt.tight_layout()
        return f,axs

    def plot_posterior(self):
        '''plot_posterior ds'''
        ranges = []
        mask1 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        mask2 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        mask3 = np.ones(len(self.results['weighted_samples']['logl']),dtype=bool)
        titles = []
        labels= []
        flabels = {
            'rprs':r'R$_{p}$/R$_{s}$',
            'tmid':r'T$_{mid}$',
            'ars':r'a/R$_{s}$',
            'inc':r'I',
            'u1':r'u$_1$',
            'fpfs':r'F$_{p}$/F$_{s}$',
            'omega':r'$\omega$',
            'ecc':r'$e$',
            'c0':r'$c_0$',
            'c1':r'$c_1$',
            'c2':r'$c_2$',
            'c3':r'$c_3$',
            'c4':r'$c_4$',
            'a0':r'$a_0$',
            'a1':r'$a_1$',
            'a2':r'$a_2$'
        }
        # constrain plots to +/- 4 sigma and estimate sigma levels
        for i, key in enumerate(self.quantiles):
            titles.append(f"{self.parameters[key]:.5f} +- {self.errors[key]:.5f}")

            if key == 'fpfs':
                ranges.append([
                    self.parameters[key] - 3*self.errors[key],
                    self.parameters[key] + 3*self.errors[key]
                ])
            else:
                ranges.append([
                    self.parameters[key] - 4*self.errors[key],
                    self.parameters[key] + 4*self.errors[key]
                ])

            mask3 = mask3 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - 3*self.errors[key])) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + 3*self.errors[key]))

            mask1 = mask1 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - self.errors[key])) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + self.errors[key]))

            mask2 = mask2 & (self.results['weighted_samples']['points'][:,i] > (self.parameters[key] - 2*self.errors[key])) & \
                (self.results['weighted_samples']['points'][:,i] < (self.parameters[key] + 2*self.errors[key]))

            labels.append(flabels.get(key, key))

        chi2 = self.results['weighted_samples']['logl']*-2
        fig = corner(self.results['weighted_samples']['points'],
                     labels=labels,
                     bins=int(np.sqrt(self.results['samples'].shape[0])),
                     range=ranges,
                     plot_contours=True,
                     levels=[chi2[mask1].max(), chi2[mask2].max(), chi2[mask3].max()],
                     plot_density=False,
                     titles=titles,
                     data_kwargs={
                         'c':chi2,
                         'vmin':np.percentile(chi2[mask3],1),
                         'vmax':np.percentile(chi2[mask3],99),
                         'cmap':'viridis'
                     },
                     label_kwargs={
                         'labelpad':15,
                     },
                     hist_kwargs={
                         'color':'black',
                     })
        return fig, None

    def plot_btempcurve(self, bandpass='IRAC 3.6um'):
        '''plot_btempcurve ds'''
        fig = plt.figure(figsize=(13,7))
        ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        phase = (self.time-self.parameters['tmid'])/self.parameters['per']
        bin_dt = 10./24./60.
        bt, bf, _ = time_bin(self.time, self.detrended, bin_dt)
        bp = (bt-self.parameters['tmid'])/self.parameters['per']
        bt, br, _ = time_bin(self.time, self.residuals, bin_dt)

        bcurve = brightness(bt,self.parameters)
        ogfpfs = self.parameters['fpfs']
        tbcurve = np.ones(bcurve.shape)
        for i,bc in enumerate(bcurve):
            self.parameters['fpfs'] = max((bc-1)/self.parameters['rprs']**2, 0.00001)
            tbcurve[i] = brightnessTemp(self.parameters,bandpass)
        self.parameters['fpfs'] = ogfpfs

        # residuals
        axs[1].plot(phase, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.15, label=fr'$\sigma$ = {np.std(self.residuals/np.median(self.data)*1e6):.0f} ppm')

        axs[1].plot(bp,1e6*br/np.median(self.data),'w.',zorder=2,
                    label=fr'$\sigma$ = {np.std(1e6*br/np.median(self.data)):.0f} ppm')

        axs[1].set_xlim([min(phase), max(phase)])
        axs[1].set_xlabel("Phase")
        axs[1].legend(loc='best')
        axs[1].set_ylabel("Residuals [ppm]")
        axs[1].grid(True,ls='--')

        axs[0].errorbar(phase, self.detrended,
                        yerr=np.std(self.residuals)/np.median(self.data), ls='none',
                        marker='.', color='black', alpha=0.1, zorder=1)

        # map color to equilibrium temperature
        im = axs[0].scatter(bp,bf,marker='o',c=tbcurve,vmin=500,vmax=2750,cmap='jet',
                            zorder=2, s=20)
        cbar = plt.colorbar(im)
        cbar.ax.set_xlabel("B. Temp. [K]")

        axs[0].plot(phase, self.transit, 'w--', zorder=3)
        axs[0].set_xlim([min(phase), max(phase)])
        axs[0].set_xlabel("Phase ")

        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')
        axs[0].set_ylim([0.955,1.03])

        plt.tight_layout()
        return fig, axs

    def plot_pixelmap(self, title='', savedir=None):
        '''plot_pixelmap ds'''
        fig,ax = plt.subplots(1,figsize=(8.5,7))
        xcent = self.syspars[:,0]  # weighted flux x-cntroid
        ycent = self.syspars[:,1]  # weighted flux y-centroid
        npp = self.syspars[:,2]  # noise pixel parameter
        normpp = (npp-npp.min())/(npp.max() - npp.min())  # norm btwn 0-1
        normpp *= 20
        normpp += 20
        im = ax.scatter(
            xcent,
            ycent,
            c=self.wf/np.median(self.wf),
            marker='.',
            vmin=0.99,
            vmax=1.01,
            alpha=0.5,
            cmap='jet',
            s=normpp
        )
        ax.set_xlim([
            np.median(xcent)-3*np.std(xcent),
            np.median(xcent)+3*np.std(xcent)
        ])
        ax.set_ylim([
            np.median(ycent)-3*np.std(ycent),
            np.median(ycent)+3*np.std(ycent)
        ])

        ax.set_title(title,fontsize=14)
        ax.set_xlabel('X-Centroid [px]',fontsize=14)
        ax.set_ylabel('Y-Centroid [px]',fontsize=14)
        cbar = fig.colorbar(im)
        cbar.set_label('Relative Pixel Response',fontsize=14,rotation=270,labelpad=15)

        plt.tight_layout()
        if savedir:
            plt.savefig(savedir+title+".png")
            plt.close()
        return fig,ax
    pass

def brightnessTemp(priors,f='IRAC 3.6um'):
    '''Solve for Tb using Fp/Fs, Ts and a filter bandpass'''
    if '3.6' in f or '36' in f:
        waveset = np.linspace(3.15, 3.9, 1000) * astropy.units.micron
    else:
        waveset = np.linspace(4,5,1000) * astropy.units.micron

    def f2min(T, *args):
        fpfs,tstar,waveset = args
        fstar = BlackBody(waveset, tstar * astropy.units.K)
        fplanet = BlackBody(waveset, T * astropy.units.K)
        fp = np.trapz(fplanet, waveset)
        fs = np.trapz(fstar, waveset)
        return (fp/fs) - fpfs

    tb = brentq(f2min, 1,3500, args=(priors['fpfs'],priors['T*'],waveset))
    return tb

# --------------------------------------------------------------------
# -- NORMALIZATION -- ------------------------------------------------
def norm_jwst_niriss(cal, tme, fin, out, selftype, _debug=False):
    '''
    K. PEARSON:
        normalize each ramp
        remove nans, remove zeros, 3 sigma clip time series
    '''

    priors = fin['priors'].copy()

    planetloop = [pnet for pnet in tme['data'].keys() if
                  (pnet in priors.keys()) and tme['data'][pnet][selftype]]

    for p in planetloop:
        out['data'][p] = {}

        keys = ['TIME','SPEC','WAVE','RAMP_NUM']
        for k in keys:
            out['data'][p][k] = np.array(cal['data'][k])

        # time order things
        ordt = np.argsort(out['data'][p]['TIME'])
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][ordt]

        # 3 sigma clip flux time series
        if selftype == 'transit':
            phase = (out['data'][p]['TIME'] - fin['priors'][p]['t0'])/fin['priors'][p]['period']
        elif selftype == 'eclipse':
            priors = fin['priors']
            w = priors[p].get('omega',0)
            tme = priors[p]['t0']+ priors[p]['period']*0.5 * (1 + priors[p]['ecc']*(4./np.pi)*np.cos(np.deg2rad(w)))
            phase = (out['data'][p]['TIME'] - tme)/fin['priors'][p]['period']

        badmask = np.zeros(out['data'][p]['TIME'].shape).astype(bool)
        for i in np.unique(tme['data'][p][selftype]):
            for r in np.unique(out['data'][p]['RAMP_NUM']):
                # mask out orbit + RAMP
                omask = np.round(phase) == i
                rmask = out['data'][p]['RAMP_NUM'] == r

                dt = np.nanmean(np.diff(out['data'][p]['TIME'][omask]))*24*60
                ndt = int(5/dt)*2+1  # number of exposures in 10 minutes
                wlc = sigma_clip(out['data'][p]['SPEC'].sum(1)[omask & rmask], ndt)
                badmask[omask & rmask] = badmask[omask & rmask] | np.isnan(wlc)

        # remove outliers
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][~badmask]

        # pass information along
        out['data'][p]['transit'] = tme['data'][p]['transit']
        out['data'][p]['eclipse'] = tme['data'][p]['eclipse']
        out['data'][p]['phasecurve'] = tme['data'][p]['phasecurve']

        if out['data'][p][selftype]:
            normed = True
            out['STATUS'].append(True)

    return normed

def norm_spitzer(cal, tme, fin, out, selftype, debug=False):
    '''
    K. PEARSON: aperture selection, remove nans, remove zeros, 3 sigma clip time series
    '''
    normed = False
    priors = fin['priors'].copy()

    planetloop = [pnet for pnet in tme['data'].keys() if
                  (pnet in priors.keys()) and tme['data'][pnet][selftype]]

    for p in planetloop:
        out['data'][p] = {}

        # make sure cal/tme are in sync
        # mask = ~np.array(cal['data']['FAILED'])

        keys = [
            'TIME','WX','WY','FRAME',
        ]
        for k in keys:
            out['data'][p][k] = np.array(cal['data'][k])

        # is set later during aperture selection
        out['data'][p]['PHOT'] = np.zeros(len(cal['data']['TIME']))
        out['data'][p]['NOISEPIXEL'] = np.zeros(len(cal['data']['TIME']))

        # remove nans
        nanmask = np.isnan(out['data'][p]['PHOT'])
        for k in out['data'][p].keys():
            nanmask = nanmask | np.isnan(out['data'][p][k])

        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][~nanmask]

        # time order things
        ordt = np.argsort(out['data'][p]['TIME'])
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][ordt]
        cflux = np.array(cal['data']['PHOT'])[~nanmask][ordt]
        cnpp = np.array(cal['data']['NOISEPIXEL'])[~nanmask][ordt]

        # 3 sigma clip flux time series
        phase = (out['data'][p]['TIME'] - fin['priors'][p]['t0'])/fin['priors'][p]['period']
        badmask = np.zeros(out['data'][p]['TIME'].shape).astype(bool)
        for i in np.unique(tme['data'][p][selftype]):
            # mask out orbit
            omask = np.round(phase) == i
            dt = np.nanmean(np.diff(out['data'][p]['TIME'][omask]))*24*60
            ndt = int(7/dt)*2+1

            # aperture selection
            stds = []
            for j in range(cflux.shape[1]):
                stds.append(np.nanstd(sigma_clip(cflux[omask,j], ndt)))

            bi = np.argmin(stds)
            out['data'][p]['PHOT'][omask] = cflux[omask,bi]
            out['data'][p]['NOISEPIXEL'][omask] = cnpp[omask,bi]

            # sigma clip and remove nans
            photmask = np.isnan(sigma_clip(out['data'][p]['PHOT'][omask], ndt))
            xmask = np.isnan(sigma_clip(out['data'][p]['WX'][omask], ndt))
            ymask = np.isnan(sigma_clip(out['data'][p]['WY'][omask], ndt))
            nmask = np.isnan(sigma_clip(out['data'][p]['NOISEPIXEL'][omask], ndt))
            zmask = out['data'][p]['PHOT'][omask] == 0

            badmask[omask] = photmask | xmask | ymask | nmask | zmask

        # remove outliers
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][~badmask]

        # pass information along
        out['data'][p]['transit'] = tme['data'][p]['transit']
        out['data'][p]['eclipse'] = tme['data'][p]['eclipse']
        out['data'][p]['phasecurve'] = tme['data'][p]['phasecurve']

        if debug:
            plt.plot(out['data'][p]['TIME'][~badmask], out['data'][p]['PHOT'][~badmask], 'k.')
            plt.plot(out['data'][p]['TIME'][badmask], out['data'][p]['PHOT'][badmask], 'r.')
            plt.show()

            plt.plot(out['data'][p]['TIME'], out['data'][p]['PHOT'],'k.')
            plt.xlabel('Time')
            plt.ylabel('Flux')
            plt.show()

        if out['data'][p][selftype]:
            normed = True
            out['STATUS'].append(True)

    return normed

def get_ld(priors, band='Spit36'):
    '''
    Query the web for limb darkening coefficients in the Spitzer bandpass
    Problem with LDTK + Spitzer: https://github.com/hpparvi/ldtk/issues/11
    '''
    url = 'http://astroutils.astronomy.ohio-state.edu/exofast/quadld.php'

    form = {
        'action':url,
        'pname':'Select Planet',
        'bname':band,
        'teff':priors['T*'],
        'feh':priors['FEH*'],
        'logg':priors['LOGG*']
    }
    session = requests.Session()
    res = session.post(url,data=form)
    lin,quad = re.findall(r"\d+\.\d+",res.text)
    return float(lin), float(quad)

def sigma_clip(ogdata,dt):
    '''sigma_clip ds'''
    mdata = savgol_filter(ogdata, dt, 2)
    res = ogdata - mdata
    try:
        std = np.nanmedian([np.nanstd(np.random.choice(res,100)) for i in range(250)])
    except IndexError:
        std = np.nanstd(res)  # biased by outliers
    mask = np.abs(res) > 3*std
    data = copy.deepcopy(ogdata)
    data[mask] = np.nan
    return data

def time2z(time, ipct, tknot, sma, orbperiod, ecc, tperi=None, epsilon=1e-5):
    '''
    G. ROUDIER: Time samples in [Days] to separation in [R*]
    '''
    if tperi is not None:
        ft0 = (tperi - tknot) % orbperiod
        ft0 /= orbperiod
        if ft0 > 0.5: ft0 += -1e0
        M0 = 2e0*np.pi*ft0
        E0 = solveme(M0, ecc, epsilon)
        realf = np.sqrt(1e0 - ecc)*np.cos(E0/2e0)
        imagf = np.sqrt(1e0 + ecc)*np.sin(E0/2e0)
        w = np.angle(np.complex(realf, imagf))
        if abs(ft0) < epsilon:
            w = np.pi/2e0
            tperi = tknot
            pass
        pass
    else:
        w = np.pi/2e0
        tperi = tknot
        pass
    ft = (time - tperi) % orbperiod
    ft /= orbperiod
    sft = np.copy(ft)
    sft[(sft > 0.5)] += -1e0
    M = 2e0*np.pi*ft
    E = solveme(M, ecc, epsilon)
    realf = np.sqrt(1. - ecc)*np.cos(E/2e0)
    imagf = np.sqrt(1. + ecc)*np.sin(E/2e0)
    f = []
    for r, i in zip(realf, imagf):
        cn = np.complex(r, i)
        f.append(2e0*np.angle(cn))
        pass
    f = np.array(f)
    r = sma*(1e0 - ecc**2)/(1e0 + ecc*np.cos(f))
    z = r*np.sqrt(1e0**2 - (np.sin(w+f)**2)*(np.sin(ipct*np.pi/180e0))**2)
    z[sft < 0] *= -1e0
    return z, sft

def solveme(M, e, eps):
    '''
    G. ROUDIER: Newton Raphson solver for true anomaly
    M is a numpy array
    '''
    E = np.copy(M)
    for i in np.arange(M.shape[0]):
        while abs(E[i] - e*np.sin(E[i]) - M[i]) > eps:
            num = E[i] - e*np.sin(E[i]) - M[i]
            den = 1. - e*np.cos(E[i])
            E[i] = E[i] - num/den
            pass
        pass
    return E

################################################
# PyLightcurve "mini" - https://github.com/ucl-exoplanets/pylightcurve

# coefficients from https://pomax.github.io/bezierinfo/legendre-gauss.html

gauss0 = [
    [1.0000000000000000, -0.5773502691896257],
    [1.0000000000000000, 0.5773502691896257]
]

gauss10 = [
    [0.2955242247147529, -0.1488743389816312],
    [0.2955242247147529, 0.1488743389816312],
    [0.2692667193099963, -0.4333953941292472],
    [0.2692667193099963, 0.4333953941292472],
    [0.2190863625159820, -0.6794095682990244],
    [0.2190863625159820, 0.6794095682990244],
    [0.1494513491505806, -0.8650633666889845],
    [0.1494513491505806, 0.8650633666889845],
    [0.0666713443086881, -0.9739065285171717],
    [0.0666713443086881, 0.9739065285171717]
]

gauss20 = [
    [0.1527533871307258, -0.0765265211334973],
    [0.1527533871307258, 0.0765265211334973],
    [0.1491729864726037, -0.2277858511416451],
    [0.1491729864726037, 0.2277858511416451],
    [0.1420961093183820, -0.3737060887154195],
    [0.1420961093183820, 0.3737060887154195],
    [0.1316886384491766, -0.5108670019508271],
    [0.1316886384491766, 0.5108670019508271],
    [0.1181945319615184, -0.6360536807265150],
    [0.1181945319615184, 0.6360536807265150],
    [0.1019301198172404, -0.7463319064601508],
    [0.1019301198172404, 0.7463319064601508],
    [0.0832767415767048, -0.8391169718222188],
    [0.0832767415767048, 0.8391169718222188],
    [0.0626720483341091, -0.9122344282513259],
    [0.0626720483341091, 0.9122344282513259],
    [0.0406014298003869, -0.9639719272779138],
    [0.0406014298003869, 0.9639719272779138],
    [0.0176140071391521, -0.9931285991850949],
    [0.0176140071391521, 0.9931285991850949],
]

gauss30 = [
    [0.1028526528935588, -0.0514718425553177],
    [0.1028526528935588, 0.0514718425553177],
    [0.1017623897484055, -0.1538699136085835],
    [0.1017623897484055, 0.1538699136085835],
    [0.0995934205867953, -0.2546369261678899],
    [0.0995934205867953, 0.2546369261678899],
    [0.0963687371746443, -0.3527047255308781],
    [0.0963687371746443, 0.3527047255308781],
    [0.0921225222377861, -0.4470337695380892],
    [0.0921225222377861, 0.4470337695380892],
    [0.0868997872010830, -0.5366241481420199],
    [0.0868997872010830, 0.5366241481420199],
    [0.0807558952294202, -0.6205261829892429],
    [0.0807558952294202, 0.6205261829892429],
    [0.0737559747377052, -0.6978504947933158],
    [0.0737559747377052, 0.6978504947933158],
    [0.0659742298821805, -0.7677774321048262],
    [0.0659742298821805, 0.7677774321048262],
    [0.0574931562176191, -0.8295657623827684],
    [0.0574931562176191, 0.8295657623827684],
    [0.0484026728305941, -0.8825605357920527],
    [0.0484026728305941, 0.8825605357920527],
    [0.0387991925696271, -0.9262000474292743],
    [0.0387991925696271, 0.9262000474292743],
    [0.0287847078833234, -0.9600218649683075],
    [0.0287847078833234, 0.9600218649683075],
    [0.0184664683110910, -0.9836681232797472],
    [0.0184664683110910, 0.9836681232797472],
    [0.0079681924961666, -0.9968934840746495],
    [0.0079681924961666, 0.9968934840746495]
]

gauss40 = [
    [0.0775059479784248, -0.0387724175060508],
    [0.0775059479784248, 0.0387724175060508],
    [0.0770398181642480, -0.1160840706752552],
    [0.0770398181642480, 0.1160840706752552],
    [0.0761103619006262, -0.1926975807013711],
    [0.0761103619006262, 0.1926975807013711],
    [0.0747231690579683, -0.2681521850072537],
    [0.0747231690579683, 0.2681521850072537],
    [0.0728865823958041, -0.3419940908257585],
    [0.0728865823958041, 0.3419940908257585],
    [0.0706116473912868, -0.4137792043716050],
    [0.0706116473912868, 0.4137792043716050],
    [0.0679120458152339, -0.4830758016861787],
    [0.0679120458152339, 0.4830758016861787],
    [0.0648040134566010, -0.5494671250951282],
    [0.0648040134566010, 0.5494671250951282],
    [0.0613062424929289, -0.6125538896679802],
    [0.0613062424929289, 0.6125538896679802],
    [0.0574397690993916, -0.6719566846141796],
    [0.0574397690993916, 0.6719566846141796],
    [0.0532278469839368, -0.7273182551899271],
    [0.0532278469839368, 0.7273182551899271],
    [0.0486958076350722, -0.7783056514265194],
    [0.0486958076350722, 0.7783056514265194],
    [0.0438709081856733, -0.8246122308333117],
    [0.0438709081856733, 0.8246122308333117],
    [0.0387821679744720, -0.8659595032122595],
    [0.0387821679744720, 0.8659595032122595],
    [0.0334601952825478, -0.9020988069688743],
    [0.0334601952825478, 0.9020988069688743],
    [0.0279370069800234, -0.9328128082786765],
    [0.0279370069800234, 0.9328128082786765],
    [0.0222458491941670, -0.9579168192137917],
    [0.0222458491941670, 0.9579168192137917],
    [0.0164210583819079, -0.9772599499837743],
    [0.0164210583819079, 0.9772599499837743],
    [0.0104982845311528, -0.9907262386994570],
    [0.0104982845311528, 0.9907262386994570],
    [0.0045212770985332, -0.9982377097105593],
    [0.0045212770985332, 0.9982377097105593],
]

gauss50 = [
    [0.0621766166553473, -0.0310983383271889],
    [0.0621766166553473, 0.0310983383271889],
    [0.0619360674206832, -0.0931747015600861],
    [0.0619360674206832, 0.0931747015600861],
    [0.0614558995903167, -0.1548905899981459],
    [0.0614558995903167, 0.1548905899981459],
    [0.0607379708417702, -0.2160072368760418],
    [0.0607379708417702, 0.2160072368760418],
    [0.0597850587042655, -0.2762881937795320],
    [0.0597850587042655, 0.2762881937795320],
    [0.0586008498132224, -0.3355002454194373],
    [0.0586008498132224, 0.3355002454194373],
    [0.0571899256477284, -0.3934143118975651],
    [0.0571899256477284, 0.3934143118975651],
    [0.0555577448062125, -0.4498063349740388],
    [0.0555577448062125, 0.4498063349740388],
    [0.0537106218889962, -0.5044581449074642],
    [0.0537106218889962, 0.5044581449074642],
    [0.0516557030695811, -0.5571583045146501],
    [0.0516557030695811, 0.5571583045146501],
    [0.0494009384494663, -0.6077029271849502],
    [0.0494009384494663, 0.6077029271849502],
    [0.0469550513039484, -0.6558964656854394],
    [0.0469550513039484, 0.6558964656854394],
    [0.0443275043388033, -0.7015524687068222],
    [0.0443275043388033, 0.7015524687068222],
    [0.0415284630901477, -0.7444943022260685],
    [0.0415284630901477, 0.7444943022260685],
    [0.0385687566125877, -0.7845558329003993],
    [0.0385687566125877, 0.7845558329003993],
    [0.0354598356151462, -0.8215820708593360],
    [0.0354598356151462, 0.8215820708593360],
    [0.0322137282235780, -0.8554297694299461],
    [0.0322137282235780, 0.8554297694299461],
    [0.0288429935805352, -0.8859679795236131],
    [0.0288429935805352, 0.8859679795236131],
    [0.0253606735700124, -0.9130785566557919],
    [0.0253606735700124, 0.9130785566557919],
    [0.0217802431701248, -0.9366566189448780],
    [0.0217802431701248, 0.9366566189448780],
    [0.0181155607134894, -0.9566109552428079],
    [0.0181155607134894, 0.9566109552428079],
    [0.0143808227614856, -0.9728643851066920],
    [0.0143808227614856, 0.9728643851066920],
    [0.0105905483836510, -0.9853540840480058],
    [0.0105905483836510, 0.9853540840480058],
    [0.0067597991957454, -0.9940319694320907],
    [0.0067597991957454, 0.9940319694320907],
    [0.0029086225531551, -0.9988664044200710],
    [0.0029086225531551, 0.9988664044200710]
]

gauss60 = [
    [0.0519078776312206, -0.0259597723012478],
    [0.0519078776312206, 0.0259597723012478],
    [0.0517679431749102, -0.0778093339495366],
    [0.0517679431749102, 0.0778093339495366],
    [0.0514884515009809, -0.1294491353969450],
    [0.0514884515009809, 0.1294491353969450],
    [0.0510701560698556, -0.1807399648734254],
    [0.0510701560698556, 0.1807399648734254],
    [0.0505141845325094, -0.2315435513760293],
    [0.0505141845325094, 0.2315435513760293],
    [0.0498220356905502, -0.2817229374232617],
    [0.0498220356905502, 0.2817229374232617],
    [0.0489955754557568, -0.3311428482684482],
    [0.0489955754557568, 0.3311428482684482],
    [0.0480370318199712, -0.3796700565767980],
    [0.0480370318199712, 0.3796700565767980],
    [0.0469489888489122, -0.4271737415830784],
    [0.0469489888489122, 0.4271737415830784],
    [0.0457343797161145, -0.4735258417617071],
    [0.0457343797161145, 0.4735258417617071],
    [0.0443964787957871, -0.5186014000585697],
    [0.0443964787957871, 0.5186014000585697],
    [0.0429388928359356, -0.5622789007539445],
    [0.0429388928359356, 0.5622789007539445],
    [0.0413655512355848, -0.6044405970485104],
    [0.0413655512355848, 0.6044405970485104],
    [0.0396806954523808, -0.6449728284894770],
    [0.0396806954523808, 0.6449728284894770],
    [0.0378888675692434, -0.6837663273813555],
    [0.0378888675692434, 0.6837663273813555],
    [0.0359948980510845, -0.7207165133557304],
    [0.0359948980510845, 0.7207165133557304],
    [0.0340038927249464, -0.7557237753065856],
    [0.0340038927249464, 0.7557237753065856],
    [0.0319212190192963, -0.7886937399322641],
    [0.0319212190192963, 0.7886937399322641],
    [0.0297524915007889, -0.8195375261621458],
    [0.0297524915007889, 0.8195375261621458],
    [0.0275035567499248, -0.8481719847859296],
    [0.0275035567499248, 0.8481719847859296],
    [0.0251804776215212, -0.8745199226468983],
    [0.0251804776215212, 0.8745199226468983],
    [0.0227895169439978, -0.8985103108100460],
    [0.0227895169439978, 0.8985103108100460],
    [0.0203371207294573, -0.9200784761776275],
    [0.0203371207294573, 0.9200784761776275],
    [0.0178299010142077, -0.9391662761164232],
    [0.0178299010142077, 0.9391662761164232],
    [0.0152746185967848, -0.9557222558399961],
    [0.0152746185967848, 0.9557222558399961],
    [0.0126781664768160, -0.9697017887650528],
    [0.0126781664768160, 0.9697017887650528],
    [0.0100475571822880, -0.9810672017525982],
    [0.0100475571822880, 0.9810672017525982],
    [0.0073899311633455, -0.9897878952222218],
    [0.0073899311633455, 0.9897878952222218],
    [0.0047127299269536, -0.9958405251188381],
    [0.0047127299269536, 0.9958405251188381],
    [0.0020268119688738, -0.9992101232274361],
    [0.0020268119688738, 0.9992101232274361],
]

gauss_table = [np.swapaxes(gauss0, 0, 1), np.swapaxes(gauss10, 0, 1), np.swapaxes(gauss20, 0, 1),
               np.swapaxes(gauss30, 0, 1), np.swapaxes(gauss40, 0, 1), np.swapaxes(gauss50, 0, 1),
               np.swapaxes(gauss60, 0, 1)]


def gauss_numerical_integration(f, x1, x2, precision, *f_args):
    '''gauss_numerical_integration ds'''
    x1, x2 = (x2 - x1) / 2, (x2 + x1) / 2

    return x1 * np.sum(gauss_table[precision][0][:, None] *
                       f(x1[None, :] * gauss_table[precision][1][:, None] + x2[None, :], *f_args), 0)


def sample_function(f, precision=3):
    '''sample_function ds'''
    def sampled_function(x12_array, *args):
        '''sampled_function ds'''
        x1_array, x2_array = x12_array

        return gauss_numerical_integration(f, x1_array, x2_array, precision, *list(args))

    return sampled_function


# orbit
def planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array, ww=0):
    '''planet_orbit ds'''
    # pylint: disable=no-member
    inclination = inclination * np.pi / 180.0
    periastron = periastron * np.pi / 180.0
    ww = ww * np.pi / 180.0

    if eccentricity == 0 and ww == 0:
        vv = 2 * np.pi * (time_array - mid_time) / period
        bb = sma_over_rs * np.cos(vv)
        return [bb * np.sin(inclination), sma_over_rs * np.sin(vv), - bb * np.cos(inclination)]

    if periastron < np.pi / 2:
        aa = 1.0 * np.pi / 2 - periastron
    else:
        aa = 5.0 * np.pi / 2 - periastron
    bb = 2 * np.arctan(np.sqrt((1 - eccentricity) / (1 + eccentricity)) * np.tan(aa / 2))
    if bb < 0:
        bb += 2 * np.pi
    mid_time = float(mid_time) - (period / 2.0 / np.pi) * (bb - eccentricity * np.sin(bb))
    m = (time_array - mid_time - np.int_((time_array - mid_time) / period) * period) * 2.0 * np.pi / period
    u0 = m
    stop = False
    u1 = 0
    for _ in range(10000):  # setting a limit of 1k iterations - arbitrary limit
        u1 = u0 - (u0 - eccentricity * np.sin(u0) - m) / (1 - eccentricity * np.cos(u0))
        stop = (np.abs(u1 - u0) < 10 ** (-7)).all()
        if stop:
            break
        u0 = u1
        pass

    if not stop:
        raise RuntimeError('Failed to find a solution in 10000 loops')

    vv = 2 * np.arctan(np.sqrt((1 + eccentricity) / (1 - eccentricity)) * np.tan(u1 / 2))
    #
    rr = sma_over_rs * (1 - (eccentricity ** 2)) / (np.ones_like(vv) + eccentricity * np.cos(vv))
    aa = np.cos(vv + periastron)
    bb = np.sin(vv + periastron)
    x = rr * bb * np.sin(inclination)
    y = rr * (-aa * np.cos(ww) + bb * np.sin(ww) * np.cos(inclination))
    z = rr * (-aa * np.sin(ww) - bb * np.cos(ww) * np.cos(inclination))

    return [x, y, z]


def planet_star_projected_distance(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array):
    '''planet_star_projected_distance ds'''
    position_vector = planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)

    return np.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2])


def planet_phase(period, mid_time, time_array):
    '''planet_phase ds'''
    return (time_array - mid_time)/period


# flux drop


def integral_r_claret(limb_darkening_coefficients, r):
    '''integral_r_claret ds'''
    a1, a2, a3, a4 = limb_darkening_coefficients
    mu44 = 1.0 - r * r
    mu24 = np.sqrt(mu44)
    mu14 = np.sqrt(mu24)
    return - (2.0 * (1.0 - a1 - a2 - a3 - a4) / 4) * mu44 \
           - (2.0 * a1 / 5) * mu44 * mu14 \
           - (2.0 * a2 / 6) * mu44 * mu24 \
           - (2.0 * a3 / 7) * mu44 * mu24 * mu14 \
           - (2.0 * a4 / 8) * mu44 * mu44


def num_claret(r, limb_darkening_coefficients, rprs, z):
    '''num_claret ds'''
    # pylint: disable=no-member
    a1, a2, a3, a4 = limb_darkening_coefficients
    rsq = r * r
    mu44 = 1.0 - rsq
    mu24 = np.sqrt(mu44)
    mu14 = np.sqrt(mu24)
    return ((1.0 - a1 - a2 - a3 - a4) + a1 * mu14 + a2 * mu24 + a3 * mu24 * mu14 + a4 * mu44) \
        * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_claret(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    '''integral_r_f_claret ds'''
    return gauss_numerical_integration(num_claret, r1, r2, precision, limb_darkening_coefficients, rprs, z)

def integral_r_zero(_, r):
    '''integral definitions for zero method'''
    musq = 1 - r * r
    return (-1.0 / 6) * musq * 3.0


def num_zero(r, _, rprs, z):
    '''num_zero ds'''
    # pylint: disable=no-member
    rsq = r * r
    return r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_zero(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    '''integral_r_f_zero ds'''
    return gauss_numerical_integration(num_zero, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for linear method
def integral_r_linear(limb_darkening_coefficients, r):
    '''integral_r_linear ds'''
    a1 = limb_darkening_coefficients[0]
    musq = 1 - r * r
    return (-1.0 / 6) * musq * (3.0 + a1 * (-3.0 + 2.0 * np.sqrt(musq)))


def num_linear(r, limb_darkening_coefficients, rprs, z):
    '''num_linear ds'''
    # pylint: disable=no-member
    a1 = limb_darkening_coefficients[0]
    rsq = r * r
    return (1.0 - a1 * (1.0 - np.sqrt(1.0 - rsq))) \
        * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_linear(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    '''integral_r_f_linear ds'''
    return gauss_numerical_integration(num_linear, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# integral definitions for quadratic method

def integral_r_quad(limb_darkening_coefficients, r):
    '''integral_r_quad ds'''
    a1, a2 = limb_darkening_coefficients[:2]
    musq = 1 - r * r
    mu = np.sqrt(musq)
    return (1.0 / 12) * (-4.0 * (a1 + 2.0 * a2) * mu * musq + 6.0 * (-1 + a1 + a2) * musq + 3.0 * a2 * musq * musq)


def num_quad(r, limb_darkening_coefficients, rprs, z):
    '''num_quad ds'''
    # pylint: disable=no-member
    a1, a2 = limb_darkening_coefficients[:2]
    rsq = r * r
    cc = 1.0 - np.sqrt(1.0 - rsq)
    return (1.0 - a1 * cc - a2 * cc * cc) \
        * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_quad(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    '''integral_r_f_quad ds'''
    return gauss_numerical_integration(num_quad, r1, r2, precision, limb_darkening_coefficients, rprs, z)

# integral definitions for square root method

def integral_r_sqrt(limb_darkening_coefficients, r):
    '''integral_r_sqrt ds'''
    a1, a2 = limb_darkening_coefficients[:2]
    musq = 1 - r * r
    mu = np.sqrt(musq)
    return ((-2.0 / 5) * a2 * np.sqrt(mu) - (1.0 / 3) * a1 * mu + (1.0 / 2) * (-1 + a1 + a2)) * musq


def num_sqrt(r, limb_darkening_coefficients, rprs, z):
    '''num_sqrt ds'''
    # pylint: disable=no-member
    a1, a2 = limb_darkening_coefficients[:2]
    rsq = r * r
    mu = np.sqrt(1.0 - rsq)
    return (1.0 - a1 * (1 - mu) - a2 * (1.0 - np.sqrt(mu))) \
        * r * np.arccos(np.minimum((-rprs ** 2 + z * z + rsq) / (2.0 * z * r), 1.0))


def integral_r_f_sqrt(limb_darkening_coefficients, rprs, z, r1, r2, precision=3):
    '''integral_r_f_sqrt ds'''
    return gauss_numerical_integration(num_sqrt, r1, r2, precision, limb_darkening_coefficients, rprs, z)


# dictionaries containing the different methods,
# if you define a new method, include the functions in the dictionary as well

integral_r = {
    'claret': integral_r_claret,
    'linear': integral_r_linear,
    'quad': integral_r_quad,
    'sqrt': integral_r_sqrt,
    'zero': integral_r_zero
}

integral_r_f = {
    'claret': integral_r_f_claret,
    'linear': integral_r_f_linear,
    'quad': integral_r_f_quad,
    'sqrt': integral_r_f_sqrt,
    'zero': integral_r_f_zero,
}

def integral_centred(method, limb_darkening_coefficients, rprs, ww1, ww2):
    '''integral_centred ds'''
    return (integral_r[method](limb_darkening_coefficients, rprs) - integral_r[method](limb_darkening_coefficients, 0.0)) * np.abs(ww2 - ww1)

def integral_plus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    '''integral_plus_core ds'''
    # pylint: disable=len-as-condition,no-member
    if len(z) == 0: return z
    rr1 = z * np.cos(ww1) + np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww1)) ** 2, 0))
    rr1 = np.clip(rr1, 0, 1)
    rr2 = z * np.cos(ww2) + np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww2)) ** 2, 0))
    rr2 = np.clip(rr2, 0, 1)
    w1 = np.minimum(ww1, ww2)
    r1 = np.minimum(rr1, rr2)
    w2 = np.maximum(ww1, ww2)
    r2 = np.maximum(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, 0.0) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * w2
    partc = integral_r[method](limb_darkening_coefficients, r2) * (-w1)
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc + partd

def integral_minus_core(method, limb_darkening_coefficients, rprs, z, ww1, ww2, precision=3):
    '''integral_minus_core ds'''
    # pylint: disable=len-as-condition,no-member
    if len(z) == 0: return z
    rr1 = z * np.cos(ww1) - np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww1)) ** 2, 0))
    rr1 = np.clip(rr1, 0, 1)
    rr2 = z * np.cos(ww2) - np.sqrt(np.maximum(rprs ** 2 - (z * np.sin(ww2)) ** 2, 0))
    rr2 = np.clip(rr2, 0, 1)
    w1 = np.minimum(ww1, ww2)
    r1 = np.minimum(rr1, rr2)
    w2 = np.maximum(ww1, ww2)
    r2 = np.maximum(rr1, rr2)
    parta = integral_r[method](limb_darkening_coefficients, 0.0) * (w1 - w2)
    partb = integral_r[method](limb_darkening_coefficients, r1) * (-w1)
    partc = integral_r[method](limb_darkening_coefficients, r2) * w2
    partd = integral_r_f[method](limb_darkening_coefficients, rprs, z, r1, r2, precision=precision)
    return parta + partb + partc - partd


def transit_flux_drop(limb_darkening_coefficients, rp_over_rs, z_over_rs, method='claret', precision=3):
    '''transit_flux_drop ds'''
    # pylint: disable=len-as-condition,no-member
    z_over_rs = np.where(z_over_rs < 0, 1.0 + 100.0 * rp_over_rs, z_over_rs)
    z_over_rs = np.maximum(z_over_rs, 10**(-10))

    # cases
    zsq = z_over_rs * z_over_rs
    sum_z_rprs = z_over_rs + rp_over_rs
    dif_z_rprs = rp_over_rs - z_over_rs
    sqr_dif_z_rprs = zsq - rp_over_rs ** 2
    case0 = np.where((z_over_rs == 0) & (rp_over_rs <= 1))
    case1 = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs <= 1))
    casea = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs < 1))
    caseb = np.where((z_over_rs < rp_over_rs) & (sum_z_rprs > 1) & (dif_z_rprs > 1))
    case2 = np.where((z_over_rs == rp_over_rs) & (sum_z_rprs <= 1))
    casec = np.where((z_over_rs == rp_over_rs) & (sum_z_rprs > 1))
    case3 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs < 1))
    case4 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs == 1))
    case5 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs < 1))
    case6 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs == 1))
    case7 = np.where((z_over_rs > rp_over_rs) & (sum_z_rprs > 1) & (sqr_dif_z_rprs > 1) & (-1 < dif_z_rprs))
    plus_case = np.concatenate((case1[0], case2[0], case3[0], case4[0], case5[0], casea[0], casec[0]))
    minus_case = np.concatenate((case3[0], case4[0], case5[0], case6[0], case7[0]))
    star_case = np.concatenate((case5[0], case6[0], case7[0], casea[0], casec[0]))

    # cross points
    ph = np.arccos(np.clip((1.0 - rp_over_rs ** 2 + zsq) / (2.0 * z_over_rs), -1, 1))
    theta_1 = np.zeros(len(z_over_rs))
    ph_case = np.concatenate((case5[0], casea[0], casec[0]))
    theta_1[ph_case] = ph[ph_case]
    theta_2 = np.arcsin(np.minimum(rp_over_rs / z_over_rs, 1))
    theta_2[case1] = np.pi
    theta_2[case2] = np.pi / 2.0
    theta_2[casea] = np.pi
    theta_2[casec] = np.pi / 2.0
    theta_2[case7] = ph[case7]

    # flux_upper
    plusflux = np.zeros(len(z_over_rs))
    plusflux[plus_case] = integral_plus_core(method, limb_darkening_coefficients, rp_over_rs, z_over_rs[plus_case],
                                             theta_1[plus_case], theta_2[plus_case], precision=precision)
    if len(case0[0]) > 0:
        plusflux[case0] = integral_centred(method, limb_darkening_coefficients, rp_over_rs, 0.0, np.pi)
    if len(caseb[0]) > 0:
        plusflux[caseb] = integral_centred(method, limb_darkening_coefficients, 1, 0.0, np.pi)

    # flux_lower
    minsflux = np.zeros(len(z_over_rs))
    minsflux[minus_case] = integral_minus_core(method, limb_darkening_coefficients, rp_over_rs,
                                               z_over_rs[minus_case], 0.0, theta_2[minus_case], precision=precision)

    # flux_star
    starflux = np.zeros(len(z_over_rs))
    starflux[star_case] = integral_centred(method, limb_darkening_coefficients, 1, 0.0, ph[star_case])

    # flux_total
    total_flux = integral_centred(method, limb_darkening_coefficients, 1, 0.0, 2.0 * np.pi)

    return 1 - (2.0 / total_flux) * (plusflux + starflux - minsflux)

# transit
def pytransit(limb_darkening_coefficients, rp_over_rs, period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array, method='claret', precision=3):
    '''pytransit ds'''

    position_vector = planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time, time_array)

    projected_distance = np.where(position_vector[0] < 0, 1.0 + 5.0 * rp_over_rs,
                                  np.sqrt(position_vector[1] * position_vector[1] + position_vector[2] * position_vector[2]))

    return transit_flux_drop(limb_darkening_coefficients, rp_over_rs, projected_distance,
                             method=method, precision=precision)

def eclipse_mid_time(period, sma_over_rs, eccentricity, inclination, periastron, mid_time):
    '''eclipse_mid_time ds'''
    test_array = np.arange(0, period, 0.001)
    xx, yy, _ = planet_orbit(period, sma_over_rs, eccentricity, inclination, periastron, mid_time,
                             test_array + mid_time)

    test1 = np.where(xx < 0)
    yy = yy[test1]
    test_array = test_array[test1]

    aprox = test_array[np.argmin(np.abs(yy))]

    def function_to_fit(x, t):
        '''function_to_fit ds'''
        _, yy, _ = planet_orbit(period, sma_over_rs, eccentricity, inclination,
                                periastron, mid_time, np.array(mid_time + t))
        return yy

    popt, *_ = curve_fit(function_to_fit, [0], [0], p0=[aprox])

    return mid_time + popt[0]

def transit(times, values):
    '''transit ds'''
    return pytransit([values['u0'], values['u1'], values['u2'], values['u3']],
                     values['rprs'], values['per'], values['ars'],
                     values['ecc'], values['inc'], values['omega'],
                     values['tmid'], times, method='claret', precision=3)
################################################

def weightedflux(flux,gw,nearest):
    '''weightedflux ds'''
    return np.sum(flux[nearest]*gw,axis=-1)

def gaussian_weights(X, w=None, neighbors=50, feature_scale=1000):
    '''gaussian_weights ds'''
    if isinstance(w, type(None)): w = np.ones(X.shape[1])
    Xm = (X - np.median(X,0))*w
    kdtree = spatial.cKDTree(Xm*feature_scale)
    nearest = np.zeros((X.shape[0],neighbors))
    gw = np.zeros((X.shape[0],neighbors),dtype=float)
    for point in range(X.shape[0]):
        ind = kdtree.query(kdtree.data[point],neighbors+1)[1][1:]
        dX = Xm[ind] - Xm[point]
        Xstd = np.std(dX,0)
        gX = np.exp(-dX**2/(2*Xstd**2))
        gwX = np.product(gX,1)
        gw[point,:] = gwX/gwX.sum()
        nearest[point,:] = ind
    gw[np.isnan(gw)] = 0.01
    return gw, nearest.astype(int)

def eclipse_ratio(priors,p='b',f='IRAC 3.6um', verbose=True):
    '''eclipse_ratio ds'''
    Te = priors['T*']*(1-0.1)**0.25 * np.sqrt(0.5/priors[p]['ars'])

    rprs = priors[p]['rp'] * astropy.constants.R_jup / (priors['R*'] * astropy.constants.R_sun)
    tdepth = rprs.value**2

    # bandpass integrated flux for planet
    wave36 = np.linspace(3.15,3.95,1000) * astropy.units.micron
    wave45 = np.linspace(4,5,1000) * astropy.units.micron

    try:
        fplanet = BlackBody(Te*astropy.units.K)(wave36)
        fstar = BlackBody(priors['T*']*astropy.units.K)(wave36)
    except TypeError:
        fplanet = BlackBody(wave36, Te*astropy.units.K)
        fstar = BlackBody(wave36, priors['T*']*astropy.units.K)

    fp36 = np.trapz(fplanet, wave36)
    fs36 = np.trapz(fstar, wave36)

    try:
        fplanet = BlackBody(Te*astropy.units.K)(wave45)
        fstar = BlackBody(priors['T*']*astropy.units.K)(wave45)
    except TypeError:
        fplanet = BlackBody(wave45, Te*astropy.units.K)
        fstar = BlackBody(wave45, priors['T*']*astropy.units.K)

    fp45 = np.trapz(fplanet, wave45)
    fs45 = np.trapz(fstar, wave45)

    if verbose:
        print(f" Stellar temp: {priors['T*']:.1f} K")
        print(f" Transit Depth: {tdepth*100:.4f} %")
        pass

    if '3.6' in f or '36' in f:
        if verbose:
            print(f" Eclipse Depth @ IRAC 1 (3.6um): ~{tdepth*fp36/fs36*1e6:.0f} ppm")
            print(f"         Fp/Fs @ IRAC 1 (3.6um): ~{fp36/fs36:.4f}")
        return float(fp36/fs36)
    if verbose:
        print(f" Eclipse Depth @ IRAC 2 (4.5um): ~{tdepth*fp45/fs45*1e6:.0f} ppm")
        print(f"         Fp/Fs @ IRAC 2 (4.5um): ~{fp45/fs45:.4f}")
    return float(fp45/fs45)

def lightcurve_jwst_niriss(nrm, fin, out, selftype, _fltr, hstwhitelight_sv, method='ns'):
    '''
    K. PEARSON: white light curve fit for orbital solution
    '''
    wl = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = list(nrm['data'].keys())

    for p in planetloop:
        out['data'][p] = []

        # extract data based on phase
        if selftype == 'transit':
            phase = (nrm['data'][p]['TIME'] - fin['priors'][p]['t0'])/fin['priors'][p]['period']
        elif selftype == 'eclipse':
            priors = fin['priors']
            w = priors[p].get('omega',0)
            tme = priors[p]['t0']+ priors[p]['period']*0.5 * (1 + priors[p]['ecc']*(4./np.pi)*np.cos(np.deg2rad(w)))
            phase = (nrm['data'][p]['TIME'] - tme)/fin['priors'][p]['period']

        # loop through epochs
        ec = 0  # event counter
        for event in nrm['data'][p][selftype]:
            print('processing event:',event)

            # compute phase + priors
            smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
            # smaors_up = (priors[p]['sma']+3*priors[p]['sma_uperr'])/(priors['R*']-abs(priors['R*_lowerr']))/ssc['Rsun/AU']
            # smaors_lo = (priors[p]['sma']-abs(3*priors[p]['sma_lowerr']))/(priors['R*']+priors['R*_uperr'])/ssc['Rsun/AU']

            _tmid = priors[p]['t0'] + event*priors[p]['period']

            # to do: update duration for eccentric orbits
            # https://arxiv.org/pdf/1001.2010.pdf eq 16
            tdur = priors[p]['period']/(np.pi)/smaors
            rprs = (priors[p]['rp']*7.1492e7) / (priors['R*']*6.955e8)
            # inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
            w = priors[p].get('omega',0)

            # mask out data by event type
            pmask = (phase > event-1.5*tdur/priors[p]['period']) & (phase < event+1.5*tdur/priors[p]['period'])

            # extract data + collapse into whitelight
            subt = nrm['data'][p]['TIME'][pmask]
            rnum = nrm['data'][p]['RAMP_NUM'][pmask]

            # future filter out noise channels?
            aper = nrm['data'][p]['SPEC'][pmask].sum(1)
            aper_err = np.sqrt(aper)

            # normalize each ramp by OoT baseline
            # for r in np.unique(rnum):
            #    rmask = rnum==r
            # ootmask = (phase[pmask] < event-.5*tdur/priors[p]['period']) | (phase[pmask] > event+.5*tdur/priors[p]['period'])
            #    aper_err[rmask] /= np.median(aper[ootmask&rmask])
            #    aper[rmask] /= np.median(aper[ootmask&rmask])
            #    # future improve with quadratic detrend?

            # take max ramp
            rmask = rnum==np.max(rnum)
            aper = aper[rmask]
            aper_err = aper_err[rmask]
            subt = subt[rmask]

            # diagnostic
            # plt.scatter(subt,aper,c=rnum,marker='.'); plt.colorbar(); plt.show()

            # can't solve for wavelengths greater than below
            # whiteld = createldgrid([2.5],[2.6], priors, segmentation=int(10), verbose=verbose)
            # whiteld = createldgrid([wmin],[wmax], priors, segmentation=int(1), verbose=verbose)

            # LDTK breaks for Spitzer https://github.com/hpparvi/ldtk/issues/11
            # filters = [BoxcarFilter('a', 3150, 3950)]
            # tstar = priors['T*']
            # terr = np.sqrt(abs(priors['T*_uperr']*priors['T*_lowerr']))
            # fehstar = priors['FEH*']
            # feherr = np.sqrt(abs(priors['FEH*_uperr']*priors['FEH*_lowerr']))
            # loggstar = priors['LOGG*']
            # loggerr = np.sqrt(abs(priors['LOGG*_uperr']*priors['LOGG*_lowerr']))
            # sc = LDPSetCreator(teff=(tstar, terr), logg=(loggstar, loggerr), z=(fehstar, feherr), filters=filters)
            # ps = sc.create_profiles(nsamples=int(1e4))
            # cq,eq = ps.coeffs_qd(do_mc=True)

            tpars = {
                'rprs': rprs,
                'tmid':priors[p]['t0'] + event*priors[p]['period'],
                'inc':priors[p]['inc'],
                'ars':smaors,

                'per':priors[p]['period'],
                'ecc':priors[p]['ecc'],
                'omega': priors[p].get('omega',0),

                # non-linear limb darkening TODO
                'u0':0, 'u1':0, 'u2':0, 'u3':0,

                # quadratic detrending model
                # a0 + a1*t + a2*t^2
                'a0':np.median(aper), 'a1':0, 'a2':0
            }

            try:
                tpars['inc'] = hstwhitelight_sv['data'][p]['mcpost']['mean']['inc']
            except KeyError:
                tpars['inc'] = priors[p]['inc']

            # future impose smaller constraint on tmid ?

            # define free parameters
            if selftype == 'transit':
                mybounds = {
                    'rprs':[0,1.25*tpars['rprs']],
                    'tmid':[tpars['tmid']-10./(24*60), tpars['tmid']+10./(24*60)],
                    'ars':[tpars['ars_lowerr'], tpars['ars_uperr']],
                    'a0':[min(aper), max(aper)]
                }
            elif selftype == 'eclipse':
                mybounds = {
                    'rprs':[0,0.5*tpars['rprs']],
                    'tmid':[min(subt),max(subt)],
                    'ars':[tpars['ars_lowerr'], tpars['ars_uperr']],
                    'a0':[min(aper), max(aper)]
                }

            # switch later
            myfit = pc_fitter(subt, aper, aper_err, tpars, mybounds, [], mode=method)

            # write fit to state vector
            terrs = {}
            for k in myfit.bounds.keys():
                tpars[k] = myfit.parameters[k]
                terrs[k] = myfit.errors[k]

            out['data'][p].append({})
            out['data'][p][ec]['time'] = subt
            out['data'][p][ec]['flux'] = aper
            out['data'][p][ec]['err'] = aper_err
            out['data'][p][ec]['ramp_num'] = rnum
            try:
                # keys specific to nested sampling
                del myfit.results['bound']
                out['data'][p][ec]['results'] = myfit.results
                out['data'][p][ec]['quantiles'] = myfit.quantiles
            except KeyError:
                print('no nested values found')
            out['data'][p][ec]['model'] = myfit.model
            out['data'][p][ec]['transit'] = myfit.transit
            out['data'][p][ec]['residuals'] = myfit.residuals
            out['data'][p][ec]['detrended'] = myfit.detrended

            out['data'][p][ec]['pars'] = copy.deepcopy(tpars)
            out['data'][p][ec]['errs'] = copy.deepcopy(terrs)

            # state vectors for classifer
            z, _phase = datcore.time2z(subt, tpars['inc'], tpars['tmid'], tpars['ars'], tpars['per'], tpars['ecc'])
            out['data'][p][ec]['postsep'] = z
            out['data'][p][ec]['allwhite'] = myfit.detrended
            out['data'][p][ec]['postlc'] = myfit.transit
            out['STATUS'].append(True)
            wl = True
            ec+=1
    return wl

def jwst_niriss_spectrum(nrm, fin, out, selftype, wht, method='lm'):
    '''
    K. PEARSON: multi-wavelength transit fitting - priors from whitelight
    '''
    spec = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = list(nrm['data'].keys())

    for p in planetloop:
        out['data'][p] = []

        # extract data based on phase
        if selftype == 'transit':
            phase = (nrm['data'][p]['TIME'] - fin['priors'][p]['t0'])/fin['priors'][p]['period']
        elif selftype == 'eclipse':
            priors = fin['priors']
            w = priors[p].get('omega',0)
            tme = priors[p]['t0']+ priors[p]['period']*0.5 * (1 + priors[p]['ecc']*(4./np.pi)*np.cos(np.deg2rad(w)))
            phase = (nrm['data'][p]['TIME'] - tme)/fin['priors'][p]['period']

        # loop through epochs
        ec = 0  # event counter
        for event in nrm['data'][p][selftype]:
            print('processing event:',event)

            # compute phase + priors
            smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
            # smaors_up = (priors[p]['sma']+3*priors[p]['sma_uperr'])/(priors['R*']-abs(priors['R*_lowerr']))/ssc['Rsun/AU']
            # smaors_lo = (priors[p]['sma']-abs(3*priors[p]['sma_lowerr']))/(priors['R*']+priors['R*_uperr'])/ssc['Rsun/AU']

            # tmid = priors[p]['t0'] + event*priors[p]['period']

            # to do: update duration for eccentric orbits
            # https://arxiv.org/pdf/1001.2010.pdf eq 16
            tdur = priors[p]['period']/(np.pi)/smaors
            rprs = (priors[p]['rp']*7.1492e7) / (priors['R*']*6.955e8)
            # inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
            w = priors[p].get('omega',0)

            # mask out data by event type
            pmask = (phase > event-1.5*tdur/priors[p]['period']) & (phase < event+1.5*tdur/priors[p]['period'])

            # extract data based on phase
            subt = nrm['data'][p]['TIME'][pmask]
            rnum = nrm['data'][p]['RAMP_NUM'][pmask]

            # alloc data to save each wavelength to
            out['data'][p].append({})
            out['data'][p][ec]['wave'] = []
            out['data'][p][ec]['rprs'] = []
            out['data'][p][ec]['rprs_err'] = []
            out['data'][p][ec]['time'] = []
            out['data'][p][ec]['flux'] = []
            out['data'][p][ec]['detrended'] = []
            out['data'][p][ec]['transit'] = []
            out['data'][p][ec]['flux_err'] = []
            out['data'][p][ec]['std'] = []
            out['data'][p][ec]['residuals'] = []

            # for each wavelength bin
            bs = 16  # binsize
            for wl in range(5,nrm['data'][p]['SPEC'][pmask].shape[1]-5-bs,bs):

                aper = nrm['data'][p]['SPEC'][pmask][:,wl:wl+bs].sum(1)
                aper_err = np.sqrt(aper)

                # wave solution is assumed to be the same for all images
                wmin = min(nrm['data'][p]['WAVE'][0][wl],nrm['data'][p]['WAVE'][0][wl+bs])
                wmax = max(nrm['data'][p]['WAVE'][0][wl],nrm['data'][p]['WAVE'][0][wl+bs])

                # normalize each ramp by OoT baseline
                # for r in np.unique(rnum):
                #    rmask = rnum==r
                #    ootmask = (phase[pmask] < event-.55*tdur/priors[p]['period']) | (phase[pmask] > event+.55*tdur/priors[p]['period'])
                # future fit for normalization offsets
                # aper_err[rmask] /= np.median(aper[ootmask&rmask])
                # aper[rmask] /= np.median(aper[ootmask&rmask])
                # future
                # sigma clip individual time series
                # diagnostic
                # plt.scatter(subt,aper,c=rnum,marker='.'); plt.colorbar(); plt.show()

                # can't solve for wavelengths greater than below
                # whiteld = createldgrid([2.5],[2.6], priors, segmentation=int(10), verbose=verbose)
                # whiteld = createldgrid([wmin],[wmax], priors, segmentation=int(1), verbose=verbose)

                # LDTK breaks for Spitzer https://github.com/hpparvi/ldtk/issues/11
                # filters = [BoxcarFilter('a', 3150, 3950)]
                # tstar = priors['T*']
                # terr = np.sqrt(abs(priors['T*_uperr']*priors['T*_lowerr']))
                # fehstar = priors['FEH*']
                # feherr = np.sqrt(abs(priors['FEH*_uperr']*priors['FEH*_lowerr']))
                # loggstar = priors['LOGG*']
                # loggerr = np.sqrt(abs(priors['LOGG*_uperr']*priors['LOGG*_lowerr']))
                # sc = LDPSetCreator(teff=(tstar, terr), logg=(loggstar, loggerr), z=(fehstar, feherr), filters=filters)
                # ps = sc.create_profiles(nsamples=int(1e4))
                # cq,eq = ps.coeffs_qd(do_mc=True)

                # use last ramp
                rmask = rnum==np.max(rnum)
                aper = aper[rmask]
                aper_err = aper_err[rmask]
                subtt = subt[rmask]

                tpars = {
                    'rprs': wht['data'][p][ec]['pars']['rprs'],
                    'tmid':wht['data'][p][ec]['pars']['tmid'],

                    'inc':wht['data'][p][ec]['pars']['inc'],
                    'ars':wht['data'][p][ec]['pars']['ars'],

                    'per':priors[p]['period'],
                    'ecc':priors[p]['ecc'],
                    'omega': priors[p].get('omega',0),

                    # non-linear limb darkening TODO
                    'u0':0, 'u1':0, 'u2':0, 'u3':0,

                    # quadratic detrending model
                    # a0 + a1*t + a2*t^2
                    'a0':np.median(aper), 'a1':0, 'a2':0
                }

                # define free parameters
                if selftype == 'transit':
                    mybounds = {
                        'rprs':[
                            # fluxuate by +/- 2000 ppm
                            (rprs**2 - 1000/1e6)**0.5,
                            (rprs**2 + 1000/1e6)**0.5
                        ],
                        'a0':[min(aper), max(aper)]
                    }
                elif selftype == 'eclipse':
                    mybounds = {
                        'rprs':[
                            (rprs**2 - 500/1e6)**0.5,
                            (rprs**2 + 500/1e6)**0.5
                        ],
                        'a0':[min(aper), max(aper)]
                    }

                myfit = pc_fitter(subtt, aper, aper_err, tpars, mybounds,[], mode=method)

                # write to SV
                out['data'][p][ec]['wave'].append(0.5*(wmin+wmax))
                out['data'][p][ec]['rprs'].append(myfit.parameters['rprs'])
                out['data'][p][ec]['rprs_err'].append(myfit.errors['rprs'])
                out['data'][p][ec]['time'].append(myfit.time)
                out['data'][p][ec]['flux'].append(myfit.data)
                out['data'][p][ec]['detrended'].append(myfit.detrended)
                out['data'][p][ec]['transit'].append(myfit.transit)
                out['data'][p][ec]['flux_err'].append(myfit.dataerr)
                out['data'][p][ec]['std'].append(np.std(myfit.residuals))
                out['data'][p][ec]['residuals'].append(myfit.residuals)

            # copy format to match HST
            out['data'][p][ec]['ES'] = np.copy(out['data'][p][ec]['rprs'])
            out['data'][p][ec]['ESerr'] = np.copy(out['data'][p][ec]['rprs_err'])
            out['data'][p][ec]['WB'] = np.copy(out['data'][p][ec]['wave'])

            out['STATUS'].append(True)
            spec = True
            ec+=1

    return spec

def lightcurve_spitzer(nrm, fin, out, selftype, fltr, hstwhitelight_sv):
    '''
    K. PEARSON: modeling of transits and eclipses from Spitzer
    '''
    wl= False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = list(nrm['data'].keys())

    for p in planetloop:

        out['data'][p] = []

        # extract data based on phase
        if selftype == 'transit':
            phase = (nrm['data'][p]['TIME'] - fin['priors'][p]['t0'])/fin['priors'][p]['period']
        elif selftype == 'eclipse':
            priors = fin['priors']
            w = priors[p].get('omega',0)
            tme = priors[p]['t0']+ priors[p]['period']*0.5 * (1 + priors[p]['ecc']*(4./np.pi)*np.cos(np.deg2rad(w)))
            phase = (nrm['data'][p]['TIME'] - tme)/fin['priors'][p]['period']

        # loop through epochs
        ec = 0  # event counter
        for event in nrm['data'][p][selftype]:
            try:
                print('processing event:',event)

                # compute phase + priors
                smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
                smaors_up = (priors[p]['sma']+priors[p]['sma_uperr'])/(priors['R*']-abs(priors['R*_lowerr']))/ssc['Rsun/AU']
                smaors_lo = (priors[p]['sma']-abs(priors[p]['sma_lowerr']))/(priors['R*']+priors['R*_uperr'])/ssc['Rsun/AU']

                # to do: update duration for eccentric orbits
                # https://arxiv.org/pdf/1001.2010.pdf eq 16
                tdur = priors[p]['period']/(np.pi)/smaors
                rprs = (priors[p]['rp']*7.1492e7) / (priors['R*']*6.955e8)
                # inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
                w = priors[p].get('omega',0)
                tmid = priors[p]['period']*event + priors[p]['t0']

                # mask out data by event type
                pmask = (phase > event-1.5*tdur/priors[p]['period']) & (phase < event+1.5*tdur/priors[p]['period'])

                # extract aperture photometry data
                subt = nrm['data'][p]['TIME'][pmask]
                aper = nrm['data'][p]['PHOT'][pmask]
                aper_err = np.sqrt(aper)

                priors[p]['ars'] = smaors
                fpfs = eclipse_ratio(priors, p, fltr)

                tpars = {
                    # Star
                    'T*':priors['T*'],

                    # transit
                    'rprs': rprs,
                    'ars': smaors,
                    'tmid':tmid,
                    'per': priors[p]['period'],
                    'inc': priors[p]['inc'],

                    # eclipse
                    'fpfs': fpfs,
                    'omega': priors['b'].get('omega',0),
                    'ecc': priors['b']['ecc'],

                    # limb darkening (nonlinear - exotethys - pylightcurve)
                    'u0':priors[p].get('u0',0),
                    'u1':priors[p].get('u1',0),
                    'u2':priors[p].get('u2',0),
                    'u3':priors[p].get('u3',0),

                    # phase curve amplitudes
                    'c0':0, 'c1':0, 'c2':0, 'c3':0, 'c4':0
                }

                # remove first 30 min of data after any big gaps, ramp
                tmask = np.ones(subt.shape).astype(bool)

                smask = np.argsort(subt)
                dts = np.diff(subt[smask])
                dmask = dts > (2./(24*60))
                ndt = int(15./(24*60*dts.mean()))*2+1
                tmask[0:int(2*ndt)] = False  # mask first 30 minutes of data

                # feature engineer a ramp correction
                ramp = np.exp(-np.arange(len(tmask))*dts.mean()/((subt.max()-subt.min())/20))
                # reset ramp after big gap
                for idx in np.argwhere(dmask).flatten():
                    ramp[idx-1:] = np.exp(-np.arange(len(ramp[idx-1:]))*dts.mean())

                # gather detrending parameters
                wxa = nrm['data'][p]['WX'][pmask]
                wya = nrm['data'][p]['WY'][pmask]
                npp = nrm['data'][p]['NOISEPIXEL'][pmask]

                # remove zeros
                zmask = aper != 0
                aper = aper[zmask]
                aper_err = aper_err[zmask]
                subt = subt[zmask]
                wxa = wxa[zmask]
                wya = wya[zmask]
                npp = npp[zmask]
                ramp = ramp[zmask]

                syspars = np.array([wxa,wya,npp,ramp]).T

                # 10 minute time scale
                nneighbors = int(10./24./60./np.mean(np.diff(subt)))
                print("N neighbors:",nneighbors)
                print("N datapoints:", len(subt))

                # define free parameters
                if selftype == 'transit':
                    mybounds = {
                        'rprs':[0,1.25*tpars['rprs']],
                        'tmid':[tmid-0.01,tmid+0.01],
                        'ars':[smaors_lo,smaors_up]
                    }
                    myfit = pc_fitter(subt, aper, aper_err, tpars, mybounds, syspars, neighbors=nneighbors)
                elif selftype == 'eclipse':
                    mybounds = {
                        'rprs':[0,0.5*tpars['rprs']],
                        'tmid':[tme-0.01,tme+0.01],
                        'ars':[smaors_lo,smaors_up]
                    }
                    myfit = pc_fitter(subt, aper, aper_err, tpars, mybounds, syspars, neighbors=nneighbors)

                # copy best fit parameters and uncertainties
                for k in myfit.bounds.keys():
                    print(f" {k} = {myfit.parameters[k]:.6f} +/- {myfit.errors[k]:.6f}")

                out['data'][p].append({})
                out['data'][p][ec]['time'] = subt
                out['data'][p][ec]['flux'] = aper
                out['data'][p][ec]['err'] = aper_err
                out['data'][p][ec]['xcent'] = wxa
                out['data'][p][ec]['ycent'] = wya
                out['data'][p][ec]['npp'] = npp
                out['data'][p][ec]['wf'] = myfit.wf
                out['data'][p][ec]['model'] = myfit.model
                out['data'][p][ec]['transit'] = myfit.transit
                out['data'][p][ec]['residuals'] = myfit.residuals
                out['data'][p][ec]['detrended'] = myfit.detrended
                out['data'][p][ec]['filter'] = fltr
                out['data'][p][ec]['final_pars'] = copy.deepcopy(myfit.parameters)
                out['data'][p][ec]['final_errs'] = copy.deepcopy(myfit.errors)

                # extract plot data for states.py
                def save_plot(plotfn):
                    fig,_ = plotfn()
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png')
                    plt.close(fig)
                    return buf.getvalue()

                out['data'][p][ec]['plot_bestfit'] = save_plot(myfit.plot_bestfit)
                out['data'][p][ec]['plot_posterior'] = save_plot(myfit.plot_posterior)
                out['data'][p][ec]['plot_pixelmap'] = save_plot(myfit.plot_pixelmap)

                ec += 1
                out['STATUS'].append(True)
                wl = True

                pass
            except NameError as e:
                print("Error:",e)
                out['data'][p].append({"error":e})

                try:
                    out['data'][p][ec]['aper_time'] = subt
                    out['data'][p][ec]['aper_flux'] = aper
                    out['data'][p][ec]['aper_err'] = aper_err
                    out['data'][p][ec]['aper_xcent'] = wxa
                    out['data'][p][ec]['aper_ycent'] = wya
                    out['data'][p][ec]['aper_npp'] = npp
                    out['data'][p][ec]['aper_pars'] = copy.deepcopy(tpars)
                except NameError:
                    pass
                pass
        pass
    return wl

def spitzer_spectrum(wht, out, ext):
    '''
    K. PEARSON put data in same format as HST spectrum
    '''

    update = False
    for p in wht['data'].keys():
        out['data'][p] = {}
        out['data'][p]['WB'] = []
        out['data'][p]['ES'] = []
        out['data'][p]['ESerr'] = []

        for i in range(len(wht['data'][p])):
            if '36' in ext:
                out['data'][p]['WB'].append(3.6)
            elif '45' in ext:
                out['data'][p]['WB'].append(4.5)
            elif '58' in ext:
                out['data'][p]['WB'].append(5.8)
            elif '80' in ext:
                out['data'][p]['WB'].append(8.0)
            else:
                continue

            out['data'][p]['ES'].append(wht['data'][p][i]['final_pars']['rprs'])  # rp/rs
            out['data'][p]['ESerr'].append(wht['data'][p][i]['final_errs']['rprs'])  # upper bound

            update = True
    return update

def jwst_lightcurve(sv, savedir=None, suptitle=''):
    '''jwst_lightcurve ds'''
    f = plt.figure(figsize=(12,7))
    # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
    ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
    ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
    axs = [ax_lc, ax_res]

    bt, bf, _ = time_bin(sv['time'], sv['detrended'])

    axs[0].errorbar(sv['time'],  sv['detrended'], yerr=np.std(sv['residuals'])/np.median(sv['flux']), ls='none', marker='.', color='black', zorder=1, alpha=0.5)
    axs[0].plot(bt,bf,'c.',alpha=0.5,zorder=2)
    axs[0].plot(sv['time'], sv['transit'], 'r-', zorder=3)
    axs[0].set_xlabel("Time [day]")
    axs[0].set_ylabel("Relative Flux")
    axs[0].grid(True,ls='--')

    axs[1].plot(sv['time'], sv['residuals']/np.median(sv['flux'])*1e6, 'k.', alpha=0.5)
    bt, br, _ = time_bin(sv['time'], sv['residuals']/np.median(sv['flux'])*1e6)
    axs[1].plot(bt,br,'c.',alpha=0.5,zorder=2)
    axs[1].set_xlabel("Time [day]")
    axs[1].set_ylabel("Residuals [ppm]")
    axs[1].grid(True,ls='--')
    plt.tight_layout()

    if savedir:
        plt.savefig(savedir+suptitle+".png")
        plt.close()
    return f

def composite_spectrum(SV, target, p='b'):
    '''
    K. PEARSON combine the filters into one plot
    '''
    f,ax = plt.subplots(figsize=(15,7))
    colors = ['pink','red','green','cyan','blue','purple']
    ci = 0
    # keys need to be the same as active filters
    for name in SV.keys():
        if name in ('data', 'STATUS'):
            continue
        SV1 = SV[name]
        if SV1['data'].keys():
            fname = name.split('-')[1] + ' ' + name.split('-')[3]
            vspectrum=np.array(SV1['data'][p]['ES'])
            specwave=np.array(SV1['data'][p]['WB'])
            specerr=np.array(SV1['data'][p]['ESerr'])
            specerr = abs(vspectrum**2 - (vspectrum + specerr)**2)
            vspectrum = vspectrum**2
            ax.errorbar(specwave, 1e2*vspectrum, fmt='.', yerr=1e2*specerr, alpha=0.2, color=colors[ci])
            if 'Spitzer' in name:
                if specwave.shape[0] > 0:
                    waveb = np.mean(specwave)
                    specb = np.nansum(vspectrum/(specerr**2))/np.nansum(1./(specerr**2))
                    errb = np.nanmedian((specerr))/np.sqrt(specwave.shape[0])
                    ax.errorbar(waveb, 1e2*specb, fmt='^', yerr=1e2*errb, color=colors[ci], label=fname)
            else:
                # Smooth spectrum
                binsize = 4
                nspec = int(specwave.size/binsize)
                minspec = np.nanmin(specwave)
                maxspec = np.nanmax(specwave)
                scale = (maxspec - minspec)/(1e0*nspec)
                wavebin = scale*np.arange(nspec) + minspec
                deltabin = np.diff(wavebin)[0]
                cbin = wavebin + deltabin/2e0
                specbin = []
                errbin = []
                for eachbin in cbin:
                    select = specwave < (eachbin + deltabin/2e0)
                    select = select & (specwave >= (eachbin - deltabin/2e0))
                    select = select & np.isfinite(vspectrum)
                    if np.sum(np.isfinite(vspectrum[select])) > 0:
                        specbin.append(np.nansum(vspectrum[select]/(specerr[select]**2))/np.nansum(1./(specerr[select]**2)))
                        errbin.append(np.nanmedian((specerr[select]))/np.sqrt(np.sum(select)))
                        pass
                    else:
                        specbin.append(np.nan)
                        errbin.append(np.nan)
                        pass
                    pass
                waveb = np.array(cbin)
                specb = np.array(specbin)
                errb = np.array(errbin)
                ax.errorbar(waveb, 1e2*specb, fmt='^', yerr=1e2*errb, color=colors[ci], label=fname)

        try:
            if ('Hs' in SV1['data'][p]) and ('RSTAR' in SV1['data'][p]):
                rp0hs = np.sqrt(np.nanmedian(vspectrum))
                Hs = SV1['data'][p]['Hs'][0]
                # Retro compatibility for Hs in [m]
                if Hs > 1: Hs = Hs/(SV1['data'][p]['RSTAR'][0])
                ax2 = ax.twinx()
                ax2.set_ylabel('$\\Delta$ [Hs]', fontsize=14)
                axmin, axmax = ax.get_ylim()
                ax2.set_ylim((np.sqrt(1e-2*axmin) - rp0hs)/Hs,(np.sqrt(1e-2*axmax) - rp0hs)/Hs)
                f.tight_layout()
                pass
        except (KeyError,ValueError):
            # print("couldn't plot scale height")
            pass

        ci += 1

    ax.set_title(target +" "+ p, fontsize=14)
    ax.set_xlabel(str('Wavelength [$\\mu m$]'), fontsize=14)
    ax.set_ylabel(str('$(R_p/R_*)^2$ [%]'), fontsize=14)
    ax.set_xscale('log')
    ax.legend(loc='best', shadow=False, frameon=False, fontsize='20', scatterpoints=1)
    return f

def hstspectrum(out, fltrs):
    '''MERGE SPECTRUM STIS AND WFC3 AND PLACE AT THE END OF THE SPECTRUM LIST'''
    exospec = False
#     allnames = [SV.__name for SV in spec_list] #all names of the filters
#     allnames = np.array(allnames)
    allstatus = [SV['STATUS'][-1] for SV in out]  # list of True or False
    allstatus = np.array(allstatus)
    allwav = []
    allwav_lw = []
    allwav_up = []
    allspec = []
    allspec_err = []
    allfltrs = []
    checkwav = []
    for ids,status in enumerate(allstatus[:-1]):
        valid = 'STIS' in fltrs[ids]
        valid = valid or 'WFC3' in fltrs[ids]
        if status and valid:
            for planet in out[ids]['data'].keys():
                wav = out[ids]['data'][planet]['WB']
                allwav.extend(out[ids]['data'][planet]['WB'])
                allwav_lw.extend(out[ids]['data'][planet]['WBlow'])
                allwav_up.extend(out[ids]['data'][planet]['WBup'])
                allspec.extend(out[ids]['data'][planet]['ES'])
                allspec_err.extend(out[ids]['data'][planet]['ESerr'])
                # for i in range(0,len(wav)):
                for w in wav:
                    allfltrs.append(fltrs[ids])
                    checkwav.append(w)
                    pass
                # order everything using allwav before saving them
                out[-1]['data'][planet] = {'WB': np.sort(np.array(allwav)), 'WBlow': [x for _,x in sorted(zip(allwav,allwav_lw))], 'WBup': [x for _,x in sorted(zip(allwav,allwav_up))], 'ES': [x for _,x in sorted(zip(allwav,allspec))], 'ESerr': [x for _,x in sorted(zip(allwav,allspec_err))], 'Fltrs': [x for _,x in sorted(zip(allwav,allfltrs))], 'Hs': out[ids]['data'][planet]['Hs']}
            exospec = True  # return if all inputs were empty
            out[-1]['STATUS'].append(True)
    return exospec
