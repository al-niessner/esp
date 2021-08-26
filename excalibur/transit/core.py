# -- IMPORTS -- ------------------------------------------------------
import dawgie

# pylint: disable=import-self
import excalibur.data.core as datcore
import excalibur.system.core as syscore
import excalibur.cerberus.core as crbcore
import excalibur.transit.core

import re

import copy
import ctypes
import requests
import logging
import random
import lmfit as lm

import dynesty
import dynesty.plotting
from dynesty.utils import resample_equal

import pymc3 as pm
log = logging.getLogger(__name__)
pymc3log = logging.getLogger('pymc3')
pymc3log.setLevel(logging.ERROR)

import matplotlib.pyplot as plt

import scipy.constants as cst
from scipy import spatial
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from scipy.optimize import least_squares

import theano.tensor as tt
import theano.compile.ops as tco

import numpy as np

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
    def is_mime(): return True

    @property
    def profile_mu(self): return self._mu
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
            viss = [s for s in np.array(spectra)[selv]]
            visw = np.array([w for w in np.array(wave)[selv]])
            cwave, _t = tplbuild(viss, visw, vrange, disp[selv]*1e-4, medest=True)
            # OUT OF TRANSIT ORBITS SELECTION --------------------------------------------
            selvoot = selv & np.array([(test in pureoot) for test in orbits])
            selvoot = selvoot & (abs(zoot) > (1e0 + rpors))
            voots = [s for s in np.array(spectra)[selvoot]]
            vootw = [w for w in np.array(wave)[selvoot]]
            ivoots = []
            for s, w in zip(voots, vootw):
                itps = np.interp(np.array(cwave), w, s, left=np.nan, right=np.nan)
                ivoots.append(itps)
                pass
            pureootext = pureoot.copy()
            if firstorb in orbits[selv]:
                fovoots = [s for s in np.array(spectra)[selv]]
                fovootw = [w for w in np.array(wave)[selv]]
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
                                if str(closest) in witp.keys():
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
            mcpost = pm.summary(trace)
            pass
        mctrace = {}
        for key in mcpost['mean'].keys():
            if len(key.split('[')) > 1:  # change PyMC3.8 key format to previous
                pieces = key.split('[')
                key = '{}__{}'.format(pieces[0], pieces[1].strip(']'))
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
                    posttk = np.nanmedian(mctrace['dtk__%i' % ttvindex])
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
                                    vslope=np.nanmedian(mctrace['vslope__%i' % i]),
                                    vitcp=1e0,
                                    oslope=np.nanmedian(mctrace['oslope__%i' % i]),
                                    oitcp=np.nanmedian(mctrace['oitcp__%i' % i])))
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
                                    vslope=np.nanmedian(mctrace['vslope__%i' % i]),
                                    vitcp=1e0,
                                    oslope=np.nanmedian(mctrace['oslope__%i' % i]),
                                    oitcp=np.nanmedian(mctrace['oitcp__%i' % i])))
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
            mcpost = pm.summary(trace)
            pass
        mctrace = {}
        for key in mcpost['mean'].keys():
            if len(key.split('[')) > 1:  # change PyMC3.8 key format to previous
                pieces = key.split('[')
                key = '{}__{}'.format(pieces[0], pieces[1].strip(']'))
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
                                vslope=np.nanmedian(mctrace['vslope__%i' % i]),
                                vitcp=1e0,
                                oslope=np.nanmedian(mctrace['oslope__%i' % i]),
                                oitcp=np.nanmedian(mctrace['oitcp__%i' % i])))
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
    for i in range(0,len(allcl.T)):
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
    return dawgie.VERSION(1,3,0)

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
            mixratio, fH2, fHe = crbcore.crbce(pressure, eqtemp)
            mmw, fH2, fHe = crbcore.getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
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
                    if lowstart < 0: lowstart = 0
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
                mcpost = pm.summary(trace)
                pass
            # Exclude first channel with Uniform prior
            if not startflag:
                # save MCMC samples in SV
                mctrace = {}
                mcests = {}
                for key in mcpost['mean'].keys():
                    if len(key.split('[')) > 1:  # change PyMC3.8 key format to previous
                        pieces = key.split('[')
                        key = '{}__{}'.format(pieces[0], pieces[1].strip(']'))
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

                def get_ests(n, v):  # for param get all visit param values as list
                    return [mcests['{}__{}'.format(n, i)] for i in range(len(v))]

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
        mixratio, fH2, fHe = crbcore.crbce(pressure, eqtemp)
        mmw, fH2, fHe = crbcore.getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
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
            if lowstart < 0: lowstart = 0
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

            badmask[omask] = photmask | xmask | ymask | nmask

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

# Computes Hasting's polynomial approximation for the complete
# elliptic integral of the first (ek) and second (kk) kind
def ellke(k):
    m1=1.-k**2
    np.logm1 = np.log(m1)

    a1=0.44325141463
    a2=0.06260601220
    a3=0.04757383546
    a4=0.01736506451
    b1=0.24998368310
    b2=0.09200180037
    b3=0.04069697526
    b4=0.00526449639
    ee1=1.+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*(-np.logm1)
    ek = ee1+ee2

    a0=1.38629436112
    a1=0.09666344259
    a2=0.03590092383
    a3=0.03742563713
    a4=0.01451196212
    b0=0.5
    b1=0.12498593597
    b2=0.06880248576
    b3=0.03328355346
    b4=0.00441787012
    ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
    ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*np.logm1
    kk = ek1-ek2

    return [ek,kk]

# Computes the complete elliptical integral of the third kind using
# the algorithm of Bulirsch (1965):
def ellpic_bulirsch(n,k):
    kc=np.sqrt(1.-k**2); la=n+1.
    if min(la) < 0.:
        # print('Negative l')
        pass
    m0=1.; c=1.; la=np.sqrt(la); d=1./la; e=kc
    while 1:
        f = c; c = d/la+c; g = e/la; d = 2.*(f*g+d)
        la = g + la; g = m0; m0 = kc + m0
        if max(abs(1.-kc/g)) > 1.e-8:
            kc = 2*np.sqrt(e); e=kc*m0
        else: return 0.5*np.pi*(c*m0+d)/(m0*(m0+la))
        pass
    pass

#   Python translation of IDL code.
#   This routine computes the lightcurve for occultation of a
#   quadratically limb-darkened source without microlensing.  Please
#   cite Mandel & Agol (2002) and Eastman & Agol (2008) if you make use
#   of this routine in your research.  Please report errors or bugs to
#   jdeast@astronomy.ohio-state.edu

# GMR: This thing was put in excalibur but never ran right,
# there are syntax errors everywhere

# Yo Eastman use pylint next time
def occultquad(z,u1,u2,p0):  # pylint: disable=too-many-return-statements
    nz = np.size(z)
    lambdad = np.zeros(nz)
    etad = np.zeros(nz)
    lambdae = np.zeros(nz)
    omega=1.-u1/3.-u2/6.

    # tolerance for double precision equalities
    # special case integrations
    tol = 1e-14

    p = abs(p0)

    z = np.where(abs(p-z) < tol,p,z)
    z = np.where(abs((p-1)-z) < tol,p-1.,z)
    z = np.where(abs((1-p)-z) < tol,1.-p,z)
    z = np.where(z < tol,0.,z)

    x1=(p-z)**2.
    x2=(p+z)**2.
    x3=p**2.-z**2.

    # trivial case of no planet
    if p <= 0.:
        muo1 = np.zeros(nz) + 1.
        mu0 = np.zeros(nz) + 1.
        return [muo1,mu0]

    # Case 1 - the star is unocculted:
    # only consider points with z lt 1+p
    notusedyet = np.where(z < (1. + p))
    notusedyet = notusedyet[0]
    if np.size(notusedyet) == 0:
        muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega
        mu0=1.-lambdae
        return [muo1,mu0]

    # Case 11 - the  source is completely occulted:
    if p >= 1.:
        occulted = np.where(z[notusedyet]<=p-1)  # ,complement=notused2)
        if np.size(occulted) != 0:
            ndxuse = notusedyet[occulted]
            etad[ndxuse] = 0.5  # corrected typo in paper
            lambdae[ndxuse] = 1.
            # lambdad = 0 already
            notused2 = np.where(z[notusedyet] > p-1)
            if np.size(notused2) == 0:
                muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega
                mu0=1.-lambdae
                return [muo1,mu0]
            notusedyet = notusedyet[notused2]

    # Case 2, 7, 8 - ingress/egress (uniform disk only)
    inegressuni = np.where((z[notusedyet] >= abs(1.-p)) & (z[notusedyet] < 1.+p))
    if np.size(inegressuni) != 0:
        ndxuse = notusedyet[inegressuni]
        tmp = (1.-p**2.+z[ndxuse]**2.)/2./z[ndxuse]
        tmp = np.where(tmp > 1.,1.,tmp)
        tmp = np.where(tmp < -1.,-1.,tmp)
        kap1 = np.arccos(tmp)
        tmp = (p**2.+z[ndxuse]**2-1.)/2./p/z[ndxuse]
        tmp = np.where(tmp > 1.,1.,tmp)
        tmp = np.where(tmp < -1.,-1.,tmp)
        kap0 = np.arccos(tmp)
        tmp = 4.*z[ndxuse]**2-(1.+z[ndxuse]**2-p**2)**2
        tmp = np.where(tmp < 0,0,tmp)
        lambdae[ndxuse] = (p**2*kap0+kap1 - 0.5*np.sqrt(tmp))/np.pi
        # eta_1
        etad[ndxuse] = 1./2./np.pi*(kap1+p**2*(p**2+2.*z[ndxuse]**2)*kap0-(1.+5.*p**2+z[ndxuse]**2)/4.*np.sqrt((1.-x1[ndxuse])*(x2[ndxuse]-1.)))

    # Case 5, 6, 7 - the edge of planet lies at origin of star
    ocltor = np.where(z[notusedyet] == p)  # complement=notused3)
    _t = np.where(z[notusedyet] == p)
    if np.size(ocltor) != 0:
        ndxuse = notusedyet[ocltor]
        if p < 0.5:
            # Case 5
            q=2.*p  # corrected typo in paper (2k -> 2p)
            Ek,Kk = ellke(q)
            # lambda_4
            lambdad[ndxuse] = 1./3.+2./9./np.pi*(4.*(2.*p**2-1.)*Ek+(1.-4.*p**2)*Kk)
            # eta_2
            etad[ndxuse] = p**2/2.*(p**2+2.*z[ndxuse]**2)
            lambdae[ndxuse] = p**2  # uniform disk
        elif p > 0.5:
            # Case 7
            q=0.5/p  # corrected typo in paper (1/2k -> 1/2p)
            Ek,Kk = ellke(q)
            # lambda_3
            lambdad[ndxuse] = 1./3.+16.*p/9./np.pi*(2.*p**2-1.)*Ek-(32.*p**4-20.*p**2+3.)/9./np.pi/p*Kk
            # etad = eta_1 already
        else:
            # Case 6
            lambdad[ndxuse] = 1./3.-4./np.pi/9.
            etad[ndxuse] = 3./32.
        notused3 = np.where(z[notusedyet] != p)
        if np.size(notused3) == 0:
            muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega
            mu0=1.-lambdae
            return [muo1,mu0]
        notusedyet = notusedyet[notused3]

    # Case 2, Case 8 - ingress/egress (with limb darkening)
    inegress = np.where(((z[notusedyet] > 0.5+abs(p-0.5)) & (z[notusedyet] < 1.+p)) | ((p > 0.5) & (z[notusedyet] > abs(1.-p)) & (z[notusedyet] < p)))  # complement=notused4)
    if np.size(inegress) != 0:

        ndxuse = notusedyet[inegress]
        q=np.sqrt((1.-x1[ndxuse])/(x2[ndxuse]-x1[ndxuse]))
        Ek,Kk = ellke(q)
        n=1./x1[ndxuse]-1.

        # lambda_1:
        lambdad[ndxuse]=2./9./np.pi/np.sqrt(x2[ndxuse]-x1[ndxuse])*(((1.-x2[ndxuse])*(2.*x2[ndxuse]+x1[ndxuse]-3.)-3.*x3[ndxuse]*(x2[ndxuse]-2.))*Kk+(x2[ndxuse]-x1[ndxuse])*(z[ndxuse]**2+7.*p**2-4.)*Ek-3.*x3[ndxuse]/x1[ndxuse]*ellpic_bulirsch(n,q))

        notused4 = np.where(((z[notusedyet] <= 0.5+abs(p-0.5)) | (z[notusedyet] >= 1.+p)) & ((p <= 0.5) | (z[notusedyet] <= abs(1.-p)) | (z[notusedyet] >= p)))
        if np.size(notused4) == 0:
            muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega
            mu0=1.-lambdae
            return [muo1,mu0]
        notusedyet = notusedyet[notused4]

    # Case 3, 4, 9, 10 - planet completely inside star
    if p < 1.:
        inside = np.where(z[notusedyet] <= (1.-p))  # complement=notused5)
        if np.size(inside) != 0:
            ndxuse = notusedyet[inside]

            # eta_2
            etad[ndxuse] = p**2/2.*(p**2+2.*z[ndxuse]**2)

            # uniform disk
            lambdae[ndxuse] = p**2

            # Case 4 - edge of planet hits edge of star
            edge = np.where(z[ndxuse] == 1.-p)  # complement=notused6)
            if np.size(edge[0]) != 0:
                # lambda_5
                lambdad[ndxuse[edge]] = 2./3./np.pi*np.arccos(1.-2.*p)-4./9./np.pi*np.sqrt(p*(1.-p))*(3.+2.*p-8.*p**2)
                if p > 0.5:
                    lambdad[ndxuse[edge]] -= 2./3.
                notused6 = np.where(z[ndxuse] != 1.-p)
                if np.size(notused6) == 0:
                    muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega
                    mu0=1.-lambdae
                    return [muo1,mu0]
                ndxuse = ndxuse[notused6[0]]

            # Case 10 - origin of planet hits origin of star
            origin = np.where(z[ndxuse] == 0)  # complement=notused7)
            if np.size(origin) != 0:
                # lambda_6
                lambdad[ndxuse[origin]] = -2./3.*(1.-p**2)**1.5
                notused7 = np.where(z[ndxuse] != 0)
                if np.size(notused7) == 0:
                    muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega
                    mu0=1.-lambdae
                    return [muo1,mu0]
                ndxuse = ndxuse[notused7[0]]

            q=np.sqrt((x2[ndxuse]-x1[ndxuse])/(1.-x1[ndxuse]))
            n=x2[ndxuse]/x1[ndxuse]-1.
            Ek,Kk = ellke(q)

            # Case 3, Case 9 - anynp.where in between
            # lambda_2
            lambdad[ndxuse] = 2./9./np.pi/np.sqrt(1.-x1[ndxuse])*((1.-5.*z[ndxuse]**2+p**2+x3[ndxuse]**2)*Kk+(1.-x1[ndxuse])*(z[ndxuse]**2+7.*p**2-4.)*Ek-3.*x3[ndxuse]/x1[ndxuse]*ellpic_bulirsch(n,q))

        # if there are still unused elements, there's a bug in the code
        # (please report it)
        notused5 = np.where(z[notusedyet] > (1-p))
        if notused5[0].shape[0] != 0:
            print("ERROR: the following values of z didn't fit into a case:")
            return [-1,-1]

        muo1 =1.-((1.-u1-2.*u2)*lambdae+(u1+2.*u2)*(lambdad+2./3.*(p > z))+u2*etad)/omega
        mu0=1.-lambdae
        return [muo1,mu0]

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
# Try to load C code
try:

    try:
        # main pipeline
        lib_trans = np.ctypeslib.load_library('lib_transit.so','/lib')
    except OSError:
        # local pipeline:
        lib_trans = np.ctypeslib.load_library('lib_transit.so','/proj/sdp/lib')

    # load fn from library and define inputs
    occultquadC = lib_trans.occultquad

    # define 1d array pointer in python
    array_1d_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double,ndim=1,flags=['C_CONTIGUOUS','aligned'])

    # inputs
    occultquadC.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, ctypes.c_double,
                            ctypes.c_double, ctypes.c_double, array_1d_double]

    # no outputs, last *double input is saved over in C
    occultquadC.restype = None

    def transit_occultquad(t, values):
        time = np.require(t,dtype=ctypes.c_double,requirements='C')
        model = np.zeros(len(t),dtype=ctypes.c_double)
        model = np.require(model,dtype=ctypes.c_double,requirements='C')
        keys = ['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
        vals = [values[k] for k in keys]
        occultquadC(time, *vals, len(time), model)
        return model

except OSError:
    print("please load the docker image: esp_devel:C_LIB to use this feature")

    def transit_occultquad(time, values):
        '''
        K. PEARSON: use occultquad model for Spitzer b.c it's fast + data is noisy
        '''
        sep, _ = time2z(time, values['inc'], values['tmid'], values['ars'], values['per'], values['ecc'])
        model, _ = occultquad(abs(sep), values['u1'], values['u2'], values['rprs'])
        return model


def transit(time, values):
    sep,_ = time2z(time, values['inc'], values['tmid'], values['ars'], values['per'], values['ecc'])
    model = tldlc(abs(sep), values['rprs'], values['u0'], values['u1'], values['u2'], values['u3'])
    return model

def weightedflux(flux,gw,nearest):
    return np.sum(flux[nearest]*gw,axis=-1)

def gaussian_weights(X, w=None, neighbors=50, feature_scale=1000):
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

class lc_fitter:
    '''
    K. PEARSON
    transit fitting with quadratic detrend for JWST
    '''
    # pylint: disable=too-many-instance-attributes

    def __init__(self, time, data, dataerr, prior, bounds, mode='ns'):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        if mode == "lm":
            self.fit_LM()
        elif mode == "ns":
            self.fit_nested()
            return

    def fit_LM(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])

        def lc2min(pars):
            for i,par in enumerate(pars):
                self.prior[freekeys[i]] = par
            quad_model = self.prior['a0'] + self.prior['a1']*self.time + self.prior['a2']*self.time**2
            model = transit(self.time, self.prior)*quad_model
            return ((self.data-model)/self.dataerr)**2

        try:
            res = least_squares(
                lc2min, x0=[self.prior[k] for k in freekeys],
                bounds=[boundarray[:,0], boundarray[:,1]], jac='3-point',
                loss='linear', method='dogbox', xtol=None, ftol=1e-4, tr_options='exact')
        except ValueError:
            print("bounded  light curve fitting failed...check priors (e.g. estimated mid-transit time + orbital period)")

            for i,k in enumerate(freekeys):
                print(f"bound: [{boundarray[i,0]}, {boundarray[i,1]}] prior: {self.prior[k]}")

            print("removing bounds and trying again...")
            res = least_squares(lc2min, x0=[self.prior[k] for k in freekeys], method='lm', jac='3-point', loss='linear')

        self.parameters = copy.deepcopy(self.prior)
        self.errors = {}

        J = res.jac
        cov = np.linalg.inv(J.T.dot(J))
        std = np.sqrt(np.diagonal(cov))

        for i,k in enumerate(freekeys):
            self.parameters[k] = res.x[i]
            self.errors[k] = std[i]

        self.create_fit_variables()

    def create_fit_variables(self):
        # pylint: disable=attribute-defined-outside-init
        self.phase = (self.time-self.parameters['tmid'])/self.parameters['per']
        self.transit = transit(self.time, self.parameters)
        self.quad_model = self.parameters['a0'] + self.parameters['a1']*self.time + self.parameters['a2']*self.time**2
        self.model = self.transit * self.quad_model
        self.detrended = self.data / self.quad_model
        self.detrendederr = self.dataerr / self.quad_model
        self.residuals = self.data - self.model
        self.chi2 = np.sum(self.residuals**2/self.dataerr**2)
        self.bic = len(self.bounds) * np.log(len(self.time)) - 2*np.log(self.chi2)

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)

        def loglike(pars):
            # chi-squared
            for i,par in enumerate(pars):
                self.prior[freekeys[i]] = par
            quad_model = self.prior['a0'] + self.prior['a1']*self.time + self.prior['a2']*self.time**2
            model = transit(self.time, self.prior)*quad_model
            return -0.5 * np.sum(((self.data-model)/self.dataerr)**2)

        def prior_transform(upars):
            # transform unit cube to prior volume
            return (boundarray[:,0] + bounddiff*upars)

        dsampler = dynesty.NestedSampler(
            loglike, prior_transform, len(freekeys), dlogz=0.05,
            sample='unif', bound='multi', nlive=1000
        )
        dsampler.run_nested(maxiter=2e6, print_progress=False, dlogz=0.05, maxcall=2e6)
        self.results = dsampler.results

        # alloc data for best fit + error
        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)

        tests = [copy.deepcopy(self.prior) for i in range(6)]

        # Derive kernel density estimate for best fit
        weights = np.exp(self.results.logwt - self.results.logz[-1])
        samples = self.results['samples']
        logvol = self.results['logvol']
        wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
        logvol_grid = np.linspace(logvol[0], logvol[-1], 1000)  # resample
        wt_grid = wt_kde.pdf(-1*logvol_grid)  # evaluate KDE PDF
        self.weights = np.interp(-1*logvol, -1*logvol_grid, wt_grid)  # interpolate

        # errors + final values
        mean, cov = dynesty.utils.mean_and_cov(self.results.samples, weights)
        mean2, _cov2 = dynesty.utils.mean_and_cov(self.results.samples, self.weights)
        for i,fkey in enumerate(freekeys):
            self.errors[freekeys[i]] = cov[i,i]**0.5
            tests[0][freekeys[i]] = mean[i]
            tests[1][freekeys[i]] = mean2[i]

            counts, bins = np.histogram(samples[:,i], bins=100, weights=weights)
            mi = np.argmax(counts)
            tests[5][freekeys[i]] = bins[mi] + 0.5*np.mean(np.diff(bins))

            # finds median and +- 2sigma, will vary from mode if non-gaussian
            self.quantiles[freekeys[i]] = dynesty.utils.quantile(self.results.samples[:,i], [0.025, 0.5, 0.975], weights=weights)
            tests[2][freekeys[i]] = self.quantiles[freekeys[i]][1]

        # find minimum near weighted mean
        mask = (samples[:,0] < self.parameters[freekeys[0]]+2*self.errors[freekeys[0]]) & (samples[:,0] > self.parameters[freekeys[0]]-2*self.errors[freekeys[0]])
        bi = np.argmin(self.weights[mask])

        for i,fkey in enumerate(freekeys):
            tests[3][fkey] = samples[mask][bi,i]
            tests[4][fkey] = np.average(samples[mask][:,i],weights=self.weights[mask],axis=0)

        # find best fit
        chis = []
        for i,test in enumerate(tests):
            lightcurve = transit(self.time, test)
            quad_model = test['a0'] + test['a1']*self.time + test['a2']*self.time**2
            residuals = self.data - (lightcurve*quad_model)
            chis.append(np.sum(residuals**2))

        mi = np.argmin(chis)
        self.parameters = copy.deepcopy(tests[mi])

        # final model
        self.create_fit_variables()

    def plot_bestfit(self, phase=True):
        f = plt.figure(figsize=(12,7))
        # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        if phase:

            axs[0].errorbar(self.phase, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.5)
            bt, bf = time_bin(self.time, self.detrended)
            # bp = (bt-self.parameters['tmid'])/self.parameters['per']
            # axs[0].plot(bp,bf,'c.',alpha=0.5,zorder=2)
            axs[0].plot(self.phase, self.transit, 'r-', zorder=3)
            axs[0].set_xlabel("Phase")

            # residuals
            axs[1].plot(self.phase, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.5)
            bt, br = time_bin(self.time, self.residuals/np.median(self.data)*1e6)
            # bp = (bt-self.parameters['tmid'])/self.parameters['per']

            # axs[1].plot(bp,br,'c.',alpha=0.5,zorder=2)
            axs[1].set_xlabel("Phase")
        else:

            axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.5)
            bt, bf = time_bin(self.time, self.detrended)
            axs[0].plot(bt,bf,'c.',alpha=0.5,zorder=2)
            axs[0].plot(self.time, self.transit, 'r-', zorder=3)
            axs[0].set_xlabel("Time [day]")

            # residuals
            axs[1].plot(self.time, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.5)
            bt, br = time_bin(self.time, self.residuals/np.median(self.data)*1e6)
            axs[1].plot(bt,br,'c.',alpha=0.5,zorder=2)
            axs[1].set_xlabel("Time [day]")

        axs[1].set_ylabel("Residuals [ppm]")
        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')
        axs[1].grid(True,ls='--')
        plt.tight_layout()

        return f,axs

    def plot_triangle(self):
        fig,axs = dynesty.plotting.cornerplot(self.results, labels=list(self.bounds.keys()), quantiles_2d=[0.4,0.85], smooth=0.015, show_titles=True,use_math_text=True, title_fmt='.2e',hist2d_kwargs={'alpha':1,'zorder':2,'fill_contours':False})
        dynesty.plotting.cornerpoints(self.results, labels=list(self.bounds.keys()), fig=[fig,axs[1:,:-1]],plot_kwargs={'alpha':0.1,'zorder':1,})
        return fig, axs


class lc_fitter_spitzer:
    '''
    K. PEARSON
    Transit fitting with Gaussian Kernel regression for position dependent detrending
    uses occult quad
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self, time, data, dataerr, prior, bounds, syspars, neighbors=50,verbose=False, eclipse=False):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        self.syspars = syspars
        self.verbose = verbose
        self.eclipse = eclipse
        self.gw, self.nearest = gaussian_weights(syspars, neighbors=neighbors)
        self.fit_nested()

    def fit_nested(self):
        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])
        bounddiff = np.diff(boundarray,1).reshape(-1)

        # alloc arrays for C
        time = np.require(self.time,dtype=ctypes.c_double,requirements='C')
        self.lightcurve = np.zeros(len(self.time),dtype=ctypes.c_double)
        self.lightcurve = np.require(self.lightcurve,dtype=ctypes.c_double,requirements='C')

        def loglike(pars):
            # update free parameters
            for i,par in enumerate(pars):
                self.prior[freekeys[i]] = par

            # call C function
            keys = ['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
            vals = [self.prior[k] for k in keys]
            occultquadC(time, *vals, len(time), self.lightcurve)
            self.lightcurve += self.eclipse*(1-np.min(self.lightcurve))
            detrended = self.data/self.lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = self.lightcurve*wf
            return -0.5 * np.sum(((self.data-model)**2/self.dataerr**2))

        def prior_transform(upars):
            # transform unit cube to prior volume
            return (boundarray[:,0] + bounddiff*upars)

        dsampler = dynesty.NestedSampler(loglike, prior_transform, len(freekeys), sample='unif', bound='multi', nlive=1000)

        # dsampler = dynesty.DynamicNestedSampler(
        #    loglike, prior_transform,
        #    ndim=len(freekeys), bound='multi', sample='unif',
        #    maxiter_init=5000, dlogz_init=1, dlogz=0.05,
        #    maxiter_batch=1000, maxbatch=10, nlive_batch=100
        # )

        dsampler.run_nested(maxiter=1e6, maxcall=1e6, print_progress=False)
        self.results = dsampler.results

        # alloc data for best fit + error
        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)

        tests = [copy.deepcopy(self.prior) for i in range(6)]

        # Derive kernel density estimate for best fit
        weights = np.exp(self.results.logwt - self.results.logz[-1])
        samples = self.results['samples']
        logvol = self.results['logvol']
        wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
        logvol_grid = np.linspace(logvol[0], logvol[-1], 1000)  # resample
        wt_grid = wt_kde.pdf(-1*logvol_grid)  # evaluate KDE PDF
        self.weights = np.interp(-1*logvol, -1*logvol_grid, wt_grid)  # interpolate

        # errors + final values
        mean, cov = dynesty.utils.mean_and_cov(self.results.samples, weights)
        mean2, _cov2 = dynesty.utils.mean_and_cov(self.results.samples, self.weights)
        for i,fkey in enumerate(freekeys):
            self.errors[fkey] = cov[i,i]**0.5
            tests[0][fkey] = mean[i]
            tests[1][fkey] = mean2[i]

            counts, bins = np.histogram(samples[:,i], bins=100, weights=weights)
            mi = np.argmax(counts)
            tests[5][freekeys[i]] = bins[mi] + 0.5*np.mean(np.diff(bins))

            # finds median and +- 2sigma, will vary from mode if non-gaussian
            self.quantiles[freekeys[i]] = dynesty.utils.quantile(self.results.samples[:,i], [0.025, 0.5, 0.975], weights=weights)
            tests[2][freekeys[i]] = self.quantiles[freekeys[i]][1]

        # find minimum near weighted mean
        mask = (samples[:,0] < self.parameters[freekeys[0]]+2*self.errors[freekeys[0]]) & (samples[:,0] > self.parameters[freekeys[0]]-2*self.errors[freekeys[0]])
        bi = np.argmin(self.weights[mask])

        for i,fkey in enumerate(freekeys):
            tests[3][fkey] = samples[mask][bi,i]
            tests[4][fkey] = np.average(samples[mask][:,i],weights=self.weights[mask],axis=0)

        # find best fit
        chis = []
        res = []
        for i,test in enumerate(tests):
            lightcurve = transit_occultquad(self.time, test)
            lightcurve += self.eclipse*(1-np.min(lightcurve))
            detrended = self.data / lightcurve
            wf = weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            residuals = self.data - model
            res.append(residuals)
            btime, br = time_bin(self.time, residuals)
            blc = transit_occultquad(btime, tests[i])
            mask = blc < 1
            if mask.shape[0] == 0:
                mask = np.ones(blc.shape,dtype=bool)
            if mask.sum() == 0:
                mask = np.ones(blc.shape,dtype=bool)
            duration = btime[mask].max() - btime[mask].min()
            tmask = ((btime - tests[i]['tmid']) < duration) & ((btime - tests[i]['tmid']) > -1*duration)
            chis.append(np.mean(br[tmask]**2))

        mi = np.argmin(chis)
        self.parameters = copy.deepcopy(tests[mi])
        # plt.scatter(samples[mask,0], samples[mask,1], c=weights[mask]); plt.show()

        # best fit model
        self.transit = transit_occultquad(self.time, self.parameters)
        self.transit += self.eclipse*(1-np.min(self.transit))
        detrended = self.data / self.transit
        self.wf = weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.residuals = self.data - self.model
        self.detrended = self.data/self.wf

    def plot_bestfit(self):
        f = plt.figure(figsize=(12,7))
        # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        bt, bf = time_bin(self.time, self.detrended)

        axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.5)
        axs[0].plot(bt,bf,'c.',alpha=0.5,zorder=2)
        axs[0].plot(self.time, self.transit, 'r-', zorder=3)
        axs[0].set_xlabel("Time [day]")
        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')

        axs[1].plot(self.time, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.5)
        bt, br = time_bin(self.time, self.residuals/np.median(self.data)*1e6)
        axs[1].plot(bt,br,'c.',alpha=0.5,zorder=2)
        axs[1].set_xlabel("Time [day]")
        axs[1].set_ylabel("Residuals [ppm]")
        axs[1].grid(True,ls='--')
        plt.tight_layout()

        return f,axs

def time_bin(time, flux, dt=1./(60*24)):
    bins = int(np.floor((max(time) - min(time))/dt))
    bflux = np.zeros(bins)
    btime = np.zeros(bins)
    for i in range(bins):
        mask = (time >= (min(time)+i*dt)) & (time < (min(time)+(i+1)*dt))
        if mask.sum() > 0:
            bflux[i] = np.nanmean(flux[mask])
            btime[i] = np.nanmean(time[mask])
    zmask = (bflux==0) | (btime==0) | np.isnan(bflux) | np.isnan(btime)
    return btime[~zmask], bflux[~zmask]

def lightcurve_jwst_niriss(nrm, fin, out, selftype, _fltr, hstwhitelight_sv, method='ns'):
    '''
    K. PEARSON: white light curve fit for orbital solution
    '''
    wl = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = [pnet for pnet in nrm['data'].keys()]

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

            myfit = lc_fitter(subt, aper, aper_err, tpars, mybounds, mode=method)

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
                out['data'][p][ec]['weights'] = myfit.weights
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
    planetloop = [pnet for pnet in nrm['data'].keys()]

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

                myfit = lc_fitter(subtt, aper, aper_err, tpars, mybounds, mode=method)

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
    planetloop = [pnet for pnet in nrm['data'].keys()]

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
                inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
                w = priors[p].get('omega',0)

                # mask out data by event type
                pmask = (phase > event-1.5*tdur/priors[p]['period']) & (phase < event+1.5*tdur/priors[p]['period'])

                # extract aperture photometry data
                subt = nrm['data'][p]['TIME'][pmask]
                aper = nrm['data'][p]['PHOT'][pmask]
                aper_err = np.sqrt(aper)

                try:
                    if '36' in fltr:
                        lin,quad = get_ld(priors,'Spit36')
                    elif '45' in fltr:
                        lin,quad = get_ld(priors,'Spit45')
                except ValueError:
                    lin,quad = 0,0

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
                    'tmid':np.median(subt),

                    'inc':priors[p]['inc'],
                    'inc_lowerr': max(priors[p]['inc']-3*abs(priors[p]['inc_lowerr']),inc_lim),
                    'inc_uperr': min(priors[p]['inc']+3*abs(priors[p]['inc_uperr']),90),

                    'ars':smaors,
                    'ars_lowerr':smaors_lo,
                    'ars_uperr':smaors_up,

                    'per':priors[p]['period'],
                    'u1':lin, 'u2': quad,
                    'ecc':priors[p]['ecc'],
                    'omega': priors[p].get('omega',0),
                }

                try:
                    tpars['inc'] = hstwhitelight_sv['data'][p]['mcpost']['mean']['inc']
                except KeyError:
                    tpars['inc'] = priors[p]['inc']

                # gather detrending parameters
                wxa = nrm['data'][p]['WX'][pmask]
                wya = nrm['data'][p]['WY'][pmask]
                npp = nrm['data'][p]['NOISEPIXEL'][pmask]
                syspars = np.array([wxa,wya,npp]).T

                # 10 minute time scale
                nneighbors = int(10./24./60./np.mean(np.diff(subt)))
                print("N neighbors:",nneighbors)
                print("N datapoints:", len(subt))

                # define free parameters
                if selftype == 'transit':
                    mybounds = {
                        'rprs':[0,1.25*tpars['rprs']],
                        'tmid':[min(subt),max(subt)],
                        'ars':[tpars['ars_lowerr'], tpars['ars_uperr']]
                    }
                    myfit = lc_fitter_spitzer(subt, aper, aper_err, tpars, mybounds, syspars, neighbors=nneighbors)
                elif selftype == 'eclipse':
                    mybounds = {
                        'rprs':[0,0.5*tpars['rprs']],
                        'tmid':[min(subt),max(subt)],
                        'ars':[tpars['ars_lowerr'], tpars['ars_uperr']]
                    }
                    myfit = lc_fitter_spitzer(subt, aper, aper_err, tpars, mybounds, syspars, neighbors=nneighbors, eclipse=1)

                terrs = {}
                for k in myfit.bounds.keys():
                    tpars[k] = myfit.parameters[k]
                    terrs[k] = myfit.errors[k]

                out['data'][p].append({})
                out['data'][p][ec]['aper_time'] = subt
                out['data'][p][ec]['aper_flux'] = aper
                out['data'][p][ec]['aper_err'] = aper_err
                out['data'][p][ec]['aper_xcent'] = wxa
                out['data'][p][ec]['aper_ycent'] = wya
                out['data'][p][ec]['aper_npp'] = npp
                del myfit.results['bound']
                out['data'][p][ec]['aper_weights'] = myfit.weights
                out['data'][p][ec]['aper_results'] = myfit.results
                out['data'][p][ec]['aper_quantiles'] = myfit.quantiles
                out['data'][p][ec]['aper_wf'] = myfit.wf
                out['data'][p][ec]['aper_model'] = myfit.model
                out['data'][p][ec]['aper_transit'] = myfit.transit
                out['data'][p][ec]['aper_residuals'] = myfit.residuals
                out['data'][p][ec]['aper_detrended'] = myfit.detrended

                out['data'][p][ec]['aper_pars'] = copy.deepcopy(tpars)
                out['data'][p][ec]['aper_errs'] = copy.deepcopy(terrs)

                # state vectors for classifer
                z, _phase = datcore.time2z(subt, tpars['inc'], tpars['tmid'], tpars['ars'], tpars['per'], tpars['ecc'])
                out['data'][p][ec]['postsep'] = z
                out['data'][p][ec]['allwhite'] = myfit.detrended
                out['data'][p][ec]['postlc'] = myfit.transit
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

            out['data'][p]['ES'].append(wht['data'][p][i]['aper_pars']['rprs'])  # rp/rs
            out['data'][p]['ESerr'].append(wht['data'][p][i]['aper_errs']['rprs'])  # upper bound

            update = True
    return update

def plot_posterior(SV, fltr, title='', savedir=None):
    fancy_labels = {
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
        'c4':r'$c_4$'
    }

    if "Spitzer" in fltr:
        results = SV['aper_results']
        flabels = [fancy_labels[k] for k in SV['aper_errs'].keys()]
    elif "JWST" in fltr:
        results = SV['results']
        flabels = [fancy_labels[k] for k in SV['errs'].keys()]

    # histogram + contours
    fig,axs = dynesty.plotting.cornerplot(
        results,
        labels=flabels,
        quantiles_2d=[0.4,0.85],
        smooth=0.015,
        show_titles=True,
        use_math_text=True,
        title_fmt='.2e',
        hist2d_kwargs={'alpha':1,'zorder':2,'fill_contours':False}
    )

    # colorful dots
    dynesty.plotting.cornerpoints(
        results,
        labels=flabels,
        fig=[fig,axs[1:,:-1]],
        plot_kwargs={'alpha':0.1,'zorder':1}
    )

    plt.tight_layout()

    if savedir:
        plt.savefig(savedir+title+".png")
        plt.close()
    else:
        return fig

def plot_pixelmap(sv, title='', savedir=None):
    '''
    K. PEARSON plot
    '''
    f,ax = plt.subplots(1,figsize=(8.5,7))
    im = ax.scatter(
        sv['aper_xcent'],
        sv['aper_ycent'],
        c=sv['aper_wf']/np.median(sv['aper_wf']),
        marker='.',
        vmin=0.99,
        vmax=1.01,
        alpha=0.25,
        cmap='jet',
    )
    ax.set_xlim([
        np.median(sv['aper_xcent'])-3*np.std(sv['aper_xcent']),
        np.median(sv['aper_xcent'])+3*np.std(sv['aper_xcent'])
    ])
    ax.set_ylim([
        np.median(sv['aper_ycent'])-3*np.std(sv['aper_ycent']),
        np.median(sv['aper_ycent'])+3*np.std(sv['aper_ycent'])
    ])

    ax.set_title(title,fontsize=14)
    ax.set_xlabel('X-Centroid [px]',fontsize=14)
    ax.set_ylabel('Y-Centroid [px]',fontsize=14)
    cbar = f.colorbar(im)
    cbar.set_label('Relative Pixel Response',fontsize=14,rotation=270,labelpad=15)

    plt.tight_layout()
    if savedir:
        plt.savefig(savedir+title+".png")
        plt.close()
    else:
        return f

def jwst_lightcurve(sv, savedir=None, suptitle=''):
    f = plt.figure(figsize=(12,7))
    # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
    ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
    ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
    axs = [ax_lc, ax_res]

    bt, bf = time_bin(sv['time'], sv['detrended'])

    axs[0].errorbar(sv['time'],  sv['detrended'], yerr=np.std(sv['residuals'])/np.median(sv['flux']), ls='none', marker='.', color='black', zorder=1, alpha=0.5)
    axs[0].plot(bt,bf,'c.',alpha=0.5,zorder=2)
    axs[0].plot(sv['time'], sv['transit'], 'r-', zorder=3)
    axs[0].set_xlabel("Time [day]")
    axs[0].set_ylabel("Relative Flux")
    axs[0].grid(True,ls='--')

    axs[1].plot(sv['time'], sv['residuals']/np.median(sv['flux'])*1e6, 'k.', alpha=0.5)
    bt, br = time_bin(sv['time'], sv['residuals']/np.median(sv['flux'])*1e6)
    axs[1].plot(bt,br,'c.',alpha=0.5,zorder=2)
    axs[1].set_xlabel("Time [day]")
    axs[1].set_ylabel("Residuals [ppm]")
    axs[1].grid(True,ls='--')
    plt.tight_layout()

    if savedir:
        plt.savefig(savedir+suptitle+".png")
        plt.close()
    else:
        return f

def spitzer_lightcurve(sv, savedir=None, suptitle=''):
    '''
    K. PEARSON plot of light curve fit
    '''
    f,ax = plt.subplots(3,2,figsize=(12,12))
    f.suptitle(suptitle,y=0.99)
    res = sv['aper_residuals']/np.median(sv['aper_flux'])
    detrend = sv['aper_detrended']

    # #################### RAW FLUX ################
    ax[0,0].errorbar(
        sv['aper_time'], sv['aper_flux']/np.median(sv['aper_flux']),
        yerr=0,
        marker='.', ls='none', color='black',alpha=0.5)
    ax[0,0].set_xlim([min(sv['aper_time']), max(sv['aper_time'])])
    ax[0,0].set_xlabel('Time [JD]')
    ax[0,0].set_ylabel('Raw Relative Flux')
    ax[0,0].set_ylim([
        np.nanmean(detrend)-4*np.nanstd(detrend),
        np.nanmean(detrend)+4*np.nanstd(detrend)])

    # ################# DETRENDED FLUX ##################
    ax[1,0].errorbar(
        sv['aper_time'], sv['aper_detrended'],
        yerr=0,
        marker='.', ls='none', color='black',alpha=0.15,
    )
    ax[1,0].plot(sv['aper_time'], sv['aper_transit'],'r-',zorder=4)
    bta,bfa = time_bin(sv['aper_time'], detrend)
    ax[1,0].plot(bta, bfa, 'co', zorder=3, alpha=0.75)
    ax[1,0].set_xlim([min(sv['aper_time']), max(sv['aper_time'])])
    ax[1,0].set_xlabel('Time [JD]')
    ax[1,0].set_ylabel('Relative Flux')
    ax[1,0].set_ylim([
        np.nanmean(detrend)-4*np.nanstd(detrend),
        np.nanmean(detrend)+4*np.nanstd(detrend)])

    # ################ RESIDUALS ###############
    bta,bfa = time_bin(sv['aper_time'], res)
    bstd = np.nanstd(bfa)*1e6
    std = np.nanstd(res)*1e6
    ax[2,0].plot(
        bta, bfa*1e6, 'co', zorder=3, alpha=0.75,
        label=r'$\sigma$ = {:.0f} ppm'.format(bstd)
    )
    ax[2,0].errorbar(
        sv['aper_time'], res*1e6,
        yerr=0,
        marker='.', ls='none', color='black',alpha=0.15,
        label=r'$\sigma$ = {:.0f} ppm'.format(std)
    )
    ax[2,0].legend(loc='best')
    ax[2,0].set_xlim([min(sv['aper_time']), max(sv['aper_time'])])
    ax[2,0].set_xlabel('Time [JD]')
    ax[2,0].set_ylabel('Residuals [ppm]')
    ax[2,0].set_ylim([
        np.nanmean(res*1e6)-3*np.nanstd(res*1e6),
        np.nanmean(res*1e6)+3*np.nanstd(res*1e6)])

    # ######## # # # # CENTROID X # # # #########
    ax[0,1].plot(
        sv['aper_time'], sv['aper_xcent'],
        marker='.', ls='none', color='black',alpha=0.5,
    )
    ax[0,1].set_xlim([min(sv['aper_time']), max(sv['aper_time'])])
    ax[0,1].set_xlabel('Time [JD]')
    ax[0,1].set_ylabel('X-Centroid [px]')
    ax[0,1].set_ylim([
        np.nanmean(sv['aper_xcent'])-3*np.nanstd(sv['aper_xcent']),
        np.nanmean(sv['aper_xcent'])+3*np.nanstd(sv['aper_xcent'])])

    ax[1,1].plot(
        sv['aper_time'], sv['aper_ycent'],
        marker='.', ls='none', color='black',alpha=0.5,
    )
    ax[1,1].set_xlim([min(sv['aper_time']), max(sv['aper_time'])])
    ax[1,1].set_xlabel('Time [JD]')
    ax[1,1].set_ylabel('Y-Centroid [px]')
    ax[1,1].set_ylim([
        np.nanmean(sv['aper_ycent'])-3*np.nanstd(sv['aper_ycent']),
        np.nanmean(sv['aper_ycent'])+3*np.nanstd(sv['aper_ycent'])])

    ax[2,1].plot(
        sv['aper_time'], sv['aper_npp'],
        marker='.', ls='none', color='black',alpha=0.5,
    )
    ax[2,1].set_xlim([min(sv['aper_time']), max(sv['aper_time'])])
    ax[2,1].set_xlabel('Time [JD]')
    ax[2,1].set_ylabel('Noise Pixel')
    ax[2,1].set_ylim([
        np.nanmean(sv['aper_npp'])-3*np.nanstd(sv['aper_npp']),
        np.nanmean(sv['aper_npp'])+3*np.nanstd(sv['aper_npp'])])

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    if savedir:
        plt.savefig(savedir+suptitle+".png")
        plt.close()
    else:
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
        if name == 'data' or name == 'STATUS':
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
                for i in range(0,len(wav)):
                    allfltrs.append(fltrs[ids])
                    checkwav.append(wav[i])
                # order everything using allwav before saving them
                out[-1]['data'][planet] = {'WB': np.sort(np.array(allwav)), 'WBlow': [x for _,x in sorted(zip(allwav,allwav_lw))], 'WBup': [x for _,x in sorted(zip(allwav,allwav_up))], 'ES': [x for _,x in sorted(zip(allwav,allspec))], 'ESerr': [x for _,x in sorted(zip(allwav,allspec_err))], 'Fltrs': [x for _,x in sorted(zip(allwav,allfltrs))], 'Hs': out[ids]['data'][planet]['Hs']}
            exospec = True  # return if all inputs were empty
            out[-1]['STATUS'].append(True)
    return exospec
