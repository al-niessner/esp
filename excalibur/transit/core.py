'''transit core ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

# pylint: disable=import-self
import excalibur.data.core as datcore
import excalibur.system.core as syscore
import excalibur.util.cerberus as crbutil
from excalibur.util import elca
import excalibur.transit.core

import io
import copy
import logging
import random
import lmfit as lm
import matplotlib.pyplot as plt

# pylint: disable=import-error
from ultranest import ReactiveNestedSampler

import pymc3 as pm
log = logging.getLogger(__name__)
pymc3log = logging.getLogger('pymc3')
pymc3log.setLevel(logging.ERROR)

from scipy.optimize import least_squares, brentq
import scipy.constants as cst
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde, skew, kurtosis
# from scipy.fft import fft, fftfreq
from numpy.fft import fft, fftfreq

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
        self.gw, self.nearest = elca.gaussian_weights(self.syspars, neighbors=self.neighbors)

        def lc2min(pars):
            # pylint: disable=invalid-unary-operand-type
            for i, _ in enumerate(pars):
                self.prior[freekeys[i]] = pars[i]
            lightcurve = elca.phasecurve(self.time, self.prior)
            detrended = self.data/lightcurve
            wf = elca.weightedflux(detrended, self.gw, self.nearest)
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
        self.gw, self.nearest = elca.gaussian_weights(self.syspars, neighbors=self.neighbors)

        def lc2min_transit(pars):
            '''lc2min_transit ds'''
            # pylint: disable=invalid-unary-operand-type
            for i, _ in enumerate(pars):
                self.prior[freekeys[i]] = pars[i]
            lightcurve = elca.transit(self.time, self.prior)
            detrended = self.data/lightcurve
            wf = elca.weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            return -np.sum(((self.data-model)/self.dataerr)**2)

        def lc2min_phasecurve(pars):
            '''lc2min_phasecurve ds'''
            # pylint: disable=invalid-unary-operand-type
            for i, _ in enumerate(pars):
                self.prior[freekeys[i]] = pars[i]
            lightcurve = elca.phasecurve(self.time, self.prior)
            detrended = self.data/lightcurve
            wf = elca.weightedflux(detrended, self.gw, self.nearest)
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
                self.results = ReactiveNestedSampler(freekeys, lc2min_phasecurve, prior_transform_phasecurve).run(max_ncalls=2e5)
            else:
                self.results = ReactiveNestedSampler(freekeys, lc2min_transit, prior_transform_basic).run(max_ncalls=2e5)
        else:
            if 'fpfs' in freekeys:
                self.results = ReactiveNestedSampler(freekeys, lc2min_phasecurve, prior_transform_phasecurve).run(max_ncalls=2e5, show_status=self.verbose, viz_callback=self.verbose)
            else:
                self.results = ReactiveNestedSampler(freekeys, lc2min_transit, prior_transform_basic).run(max_ncalls=2e5, show_status=self.verbose, viz_callback=self.verbose)

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
        self.transit = elca.phasecurve(self.time, self.parameters)
        detrended = self.data / self.transit
        self.wf = elca.weightedflux(detrended, self.gw, self.nearest)
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

        bt, bf, _ = elca.time_bin(self.time, self.detrended, bin_dt)
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

        bt, br, _ = elca.time_bin(self.time, self.residuals/np.median(self.data)*1e6, bin_dt)
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
        fig = elca.corner(self.results['weighted_samples']['points'],
                            labels=labels,
                            bins=int(np.sqrt(self.results['samples'].shape[0])),
                            plot_range=ranges,
                            plot_contours=True,
                            levels=[chi2[mask1].max(), chi2[mask2].max(), chi2[mask3].max()],
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
        bt, bf, _ = elca.time_bin(self.time, self.detrended, bin_dt)
        bp = (bt-self.parameters['tmid'])/self.parameters['per']
        bt, br, _ = elca.time_bin(self.time, self.residuals, bin_dt)

        bcurve = elca.brightness(bt,self.parameters)
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

        # estimate transit duration
        ssc = syscore.ssconstants()
        smaors = fin['priors'][p]['sma']/fin['priors']['R*']/ssc['Rsun/AU']
        tdur = fin['priors'][p]['period']/(np.pi)/smaors
        pdur = 1.75*tdur/fin['priors'][p]['period']

        # loop over orbital epochs
        for i in np.unique(tme['data'][p][selftype]):

            # mask out orbit
            omask = (phase > (i-pdur)) & (phase < (i+pdur))
            dt = np.nanmean(np.diff(out['data'][p]['TIME'][omask]))*24*60
            if np.isnan(dt):
                continue
            if dt == 0:
                continue

            ndt = int(7/dt)*2+1
            ndt = max(51, ndt)

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

                    # non-linear limb darkening
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

                # state vectors for classifer
                z, _phase = datcore.time2z(subt, tpars['inc'], tpars['tmid'], tpars['ars'], tpars['per'], tpars['ecc'])
                out['data'][p][ec]['postsep'] = z
                out['data'][p][ec]['allwhite'] = myfit.detrended
                out['data'][p][ec]['postlc'] = myfit.transit

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
                # smaors_up = (priors[p]['sma']+priors[p]['sma_uperr'])/(priors['R*']-abs(priors['R*_lowerr']))/ssc['Rsun/AU']
                # smaors_lo = (priors[p]['sma']-abs(priors[p]['sma_lowerr']))/(priors['R*']+priors['R*_uperr'])/ssc['Rsun/AU']

                # to do: update duration for eccentric orbits
                # https://arxiv.org/pdf/1001.2010.pdf eq 16
                tdur = priors[p]['period']/(np.pi)/smaors
                rprs = (priors[p]['rp']*7.1492e7) / (priors['R*']*6.955e8)
                # inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
                w = priors[p].get('omega',0)
                tmid = priors[p]['period']*event + priors[p]['t0']

                # mask out data by event type
                if selftype == 'transit':
                    pmask = (phase > event-1.5*tdur/priors[p]['period']) & (phase < event+1.5*tdur/priors[p]['period'])
                elif selftype == 'eclipse':
                    pmask = (phase > event-2.5*tdur/priors[p]['period']) & (phase < event+2.5*tdur/priors[p]['period'])

                # extract aperture photometry data
                subt = nrm['data'][p]['TIME'][pmask]
                aper = nrm['data'][p]['PHOT'][pmask]
                aper_err = np.sqrt(aper)

                priors[p]['ars'] = smaors
                # fpfs = eclipse_ratio(priors, p, fltr)

                # search priors for limb darkening
                if '36' in fltr:
                    pkey = 'Spitzer_IRAC1_subarray'
                elif '45' in fltr:
                    pkey = 'Spitzer_IRAC2_subarray'

                if pkey in priors[p].keys():
                    u0 = priors[p][pkey][0]
                    u1 = priors[p][pkey][1]
                    u2 = priors[p][pkey][2]
                    u3 = priors[p][pkey][3]
                else:
                    u0 = 0
                    u1 = 0
                    u2 = 0
                    u3 = 0

                tpars = {
                    # Star
                    'T*':priors['T*'],

                    # transit
                    'rprs': rprs,
                    'ars': smaors,
                    'tmid':tmid,
                    'per': priors[p]['period'],
                    'inc': priors[p]['inc'],
                    'omega': priors['b'].get('omega',0),
                    'ecc': priors['b']['ecc'],

                    # limb darkening (nonlinear - exotethys - pylightcurve)
                    # precompute and add to priors in target.edit
                    'u0':u0,
                    'u1':u1,
                    'u2':u2,
                    'u3':u3,

                    # phase curve amplitudes
                    'c0':0, 'c1':0, 'c2':0, 'c3':0, 'c4':0,
                    # 'fpfs': fpfs,
                }

                try:
                    tpars['inc'] = hstwhitelight_sv['data'][p]['mcpost']['mean']['inc']
                except KeyError:
                    tpars['inc'] = priors[p]['inc']

                # remove first 30 min of data after any big gaps, ramp
                tmask = np.ones(subt.shape).astype(bool)

                smask = np.argsort(subt)
                dts = np.diff(subt[smask])
                dmask = dts > (2./(24*60))

                if np.isnan(dts.mean()):
                    continue
                if dts.mean() == 0:
                    continue

                ndt = int(15./(24*60*dts.mean()))*2+1
                ndt = max(25, ndt)

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

                # remove zeros and nans
                zmask = (aper != 0) & (aper_err != 0) & (~np.isnan(aper)) & (~np.isnan(aper_err))
                if zmask.sum() < 10:
                    continue

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
                nneighbors = min(200, nneighbors)
                print("N neighbors:",nneighbors)
                print("N datapoints:", len(subt))

                # define free parameters
                if selftype == 'transit':
                    mybounds = {
                        'rprs':[0,1.5*tpars['rprs']],
                        'tmid':[tmid-0.01,tmid+0.01],
                        # 'ars':[smaors_lo,smaors_up]
                        'inc':[tpars['inc']-3, min(90, tpars['inc']+3)]
                    }
                    myfit = elca.lc_fitter(subt, aper, aper_err, syspars, tpars, mybounds, neighbors=nneighbors, max_ncalls=2e5, verbose=False)
                elif selftype == 'eclipse':

                    # perform another round of sigma clipping
                    photmask = ~np.isnan(sigma_clip(aper, nneighbors))
                    wxmask = ~np.isnan(sigma_clip(wxa, nneighbors))
                    wymask = ~np.isnan(sigma_clip(wya, nneighbors))
                    nppmask = ~np.isnan(sigma_clip(npp, nneighbors))
                    totalmask = photmask & wxmask & wymask & nppmask

                    aper = aper[totalmask]
                    aper_err = aper_err[totalmask]
                    subt = subt[totalmask]
                    syspars = syspars[totalmask]

                    mybounds = {
                        'rprs':[0,0.5*tpars['rprs']],
                        'tmid':[tme-0.01,tme+0.01],
                        'inc':[tpars['inc']-3, min(90, tpars['inc']+3)]
                        # 'ars':[smaors_lo,smaors_up],
                    }

                    # zero out limb darkening
                    tpars['u0'] = 0
                    tpars['u1'] = 0
                    tpars['u2'] = 0
                    tpars['u3'] = 0

                    # modify to get proper duration
                    tpars['omega'] = priors['b'].get('omega',0)+180

                    # fit the eclipse
                    myfit = elca.lc_fitter(subt, aper, aper_err, syspars, tpars, mybounds, neighbors=nneighbors, max_ncalls=2e5, verbose=False)

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

                # estimates for photon noise
                photons = aper*1.0  # already converted to e in data task

                # noise estimate in transit
                tmask = myfit.transit < 1
                photon_noise_timeseries = 1/np.sqrt(photons.mean())

                # photon noise factor based on timeseries
                res_std = np.round(np.std(myfit.residuals/np.median(aper)),7)
                nf_timeseries = res_std / photon_noise_timeseries
                raw_residual = aper/np.median(aper)-myfit.transit
                nf_timeseries_raw = np.std(raw_residual) / photon_noise_timeseries

                raw_residual = aper/np.median(aper) - myfit.transit
                rel_residuals = myfit.residuals / np.median(aper)

                # convert transit duration to seconds
                tdur = myfit.duration_measured*24*60*60
                tdur_freq = 1/tdur

                # create plot for residual statistics
                fig, ax = plt.subplots(3, figsize=(10,10))
                binspace = np.linspace(-0.02,0.02,201)
                raw_label = f"Mean: {np.mean(raw_residual,):.4f} \n"\
                            f"Stdev: {np.std(raw_residual):.4f} \n"\
                            f"Skew: {skew(raw_residual):.4f} \n"\
                            f"Kurtosis: {kurtosis(raw_residual):.4f}\n"\
                            f"Photon Noise: {nf_timeseries_raw:.2f}"
                ax[0].hist(raw_residual, bins=binspace,label=raw_label,color=plt.cm.jet(0.25),alpha=0.5)
                detrend_label = f"Mean: {np.mean(rel_residuals):.4f} \n"\
                            f"Stdev: {np.std(rel_residuals):.4f} \n"\
                            f"Skew: {skew(rel_residuals):.4f} \n"\
                            f"Kurtosis: {kurtosis(rel_residuals):.4f}\n"\
                            f"Photon Noise: {nf_timeseries:.2f}"
                ax[0].hist(rel_residuals, bins=binspace, label=detrend_label, color=plt.cm.jet(0.75),alpha=0.5)
                ax[0].set_xlabel('Relative Flux Residuals')
                ax[0].legend(loc='best')
                ax[1].scatter(subt, raw_residual, marker='.', label=f"Raw ({np.std(raw_residual,0)*100:.2f} %)",color=plt.cm.jet(0.25),alpha=0.25)
                ax[1].scatter(subt, rel_residuals, marker='.', label=f"Detrended ({np.std(rel_residuals,0)*100:.2f} %)",color=plt.cm.jet(0.75),alpha=0.25)
                ax[1].legend(loc='best')
                ax[1].set_xlabel('Time [BJD]')
                ax[0].set_title(f'Residual Statistics: {p} {selftype} {fltr}')
                ax[1].set_ylabel("Relative Flux")

                # compute fourier transform of raw_residual
                N = len(raw_residual)
                fft_raw = fft(raw_residual)
                fft_res = fft(rel_residuals)
                xf = fftfreq(len(raw_residual), d=np.diff(subt).mean()*24*60*60)[:N//2]
                # fftraw = 2.0/N * np.abs(fft_raw[0:N//2])
                # future: square + integrate under the curve and normalize such that it equals time series variance
                ax[2].loglog(xf, 2.0/N * np.abs(fft_raw[0:N//2]),alpha=0.5,label='Raw',color=plt.cm.jet(0.25))
                ax[2].loglog(xf, 2.0/N * np.abs(fft_res[0:N//2]),alpha=0.5,label='Detrended',color=plt.cm.jet(0.75))
                ax[2].axvline(tdur_freq,ls='--',color='black',alpha=0.5)
                ax[2].set_ylabel('Power')
                ax[2].set_xlabel('Frequency [Hz]')
                ax[2].legend()
                ax[2].grid(True,ls='--')
                plt.tight_layout()

                # save plot to state vector
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                out['data'][p][ec]['plot_residual_fft'] = buf.getvalue()

                # state vectors for classifer
                z, _phase = datcore.time2z(subt, tpars['inc'], tpars['tmid'], tpars['ars'], tpars['per'], tpars['ecc'])
                out['data'][p][ec]['postsep'] = z
                out['data'][p][ec]['allwhite'] = myfit.detrended
                out['data'][p][ec]['postlc'] = myfit.transit

                ec += 1
                out['STATUS'].append(True)
                wl = True
            except NameError as e:
                print("Error:",e)
                out['data'][p].append({})
                out['data'][p][ec].append({'error':e})

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

    bt, bf, _ = elca.time_bin(sv['time'], sv['detrended'])

    axs[0].errorbar(sv['time'],  sv['detrended'], yerr=np.std(sv['residuals'])/np.median(sv['flux']), ls='none', marker='.', color='black', zorder=1, alpha=0.5)
    axs[0].plot(bt,bf,'c.',alpha=0.5,zorder=2)
    axs[0].plot(sv['time'], sv['transit'], 'r-', zorder=3)
    axs[0].set_xlabel("Time [day]")
    axs[0].set_ylabel("Relative Flux")
    axs[0].grid(True,ls='--')

    axs[1].plot(sv['time'], sv['residuals']/np.median(sv['flux'])*1e6, 'k.', alpha=0.5)
    bt, br, _ = elca.time_bin(sv['time'], sv['residuals']/np.median(sv['flux'])*1e6)
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
