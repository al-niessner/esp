# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.data.core as datcore
import excalibur.system.core as syscore
import excalibur.cerberus.core as crbcore
# pylint: disable=import-self
import excalibur.transit.core

import re
import copy
import requests
import logging
import numpy as np
import lmfit as lm

import pymc3 as pm
log = logging.getLogger(__name__)
pymc3log = logging.getLogger('pymc3')
pymc3log.setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import scipy.constants as cst
from scipy import spatial
from scipy.ndimage import median_filter

import theano.tensor as tt
import theano.compile.ops as tco

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
    return dawgie.VERSION(1,1,3)
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
                        params.add('ologtau', value=np.mean([minoscale, maxoscale]),
                                   min=minoscale, max=maxoscale)
                        params.add('ologdelay', value=np.mean([minoscale, maxdelay]),
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
                noisescalethr = 9e0
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
    '''
    return dawgie.VERSION(1,2,0)

def hstwhitelight(allnrm, fin, out, allext, selftype, chainlen=int(1e4), verbose=False):
    '''
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

def whitelight(nrm, fin, out, ext, selftype, multiwl, chainlen=int(1e4), verbose=False):
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
        # PYMC3 --------------------------------------------------------------------------
        with pm.Model():
            rprs = pm.TruncatedNormal('rprs', mu=rpors, tau=taurprs,
                                      lower=rpors/2e0, upper=2e0*rpors)
            nodes.append(rprs)
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
        out['STATUS'].append(True)
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
    '''
    return dawgie.VERSION(1,1,9)

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
        if 'STIS' in ext:
            wave, _trash = binnagem(wave, 100)
            wave = np.resize(wave,(1,100))
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
            # PRIORS -----------------------------------------------------------------
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
            # propphn = np.nanmedian(dnoise)*(1e0 - whiterprs**2)*np.sqrt(1e0/nit + 1e0/noot)
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
                else: rprs = pm.Normal('rprs', mu=prcenter, tau=1e0/(prwidth**2))
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
                clspvl = np.nanmedian(trace['rprs'])
                # Spectrum outlier rejection + inpaint with np.nan
                if abs(clspvl - whiterprs) > 5e0*Hs: clspvl = np.nan
                out['data'][p]['ES'].append(clspvl)
                out['data'][p]['ESerr'].append(np.nanstd(trace['rprs']))
                out['data'][p]['MCPOST'].append(mcpost)
                out['data'][p]['WBlow'].append(wl)
                out['data'][p]['WBup'].append(wh)
                out['data'][p]['WB'].append(np.mean([wl, wh]))
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
    if 'STIS' in ext:
        wave, _trash = binnagem(wave, 100)
        wave = np.resize(wave,(1,100))
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
def norm_spitzer(cal, tme, fin, out, selftype, debug=False):
    '''
    K. PEARSON: prep data for light curve fitting
        remove nans, remove zeros, 3 sigma clip time series
    '''
    normed = False
    priors = fin['priors'].copy()

    planetloop = [pnet for pnet in tme['data'].keys() if (pnet in priors.keys()) and tme['data'][pnet][selftype]]

    for p in planetloop:
        out['data'][p] = {}

        # determine optimal aperture size
        phot = np.array(cal['data']['PHOT']).reshape(-1,5)
        sphot = np.copy(phot)
        for j in range(phot.shape[1]):
            dt = int(2.5/(24*60*np.percentile(np.diff(cal['data']['TIME']),25)))
            dt = 2*dt + 1  # force it to be odd
            sphot[:,j] = sigma_clip(sphot[:,j], dt)
            sphot[:,j] = sigma_clip(sphot[:,j], dt)
        std = np.nanstd(sphot,0)
        bi = np.argmin(std)

        # reformat some data
        flux = np.array(cal['data']['PHOT']).reshape(-1,5)[:,bi]
        noisep = np.array(cal['data']['NOISEPIXEL']).reshape(-1,5)[:,bi]
        pflux = np.array(cal['data']['G_PSF'])
        visits = tme['data'][p]['visits']  # really more like planetary orbits

        # remove nans and zeros
        mask = np.isnan(flux) | np.isnan(pflux) | (flux == 0)
        keys = [
            'TIME','WX','WY',
            'G_PSF_ERR', 'G_PSF', 'G_XCENT', 'G_YCENT',
            'G_SIGMAX', 'G_SIGMAY', 'G_ROT', 'G_MODEL',
        ]
        for k in keys:
            out['data'][p][k] = np.array(cal['data'][k])[~mask]
        out['data'][p]['visits'] = visits[~mask]
        out['data'][p]['NOISEPIXEL'] = noisep[~mask]
        out['data'][p]['PHOT'] = flux[~mask]

        # time order things
        ordt = np.argsort(out['data'][p]['TIME'])
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][ordt]

        # 3 sigma clip flux time series
        badmask = np.zeros(out['data'][p]['TIME'].shape).astype(bool)
        for i in np.unique(out['data'][p]['visits']):
            omask = out['data'][p]['visits'] == i

            dt = np.nanmean(np.diff(out['data'][p]['TIME'][omask]))*24*60
            medf = median_filter(out['data'][p]['PHOT'][omask], int(15/dt)*2+1)
            res = out['data'][p]['PHOT'][omask] - medf
            photmask = np.abs(res) > 3*np.std(res)

            # medf = median_filter(out['data'][p]['G_PSF'][omask], int(15/dt)*2+1 )
            # res = out['data'][p]['G_PSF'][omask] - medf
            # psfmask = np.abs(res) > 3*np.std(res)

            badmask[omask] = photmask  # | psfmask

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

            plt.plot(out['data'][p]['TIME'], out['data'][p]['G_PSF'],'g.')
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

def time_bin(time, flux, dt=1./(60*24)):
    '''
    K. PEARSON: bin data in time
    '''
    bins = int(np.floor((max(time) - min(time))/dt))
    bflux = np.zeros(bins)
    btime = np.zeros(bins)
    for i in range(bins):
        mask = (time >= (min(time)+i*dt)) & (time < (min(time)+(i+1)*dt))
        bflux[i] = np.mean(flux[mask])
        btime[i] = np.mean(time[mask])
    return btime, bflux

def weightedflux(flux,gw,nearest):
    '''
    K. PEARSON: weighted sum
    '''
    return np.sum(flux[nearest]*gw,axis=-1)

def gaussian_weights(X, w=None, neighbors=100, feature_scale=1000):
    '''
        K. Pearson: Gaussian weights of nearest neighbors
    '''
    if isinstance(w, type(None)): w = np.ones(X.shape[1])
    Xm = (X - np.median(X,0))/w
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
    return gw, nearest.astype(int)

def sigma_clip(data,dt):
    '''
    K. PEARSON: 3 sigma clip outliers from running median filter
    '''
    mdata = median_filter(data, dt)
    res = data - mdata
    mask = np.abs(res) > 3*np.nanstd(res)
    data[mask] = np.nan
    return data

def transit(time, values):
    # wrapper for gael's lc code
    z,phase = datcore.time2z(time, values['inc'], values['tm'], values['ar'], values['per'], values['ecc'])
    phase += 0  # this is to shut pylint up
    return tldlc(abs(z), values['rp'], g1=0, g2=values['u1'], g3=0, g4=values['u2'])

def lightcurve_spitzer(nrm, fin, out, selftype, fltr, hstwhitelight_sv):
    '''
    K. PEARSON: modeling of transits and eclipses from Spitzer
    '''
    wl= False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = [pnet for pnet in nrm['data'].keys()]

    for p in planetloop:

        time = nrm['data'][p]['TIME']
        visits = nrm['data'][p]['visits']
        out['data'][p] = []

        # loop through epochs
        ec = 0  # event counter
        for event in nrm['data'][p][selftype]:
            print('processing event:',event)
            emask = visits == event
            out['data'][p].append({})

            # get data
            time = nrm['data'][p]['TIME'][emask]
            tmask = time < 2400000.5
            time[tmask] += 2400000.5

            # compute phase + priors
            smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
            z, phase = datcore.time2z(time, priors[p]['inc'], priors[p]['t0'], smaors, priors[p]['period'], priors[p]['ecc'])
            z += 0  # fuck you pylint
            # to do: update duration for eccentric orbits
            # https://arxiv.org/pdf/1001.2010.pdf eq 16
            tdur = priors[p]['period']/(2*np.pi)/smaors
            rprs = (priors[p]['rp']*7.1492e7) / (priors['R*']*6.955e8)
            # inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
            w = priors[p].get('omega',0)

            # mask out data by event type
            if selftype == 'transit':
                pmask = (phase > -2*tdur/priors[p]['period']) & (phase < 2*tdur/priors[p]['period'])
            elif selftype == 'eclipse':
                # https://arxiv.org/pdf/1001.2010.pdf eq 33
                t0e = priors[p]['t0']+ priors[p]['period']*0.5 * (1 + priors[p]['ecc']*(4./np.pi)*np.cos(np.deg2rad(w)))
                z, phase = datcore.time2z(time, priors[p]['inc'], t0e, smaors, priors[p]['period'], priors[p]['ecc'])
                pmask = (phase > -2*tdur/priors[p]['period']) & (phase < 2*tdur/priors[p]['period'])
            elif selftype == 'phasecurve':
                print('implement phasecurve mask')

            # extract aperture photometry data
            subt = time[pmask]
            aper = nrm['data'][p]['PHOT'][emask][pmask]
            aper_err = np.sqrt(aper)
            # gpsf = nrm['data'][p]['G_PSF'][emask][pmask]
            # gpsf_err = np.sqrt(gpsf)

            if '36' in fltr:
                lin,quad = get_ld(priors,'Spit36')
            elif '45' in fltr:
                lin,quad = get_ld(priors,'Spit45')

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
            # cq,eq = ps.coeffs_qd(do_mc=True

            tpars = {
                'rp': rprs,
                'tm':np.median(subt),

                'ar':smaors,
                'per':priors[p]['period'],
                'u1':lin, 'u2': quad,
                'ecc':priors[p]['ecc'],
                'ome': priors[p].get('omega',0),
                'a0':1,
            }

            try:
                tpars['inc'] = hstwhitelight_sv['data'][p]['mcpost']['mean']['inc']
            except KeyError:
                tpars['inc'] = priors[p]['inc']

            # perform a quick sigma clip
            dt = np.nanmean(np.diff(subt))*24*60  # minutes
            medf = median_filter(aper, int(15/dt)*2+1)  # needs to be odd
            res = aper - medf
            photmask = np.abs(res) < 3*np.std(res)

            # resize aperture data
            ta = subt[photmask]
            aper = aper[photmask]
            aper_err = aper_err[photmask]
            wxa = nrm['data'][p]['WX'][emask][pmask][photmask]
            wya = nrm['data'][p]['WY'][emask][pmask][photmask]
            npp = nrm['data'][p]['NOISEPIXEL'][emask][pmask][photmask]

            def fit_data(time,flux,fluxerr, syspars, tpars, tdur):
                rprs=0

                gw, nearest = gaussian_weights(
                    np.array(syspars).T,
                    # w=np.array([1,1])
                )

                gw[np.isnan(gw)] = 0.01

                @tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar],otypes=[tt.dvector])
                def transit2min(*pars):
                    rprs, tmid, nor = pars
                    tpars['rp'] = float(rprs)
                    tpars['tm'] = float(tmid)
                    # tpars['inc']= float(inc)

                    lcmode = transit(time=time, values=tpars)
                    detrended = flux/lcmode
                    # gw, nearest = gaussian_weights( np.array([wx,wy]).T, w=np.array([w1,w2]) )
                    wf = weightedflux(detrended, gw, nearest)
                    return lcmode*wf*float(nor)

                @tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar],otypes=[tt.dvector])
                def eclipse2min(*pars):
                    tmid, fpfs, nor = pars
                    # tpars['rp'] = float(rprs)
                    tpars['tm'] = float(tmid)
                    fpfs = float(fpfs)

                    lcmode = transit(time=time, values=tpars)
                    f1 = lcmode - 1
                    model = fpfs*(2*f1 + rprs**2)+1
                    detrended = flux/model
                    # gw, nearest = gaussian_weights( np.array([wx,wy]).T, w=np.array([w1,w2]) )
                    wf = weightedflux(detrended, gw, nearest)
                    return model*wf*float(nor)

                fcn2min = {
                    'transit':transit2min,
                    'eclipse':eclipse2min,
                }

                with pm.Model():
                    if selftype == "transit":
                        priors = [
                            pm.Uniform('rprs', lower=0.5*rprs,  upper=1.5*rprs),
                            pm.Uniform('tmid', lower=max(np.min(time), np.median(time)-tdur*0.5), upper=min(np.max(time), np.median(time)+tdur*0.5)),
                            # pm.Uniform('inc', lower=inc_lim, upper=90),
                            pm.Uniform('norm', lower=0.9,  upper=1.1)
                            # pm.Uniform('w1',   lower=0, upper=1), # weights for instrument model
                            # pm.Uniform('w2',   lower=0, upper=1)
                        ]
                        pass
                    elif selftype == "eclipse":
                        priors = [
                            # pm.Uniform('rprs', lower=0.99*rprs,  upper=1.01*rprs),
                            pm.Uniform('tmid', lower=max(np.min(time), np.median(time)-tdur*0.5), upper=min(np.max(time), np.median(time)+tdur*0.5)),
                            pm.Uniform('fpfs', lower=0, upper=0.3),
                            pm.Uniform('norm', lower=0.9,  upper=1.1)
                            # pm.Uniform('w1',   lower=0, upper=1), # weights for instrument model
                            # pm.Uniform('w2',   lower=0, upper=1)
                        ]

                    pm.Normal('likelihood',
                              mu=fcn2min[selftype](*priors),
                              tau=(1./fluxerr)**2,
                              observed=flux)

                    trace = pm.sample(5000,
                                      pm.Metropolis(),
                                      chains=4,
                                      tune=500,
                                      progressbar=False)
                return trace

            pymc3log.propagate = False
            trace_aper2d = fit_data(ta,aper,aper_err, [wxa,wya], tpars, tdur)
            trace_aper3d= fit_data(ta,aper,aper_err, [wxa,wya, npp], tpars, tdur)

            tpars['tm'] = np.median(trace_aper2d['tmid'])
            tpars['a0'] = np.median(trace_aper2d['norm'])
            if selftype == 'transit':
                tpars['rp'] = np.median(trace_aper2d['rprs'])
                # tpars['inc']= np.median(trace_aper2d['inc'])
            lcmodel1 = transit(time=ta, values=tpars)
            if selftype == 'eclipse':
                fpfs = np.median(trace_aper2d['fpfs'])
                f1 = lcmodel1 - 1
                lcmodel1 = fpfs*(2*f1 + tpars['rp']**2)+1

            detrended = aper/lcmodel1
            gw, nearest = gaussian_weights(np.array([wxa,wya]).T)
            wf = weightedflux(detrended, gw, nearest)

            out['data'][p][ec]['aper_time'] = ta
            out['data'][p][ec]['aper_flux'] = aper
            out['data'][p][ec]['aper_err'] = aper_err
            out['data'][p][ec]['aper_xcent'] = wxa
            out['data'][p][ec]['aper_ycent'] = wya
            out['data'][p][ec]['aper_trace'] = pm.trace_to_dataframe(trace_aper2d)
            out['data'][p][ec]['aper_wf'] = wf
            out['data'][p][ec]['aper_model'] = lcmodel1
            out['data'][p][ec]['aper_pars'] = copy.deepcopy(tpars)
            out['data'][p][ec]['noise_pixel'] = npp

            tpars['tm'] = np.median(trace_aper3d['tmid'])
            tpars['a0'] = np.median(trace_aper3d['norm'])
            if selftype == 'transit':
                tpars['rp'] = np.median(trace_aper3d['rprs'])
                # tpars['inc']= np.median(trace_aper3d['inc'])
            lcmodel1 = transit(time=ta, values=tpars)
            if selftype == 'eclipse':
                fpfs = np.median(trace_aper3d['fpfs'])
                f1 = lcmodel1 - 1
                lcmodel1 = fpfs*(2*f1 + tpars['rp']**2)+1
            detrended = aper/lcmodel1
            gw, nearest = gaussian_weights(np.array([wxa,wya, npp]).T)
            wf = weightedflux(detrended, gw, nearest)
            out['data'][p][ec]['aper_wf_3d'] = wf
            out['data'][p][ec]['aper_model_3d'] = lcmodel1
            out['data'][p][ec]['aper_pars_3d'] = copy.deepcopy(tpars)
            out['data'][p][ec]['aper_trace_3d'] = pm.trace_to_dataframe(trace_aper3d)

            ec += 1
            out['STATUS'].append(True)
            wl = True

            pass
        pass
    return wl
