# -- IMPORTS -- ------------------------------------------------------
import os
import glob
import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context

import excalibur
import excalibur.system.core as syscore

import numpy as np
import numpy.polynomial.polynomial as poly
import lmfit as lm
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
# from astropy.wcs import WCS
import time as raissatime
from ldtk import LDPSetCreator, BoxcarFilter
import datetime

import scipy.interpolate as itp
import scipy.signal
import scipy.optimize as opt
# from scipy.misc import imresize
from scipy.optimize import least_squares
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation
from PIL import Image as pilimage


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
# -- COLLECT DATA -- -------------------------------------------------
def collect(name, scrape, out):
    '''
    G. ROUDIER: Filters data from target.scrape.databases according to active filters
    '''
    collected = False
    obs, ins, det, fil, mod = name.split('-')
    for rootname in scrape['name'].keys():
        ok = scrape['name'][rootname]['observatory'] in [obs.strip()]
        ok = ok and (scrape['name'][rootname]['instrument'] in [ins.strip()])
        ok = ok and (scrape['name'][rootname]['detector'] in [det.strip()])
        ok = ok and (scrape['name'][rootname]['filter'] in [fil.strip()])
        ok = ok and (scrape['name'][rootname]['mode'] in [mod.strip()])
        if ok:
            out['activefilters'][name]['ROOTNAME'].append(rootname)
            loc = scrape['name'][rootname]['md5']+'_'+scrape['name'][rootname]['sha']
            out['activefilters'][name]['LOC'].append(loc)
            out['activefilters'][name]['TOTAL'].append(True)
            collected = True
            pass
        pass
    if collected:
        log.warning('>-- %s', str(int(np.sum(out['activefilters'][name]['TOTAL']))))
        out['STATUS'].append(True)
        pass
    else:
        log.warning('>-- NO DATA')
        out['activefilters'].pop(name, None)
        pass
    return collected
# ------------------ -------------------------------------------------
# -- TIMING -- -------------------------------------------------------
def timingversion():
    return dawgie.VERSION(1,1,2)

def timing(force, ext, clc, out, verbose=False):
    '''
    G. ROUDIER: Uses system orbital parameters to guide the dataset towards transit, eclipse or phasecurve tasks
    K. Pearson: Spitzer
    '''
    chunked = False
    priors = force['priors'].copy()
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')
    data = {'LOC':[], 'SCANANGLE':[], 'TIME':[], 'EXPLEN':[]}
    # LOAD DATA ------------------------------------------------------
    if 'Spitzer' in ext:
        for loc in sorted(clc['LOC']):
            fullloc = os.path.join(dbs, loc)
            with pyfits.open(fullloc) as hdulist:
                header0 = hdulist[0].header
                ftime = []
                exptime = []
                for fits in hdulist:
                    start = fits.header.get('MJD_OBS')
                    if fits.data.ndim == 3:  # data cube
                        idur = fits.header.get('ATIMEEND') - fits.header.get('AINTBEG')
                        nimgs = fits.data.shape[0]
                        dt = idur/nimgs/(24*60*60)
                        for i in range(nimgs):
                            ftime.append(start+dt*i)
                            exptime.append(dt)
                            pass
                        pass
                    else:
                        ftime.append(start)
                        exptime.append(fits.header['EXPTIME'])
                    pass
                data['LOC'].append(loc)
                data['TIME'].extend(ftime)
                data['EXPLEN'].extend(exptime)
                pass
            pass
        pass
    elif ('WFC3' in ext) and ('SCAN' in ext):
        for loc in sorted(clc['LOC']):
            fullloc = os.path.join(dbs, loc)
            with pyfits.open(fullloc) as hdulist:
                header0 = hdulist[0].header
                if 'SCAN_ANG' in header0: data['SCANANGLE'].append(header0['SCAN_ANG'])
                elif 'PA_V3' in header0: data['SCANANGLE'].append(header0['PA_V3'])
                else: data['SCANANGLE'].append(666)
                ftime = []
                for fits in hdulist:
                    if (fits.size != 0) and ('DELTATIM' in fits.header.keys()):
                        ftime.append(float(fits.header['ROUTTIME']))
                        pass
                    pass
                data['LOC'].append(loc)
                data['TIME'].append(np.nanmean(ftime))
                data['EXPLEN'].append(header0['EXPTIME'])
                pass
            pass
        pass
    elif 'STARE' in ext:
        for loc in sorted(clc['LOC']):
            fullloc = os.path.join(dbs, loc)
            with pyfits.open(fullloc) as hdulist:
                header0 = hdulist[0].header
                if 'SCAN_ANG' in header0: scanangle = header0['SCAN_ANG']
                elif 'PA_V3' in header0: scanangle = header0['PA_V3']
                else: scanangle = 666
                ftime = []
                allexplen = []
                allloc = []
                for fits in hdulist:
                    if (fits.size != 0) and (fits.header['EXTNAME'] in ['SCI']):
                        ftime.append(float(fits.header['EXPEND']))
                        allexplen.append(float(fits.header['EXPTIME']))
                        allloc.append(fits.header['EXPNAME'])
                        pass
                    pass
                allscanangle = [scanangle]*len(allexplen)
                data['SCANANGLE'].extend(allscanangle)
                data['LOC'].extend(allloc)
                data['TIME'].extend(ftime)
                data['EXPLEN'].extend(allexplen)
                pass
            pass
        pass
    data['IGNORED'] = [False]*len(data['LOC'])
    time = np.array(data['TIME'].copy())
    ignore = np.array(data['IGNORED'].copy())
    exposlen = np.array(data['EXPLEN'].copy())
    scanangle = np.array(data['SCANANGLE'].copy())
    ordt = np.argsort(time)
    exlto = exposlen.copy()[ordt]
    tmeto = time.copy()[ordt]
    ssc = syscore.ssconstants()
    if 'Spitzer' not in ext:
        ignto = ignore.copy()[ordt]
        scato = scanangle.copy()[ordt]
        pass

    if tmeto.size > 1:
        for p in priors['planets']:
            out['data'][p] = {}

            if 'Spitzer' in ext:
                out['data'][p]['transit'] = []
                out['data'][p]['eclipse'] = []
                out['data'][p]['phasecurve'] = []
                vis = np.ones(time.size)

                tmjd = priors[p]['t0']
                if tmjd > 2400000.5: tmjd -= 2400000.5
                smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
                tdur = priors[p]['period']/(np.pi)/smaors  # rough estimate
                pdur = tdur/priors[p]['period']
                w = priors[p].get('omega',0)
                dp = 0.5 * (1 + priors[p]['ecc']*(4./np.pi)*np.cos(np.deg2rad(w)))  # phase offset for eccentric orbit

                # shift phase so we can measure adequate baseline surrounding a transit/eclipse
                sphase = (time-(tmjd-0.25*priors[p]['period']))/priors[p]['period']
                epochs = np.unique(np.floor(sphase))
                visto = np.floor(sphase)

                for e in epochs:
                    vmask = visto == e
                    tmask = ((sphase[vmask]-e) > (0.25-1.5*pdur)) & ((sphase[vmask]-e) < (0.25+1.5*pdur))
                    if tmask.sum() > 25:
                        out['data'][p]['transit'].append(e)
                    emask = ((sphase[vmask]-e) > (0.25+dp-1.5*pdur)) & ((sphase[vmask]-e) < (0.25+dp+1.5*pdur))
                    if emask.sum() > 25:
                        out['data'][p]['eclipse'].append(e)

                    # if (tmask.sum() > 50) & (emask.sum() > 50):
                        # out['data'][p]['phasecurve'].append(e)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Rob Zellem's method of finding phase curves
                # after some thought, RZ thinks we should run this two different ways: one with the way that Kyle has been doing
                # with his shifted phasing, and use this for transits/eclipses alone
                # and then again to find phase curves

                # RZ breaks up the data by the amount of time that has passed
                deltatime = np.gradient(time[np.argsort(time)])
                # If more than 0.25 orbital phase has passed, then this classified as a new observation
                # Chose 0.25 in case a transit/eclipse is observed after eclipse/transit, but does not classify as a phase curve
                observations = deltatime >= priors[p]['period']/4  # returns a boolean
                idxobs, = np.where(observations)  # where the boundaries are index-wise

                pcvisto = np.zeros(len(time))
                for nobs in np.arange(len(idxobs)):
                    if nobs==0:
                        tstart = 0
                    else:
                        tstart = idxobs[nobs]
                    if nobs==len(idxobs)-1:
                        tstop = -1
                    else:
                        tstop = idxobs[nobs+1]

                    try:
                        obsmask = (time >= time[np.argsort(time)][tstart]) & (time <= time[np.argsort(time)][tstop])
                    except IndexError:
                        pass
                        # import pdb; pdb.set_trace()

                    pcvisto[obsmask] = np.floor((time[np.argsort(time)][tstart] - tmjd)/priors[p]['period'])

                pcepochs = np.unique(pcvisto)

                sphase = (time - tmjd)/priors[p]['period']

                for e in pcepochs:
                    vmask = pcvisto == e
                    # could make this more robust by determining if *any* out of transit
                    # Right now, it requires there to be 1 transit duration pre-/post-transit
                    tmask = ((sphase[vmask]-e) >= 1-2*pdur) & ((sphase[vmask]-e) <= 1+2*pdur)
                    # While this does find eclipses, it fails if a phase curve starts with an eclipse
                    # or if there are multiple phase curves in our dataset
                    # Need to update to check if there is at least one eclipse pre-/post- transit
                    # Right now, it requires there to be 1 transit duration pre-eclipse
                    emask = ((sphase[vmask]-e) >= 1+dp-2*pdur) & ((sphase[vmask]-e) <= 1+dp+2*pdur)
                    if (tmask.sum() > 20) & (emask.sum() > 20):
                        out['data'][p]['phasecurve'].append(e)

                # if ("4.5" in ext):
                #     import pdb; pdb.set_trace()

                pass

                out['STATUS'].append(True)
                out['data'][p]['visits'] = visto.astype(int)  # should maintain same order as time
                out['data'][p]['pcvisits'] = pcvisto  # phase curve visits
                pass
            else:  # HST
                smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
                tmjd = priors[p]['t0']
                if tmjd > 2400000.5: tmjd -= 2400000.5
                z, phase = time2z(time, priors[p]['inc'], tmjd, smaors, priors[p]['period'], priors[p]['ecc'])
                zto = z.copy()[ordt]
                phsto = phase.copy()[ordt]
                tmetod = [np.diff(tmeto)[0]]
                tmetod.extend(list(np.diff(tmeto)))
                tmetod = np.array(tmetod)
                thrs = np.percentile(tmetod, 75)
                cftfail = tmetod > 3*thrs
                if True in cftfail: thro = np.percentile(tmetod[cftfail], 75)
                else: thro = 0
                # THRESHOLDS
                rbtthr = 25e-1*thrs  # HAT-P-11
                vstthr = 3e0*thro
                # VISIT NUMBERING --------------------------------------------
                whereo = np.where(tmetod > rbtthr)[0]
                wherev = np.where(tmetod > vstthr)[0]
                visto = np.ones(tmetod.size)
                dvis = np.ones(tmetod.size)
                vis = np.ones(tmetod.size)

                for index in wherev: visto[index:] += 1
                # DOUBLE SCAN VISIT RE NUMBERING -----------------------------
                dvisto = visto.copy()
                for v in set(visto):
                    selv = (visto == v)
                    vordsa = scato[selv].copy()
                    if len(set(vordsa)) > 1:
                        dvisto[visto > v] = dvisto[visto > v] + 1
                        dbthr = np.mean(list(set(vordsa)))
                        vdbvisto = dvisto[selv].copy()
                        vdbvisto[vordsa > dbthr] = vdbvisto[vordsa > dbthr] + 1
                        dvisto[selv] = vdbvisto
                        pass
                    pass
                # ORBIT NUMBERING --------------------------------------------
                orbto = np.ones(tmetod.size)
                orb = np.ones(tmetod.size)
                for v in set(visto):
                    selv = (visto == v)
                    if len(~ignto[selv]) < 4: ignto[selv] = True
                    else:
                        select = np.where(tmetod[selv] > rbtthr)[0]
                        incorb = orbto[selv]
                        for indice in select: incorb[indice:] = incorb[indice:] + 1
                        orbto[selv] = incorb
                        for o in set(orbto[selv]):
                            selo = (orbto[selv] == o)
                            if len(~ignto[selv][selo]) < 4:
                                visignto = ignto[selv]
                                visignto[selo] = True
                                ignto[selv] = visignto
                                pass
                            ref = np.median(exlto[selv][selo])
                            if len(set(exlto[selv][selo])) > 1:
                                rej = (exlto[selv][selo] != ref)
                                ovignto = ignto[selv][selo]
                                ovignto[rej] = True
                                ignto[selv][selo] = ovignto
                                pass
                            pass
                        pass
                    pass
                # TRANSIT VISIT PHASECURVE -----------------------------------
                out['data'][p]['svntransit'] = []
                out['data'][p]['svneclipse'] = []
                out['data'][p]['svnphasecurve'] = []
                for v in set(visto):
                    selv = (visto == v)
                    trlim = 1e0
                    posphsto = phsto.copy()
                    posphsto[posphsto < 0] = posphsto[posphsto < 0] + 1e0
                    tecrit = abs(np.arcsin(trlim/smaors))/(2e0*np.pi)
                    select = (abs(zto[selv]) < trlim)
                    pcconde = False
                    if np.any(select) and (np.min(abs(posphsto[selv][select] - 0.5)) < tecrit):
                        out['eclipse'].append(int(v))
                        out['data'][p]['svneclipse'].append(int(v))
                        pcconde = True
                        pass
                    pccondt = False
                    if np.any(select) and (np.min(abs(phsto[selv][select])) < tecrit):
                        out['transit'].append(int(v))
                        out['data'][p]['svntransit'].append(int(v))
                        pccondt = True
                        pass
                    if pcconde and pccondt:
                        out['phasecurve'].append(int(v))
                        out['data'][p]['svnphasecurve'].append(int(v))
                        pass
                    pass
                out['data'][p]['transit'] = []
                out['data'][p]['eclipse'] = []
                out['data'][p]['phasecurve'] = []
                for v in set(dvisto):
                    selv = (dvisto == v)
                    trlim = 1e0
                    posphsto = phsto.copy()
                    posphsto[posphsto < 0] = posphsto[posphsto < 0] + 1e0
                    tecrit = abs(np.arcsin(trlim/smaors))/(2e0*np.pi)
                    select = (abs(zto[selv]) < trlim)
                    pcconde = False
                    if np.any(select)and(np.min(abs(posphsto[selv][select] - 0.5)) < tecrit):
                        out['data'][p]['eclipse'].append(int(v))
                        pcconde = True
                        pass
                    pccondt = False
                    if np.any(select) and (np.min(abs(phsto[selv][select])) < tecrit):
                        out['data'][p]['transit'].append(int(v))
                        pccondt = True
                        pass
                    if pcconde and pccondt: out['data'][p]['phasecurve'].append(int(v))
                    pass
                vis[ordt] = visto.astype(int)
                orb[ordt] = orbto.astype(int)
                dvis[ordt] = dvisto.astype(int)
                ignore[ordt] = ignto

                # PLOTS ------------------------------------------------------
                if verbose:
                    plt.figure()
                    plt.plot(phsto, 'k.')
                    plt.plot(np.arange(phsto.size)[~ignto], phsto[~ignto], 'bo')
                    for i in wherev: plt.axvline(i, ls='--', color='r')
                    for i in whereo: plt.axvline(i, ls='-.', color='g')
                    plt.xlim(0, tmetod.size - 1)
                    plt.ylim(-0.5, 0.5)
                    plt.xlabel('Time index')
                    plt.ylabel('Orbital Phase [2pi rad]')

                    plt.figure()
                    plt.plot(tmetod, 'o')
                    plt.plot(tmetod*0+vstthr, 'r--')
                    plt.plot(tmetod*0+rbtthr, 'g-.')
                    for i in wherev: plt.axvline(i, ls='--', color='r')
                    for i in whereo: plt.axvline(i, ls='-.', color='g')
                    plt.xlim(0, tmetod.size - 1)
                    plt.xlabel('Time index')
                    plt.ylabel('Frame Separation [Days]')
                    plt.semilogy()

                    if np.max(dvis) > np.max(vis):
                        plt.figure()
                        plt.plot(dvisto, 'o')
                        plt.xlim(0, tmetod.size - 1)
                        plt.ylim(1, np.max(dvisto))
                        plt.xlabel('Time index')
                        plt.ylabel('Double Scan Visit Number')
                        plt.show()
                        pass
                    else: plt.show()
                    pass
                out['data'][p]['tmetod'] = tmetod
                out['data'][p]['whereo'] = whereo
                out['data'][p]['wherev'] = wherev
                out['data'][p]['thrs'] = rbtthr
                out['data'][p]['thro'] = vstthr
                out['data'][p]['visits'] = vis
                out['data'][p]['orbits'] = orb
                out['data'][p]['dvisits'] = dvis
                out['data'][p]['z'] = z
                out['data'][p]['phase'] = phase
                out['data'][p]['ordt'] = ordt
                out['data'][p]['ignore'] = ignore
                out['STATUS'].append(True)
                pass

            log.warning('>-- Planet: %s', p)
            log.warning('--< Transit: %s', str(out['data'][p]['transit']))
            log.warning('--< Eclipse: %s', str(out['data'][p]['eclipse']))
            log.warning('--< Phase Curve: %s', str(out['data'][p]['phasecurve']))

            if out['data'][p]['transit'] or out['data'][p]['eclipse'] or out['data'][p]['phasecurve']: chunked = True
    return chunked
# ------------ -------------------------------------------------------
# -- CALIBRATE SCAN DATA -- ------------------------------------------
def scancal(clc, tim, tid, flttype, out,
            emptythr=1e3, frame2png=False, verbose=False, debug=False):
    '''
    G. ROUDIER: Extracts and Wavelength calibrates WFC3 SCAN mode spectra
    '''
    # VISIT ------------------------------------------------------------------------------
    for pkey in tim['data'].keys(): visits = np.array(tim['data'][pkey]['visits'])
    for pkey in tim['data'].keys(): dvisits = np.array(tim['data'][pkey]['dvisits'])
    # DATA TYPE --------------------------------------------------------------------------
    arcsec2pix = dps(flttype)
    vrange = validrange(flttype)
    wvrng, disper, ldisp, udisp = fng(flttype)
    spectrace = np.round((np.max(wvrng) - np.min(wvrng))/disper)
    # LOAD DATA --------------------------------------------------------------------------
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')
    data = {'LOC':[], 'EPS':[], 'DISPLIM':[ldisp, udisp],
            'SCANRATE':[], 'SCANLENGTH':[], 'SCANANGLE':[],
            'EXP':[], 'EXPERR':[], 'EXPFLAG':[], 'VRANGE':vrange,
            'TIME':[], 'EXPLEN':[], 'MIN':[], 'MAX':[], 'TRIAL':[]}
    for loc in sorted(clc['LOC']):
        fullloc = os.path.join(dbs, loc)
        with pyfits.open(fullloc) as hdulist:
            header0 = hdulist[0].header
            test = header0['UNITCORR']
            eps = False
            if test in ['COMPLETE', 'PERFORM']: eps = True
            data['EPS'].append(eps)
            if 'SCAN_RAT' in header0: data['SCANRATE'].append(header0['SCAN_RAT'])
            else: data['SCANRATE'].append(np.nan)
            if 'SCAN_LEN' in header0: data['SCANLENGTH'].append(header0['SCAN_LEN'])
            else: data['SCANLENGTH'].append(np.nan)
            if 'SCAN_ANG' in header0: data['SCANANGLE'].append(header0['SCAN_ANG'])
            elif 'PA_V3' in header0: data['SCANANGLE'].append(header0['PA_V3'])
            else: data['SCANANGLE'].append(666)
            frame = []
            errframe = []
            dqframe = []
            ftime = []
            fmin = []
            fmax = []
            for fits in hdulist:
                if (fits.size != 0) and ('DELTATIM' in fits.header.keys()):
                    fitsdata = np.empty(fits.data.shape)
                    fitsdata[:] = fits.data[:]
                    frame.append(fitsdata)
                    ftime.append(float(fits.header['ROUTTIME']))
                    fmin.append(float(fits.header['GOODMIN']))
                    fmax.append(float(fits.header['GOODMAX']))
                    del fits.data
                    pass
                if 'EXTNAME' in fits.header:
                    if (fits.header['EXTNAME'] in ['ERR', 'DQ']):
                        fitsdata = np.empty(fits.data.shape)
                        fitsdata[:] = fits.data[:]
                        if fits.header['EXTNAME'] == 'ERR': errframe.append(fitsdata)
                        if fits.header['EXTNAME'] == 'DQ': dqframe.append(fitsdata)
                        del fits.data
                        pass
                    if eps and (fits.header['EXTNAME'] == 'TIME'):
                        xpsrl = np.array(float(fits.header['PIXVALUE']))
                        frame[-1] = frame[-1]*xpsrl
                        errframe[-1] = errframe[-1]*xpsrl
                        pass
                    pass
                pass
            data['LOC'].append(loc)
            data['EXP'].append(frame)
            data['EXPERR'].append(errframe)
            data['EXPFLAG'].append(dqframe)
            data['TIME'].append(ftime)
            data['EXPLEN'].append(header0['EXPTIME'])
            data['MIN'].append(fmin)
            data['MAX'].append(fmax)
            pass
        pass
    # MASK DATA --------------------------------------------------------------------------
    data['MEXP'] = data['EXP'].copy()
    data['MASK'] = data['EXPFLAG'].copy()
    data['IGNORED'] = [False]*len(data['LOC'])
    data['FLOODLVL'] = [np.nan]*len(data['LOC'])
    data['UP'] = [np.nan]*len(data['LOC'])
    data['DOWN'] = [np.nan]*len(data['LOC'])
    data['TRIAL'] = ['']*len(data['LOC'])
    for index, nm in enumerate(data['LOC']):
        maskedexp = []
        masks = []
        ignore = False
        for dd, ff in zip(data['EXP'][index], data['EXPFLAG'][index]):
            select = ff > 0
            if np.sum(select) > 0:
                dd[select] = np.nan
                if np.all(~np.isfinite(dd)):
                    data['TRIAL'][index] = 'Empty Subexposure'
                    ignore = True
                    pass
                else: maskedexp.append(dd)
                pass
            else: maskedexp.append(dd)
            mm = np.isfinite(dd)
            masks.append(mm)
            pass
        if ignore: maskedexp = data['EXP'][index].copy()
        data['MEXP'][index] = maskedexp
        data['MASK'][index] = masks
        data['IGNORED'][index] = ignore
        pass
    # ALL FLOOD LEVELS -------------------------------------------------------------------
    for index, nm in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        # MINKOWSKI ----------------------------------------------------------------------
        psdiff = np.diff(data['MEXP'][index][::-1].copy(), axis=0)
        floatsw = data['SCANLENGTH'][index]/arcsec2pix
        scanwdw = np.round(floatsw)
        if (scanwdw > psdiff[0].shape[0]) or (len(psdiff) < 2):
            scanwpi = np.round(floatsw/(len(psdiff)))
            pass
        else: scanwpi = np.round(floatsw/(len(psdiff) - 1))
        if scanwpi < 1:
            data['TRIAL'][index] = 'Subexposure Scan Length < 1 Pixel'
            ignore = True
            pass
        if not ignore:
            targetn = 0
            if tid in ['XO-2', 'HAT-P-1']: targetn = -1
            minlocs = []
            maxlocs = []
            floodlist = []
            for de, md in zip(psdiff[::-1], data['MIN'][index][::-1]):
                valid = np.isfinite(de)
                if np.nansum(~valid) > 0: de[~valid] = 0
                select = de[valid] < md
                if np.nansum(select) > 0: de[valid][select] = 0
                perfldlist = np.nanpercentile(de, np.arange(1001)/1e1)
                perfldlist = np.diff(perfldlist)
                perfldlist[:100] = 0
                perfldlist[-1] = 0
                indperfld = list(perfldlist).index(np.max(perfldlist))*1e-1
                floodlist.append(np.nanpercentile(de, indperfld))
                pass
            fldthr = np.nanmax(floodlist)
            # CONTAMINATION FROM ANOTHER SOURCE IN THE UPPER FRAME -----------------------
            if tid in ['HAT-P-41']: fldthr /= 1.5
            if 'G102' in flttype: fldthr /= 3e0
            data['FLOODLVL'][index] = fldthr
            pass
        pass
    allfloodlvl = np.array(data['FLOODLVL'])
    for dv in set(dvisits):
        allfloodlvl[dvisits == dv] = np.nanmedian(allfloodlvl[dvisits == dv])
        pass
    data['FLOODLVL'] = allfloodlvl
    # ALL LIMITS  ------------------------------------------------------------------------
    for index, nm in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        # MINKOWSKI FLOOD LEVEL ----------------------------------------------------------
        psdiff = np.diff(data['MEXP'][index][::-1].copy(), axis=0)
        floatsw = data['SCANLENGTH'][index]/arcsec2pix
        scanwdw = np.round(floatsw)
        if (scanwdw > psdiff[0].shape[0]) or (len(psdiff) < 2):
            scanwpi = np.round(floatsw/(len(psdiff)))
            pass
        else: scanwpi = np.round(floatsw/(len(psdiff) - 1))
        if scanwpi < 1:
            data['TRIAL'][index] = 'Subexposure Scan Length < 1 Pixel'
            ignore = True
            pass
        if not ignore:
            targetn = 0
            if tid in ['XO-2', 'HAT-P-1']: targetn = -1
            minlocs = []
            maxlocs = []
            for de, md in zip(psdiff.copy()[::-1], data['MIN'][index][::-1]):
                lmn, lmx = isolate(de, md, spectrace, scanwpi, targetn,
                                   data['FLOODLVL'][index])
                minlocs.append(lmn)
                maxlocs.append(lmx)
                pass
            # HEAVILY FLAGGED SCAN -------------------------------------------------------
            nanlocs = np.all(~np.isfinite(minlocs)) or np.all(~np.isfinite(maxlocs))
            almstare = scanwpi < 5
            if nanlocs or almstare:
                for de, md in zip(psdiff.copy()[::-1], data['MIN'][index][::-1]):
                    if (scanwpi/3) < 2: redscanwpi = scanwpi/2
                    else: redscanwpi = scanwpi/3
                    lmn, lmx = isolate(de, md, spectrace, redscanwpi, targetn,
                                       data['FLOODLVL'][index])
                    minlocs.append(lmn)
                    maxlocs.append(lmx)
                    pass
                pass
            ignore = ignore or not((np.any(np.isfinite(minlocs))) and
                                   (np.any(np.isfinite(maxlocs))))
            if not ignore:
                minl = np.nanmin(minlocs)
                maxl = np.nanmax(maxlocs)
                # CONTAMINATION FROM ANOTHER SOURCE IN THE UPPER FRAME -------------------
                if (tid in ['HAT-P-41']) and ((maxl - minl) > 15): minl = maxl - 15
                if minl < 10: minl = 10
                if maxl > (psdiff[0].shape[0] - 10): maxl = psdiff[0].shape[0] - 10
                pass
            else:
                minl = np.nan
                maxl = np.nan
                pass
            data['UP'][index] = minl
            data['DOWN'][index] = maxl
            pass
        pass
    allminl = np.array(data['UP'])
    allmaxl = np.array(data['DOWN'])
    for dv in set(dvisits):
        allminl[dv == dvisits] = np.nanpercentile(allminl[dv == dvisits], 2.5)
        allmaxl[dv == dvisits] = np.nanpercentile(allmaxl[dv == dvisits], 97.5)
        pass
    data['UP'] = allminl
    data['DOWN'] = allmaxl
    allscanlen = np.array(data['SCANLENGTH'])
    allignore = np.array(data['IGNORED'])
    alltrials = np.array(['Exposure Scan Length Rejection']*len(data['TRIAL']))
    for v in set(visits):
        allfloodlvl[visits == v] = np.nanmedian(allfloodlvl[visits == v])
        visitign = allignore[visits == v]
        visittrials = alltrials[visits == v]
        select = allscanlen[visits == v] != np.nanmedian(allscanlen[visits == v])
        visitign[select] = True
        visittrials[~select] = ''
        allignore[visits == v] = visitign
        alltrials[visits == v] = visittrials
        pass
    data['FLOODLVL'] = list(allfloodlvl)
    data['IGNORED'] = list(allignore)
    data['TRIAL'] = list(alltrials)
    ovszspc = False
    # BACKGROUND SUB AND ISOLATE ---------------------------------------------------------
    for index, nm in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        psdiff = np.diff(data['MEXP'][index][::-1].copy(), axis=0)
        psminsel = np.array(data['MIN'][index]) < 0
        if True in psminsel: psmin = np.nansum(np.array(data['MIN'][index])[psminsel])
        else: psmin = np.nanmin(data['MIN'][index])
        minl = data['UP'][index]
        maxl = data['DOWN'][index]
        if not ignore:
            # BACKGROUND SUBTRACTION -----------------------------------------------------
            for eachdiff in psdiff:
                background = []
                for eachcol in eachdiff.T:
                    if True in np.isfinite(eachcol):
                        selfinite = np.isfinite(eachcol)
                        fineachcol = eachcol[selfinite]
                        test = fineachcol < psmin
                        if True in test: fineachcol[test] = np.nan
                        eachcol[selfinite] = fineachcol
                        pass
                    nancounts = np.sum(~np.isfinite(eachcol))
                    thr = 1e2*(1e0 - (scanwpi + nancounts)/eachcol.size)
                    if thr <= 0: bcke = np.nan
                    else:
                        test = eachcol < np.nanpercentile(eachcol, thr)
                        if True in test: bcke = np.nanmedian(eachcol[test])
                        else: bcke = np.nan
                        pass
                    background.append(bcke)
                    pass
                background = np.array([np.array(background)]*eachdiff.shape[0])
                eachdiff -= background
                eachdiff[:int(minl),:] = 0
                eachdiff[int(maxl):,:] = 0
                pass
            # DIFF ACCUM -----------------------------------------------------------------
            thispstamp = np.nansum(psdiff, axis=0)
            thispstamp[thispstamp <= psmin] = np.nan
            thispstamp[thispstamp == 0] = np.nan
            if abs(spectrace - thispstamp.shape[1]) < 36: ovszspc = True
            # PLOTS ----------------------------------------------------------------------
            if debug:
                show = thispstamp.copy()
                valid = np.isfinite(show)
                show[~valid] = 0
                show[show < data['FLOODLVL'][index]] = np.nan
                plt.figure()
                plt.title('Isolate Flood Level')
                plt.imshow(show)
                plt.colorbar()

                plt.figure()
                plt.title('Diff Accum')
                plt.imshow(thispstamp)
                plt.colorbar()

                colorlim = np.nanmin(thispstamp)
                plt.figure()
                plt.title('Background Sub')
                plt.imshow(thispstamp)
                plt.colorbar()
                plt.clim(colorlim, abs(colorlim))
                plt.show()
                pass
            # ISOLATE SCAN X -------------------------------------------------------------
            mltord = thispstamp.copy()
            targetn = 0
            if tid in ['XO-2']: targetn = -1
            minx, maxx = isolate(mltord, psmin, spectrace, scanwpi, targetn,
                                 data['FLOODLVL'][index], axis=0, debug=False)
            if np.isfinite(minx*maxx):
                minx -= (1.5*12)
                maxx += (1.5*12)
                if minx < 0: minx = 5
                thispstamp[:,:int(minx)] = np.nan
                if maxx > (thispstamp.shape[1] - 1): maxx = thispstamp.shape[1] - 5
                thispstamp[:,int(maxx):] = np.nan
                if ((maxx - minx) < spectrace) and not ovszspc:
                    data['TRIAL'][index] = 'Could Not Find Full Spectrum'
                    ignore = True
                    pass
                pstamperr = np.array(data['EXPERR'][index].copy())
                select = ~np.isfinite(pstamperr)
                if np.nansum(select) > 0: pstamperr[select] = 0
                pstamperr = np.sqrt(np.nansum(pstamperr**2, axis=0))
                select = ~np.isfinite(thispstamp)
                if np.nansum(select) > 0: pstamperr[select] = np.nan
                pass
            else:
                garbage = data['EXP'][index][::-1].copy()
                thispstamp = np.sum(np.diff(garbage, axis=0), axis=0)
                pstamperr = thispstamp*np.nan
                data['TRIAL'][index] = 'Could Not Find X Edges'
                ignore = True
                pass
            pass
        else:
            garbage = data['EXP'][index][::-1].copy()
            thispstamp = np.sum(np.diff(garbage, axis=0), axis=0)
            pstamperr = thispstamp*np.nan
            if len(data['TRIAL'][index]) < 1:
                data['TRIAL'][index] = 'Could Not Find Y Edges'
                pass
            ignore = True
            pass
        data['MEXP'][index] = thispstamp
        data['TIME'][index] = np.nanmean(data['TIME'][index].copy())
        data['IGNORED'][index] = ignore
        data['EXPERR'][index] = pstamperr
        if debug: log.warning('>-- %s / %s', str(index), str(len(data['LOC'])-1))
        # PLOTS --------------------------------------------------------------------------
        if frame2png:
            if not os.path.exists('TEST'): os.mkdir('TEST')
            if not os.path.exists('TEST/'+tid): os.mkdir('TEST/'+tid)
            plt.figure()
            plt.title('Index: '+str(index)+' Ignored='+str(ignore))
            plt.imshow(thispstamp)
            plt.colorbar()
            plt.savefig('TEST/'+tid+'/'+nm+'.png')
            plt.close()
            pass
        pass
    maxwasize = []
    for mexp in data['MEXP']: maxwasize.append(mexp.shape[1])
    maxwasize = np.nanmax(maxwasize)
    # SPECTRUM EXTRACTION ----------------------------------------------------------------
    data['SPECTRUM'] = [np.array([np.nan]*maxwasize)]*len(data['LOC'])
    data['SPECERR'] = [np.array([np.nan]*maxwasize)]*len(data['LOC'])
    data['NSPEC'] = [np.nan]*len(data['LOC'])
    for index, loc in enumerate(data['LOC']):
        floodlevel = data['FLOODLVL'][index]
        if floodlevel < emptythr:
            data['IGNORED'][index] = True
            data['TRIAL'][index] = 'Empty Frame'
            pass
        ignore = data['IGNORED'][index]
        if not ignore:
            frame = data['MEXP'][index].copy()
            frame = [line for line in frame if not np.all(~np.isfinite(line))]
            # OVERSIZED MASK -------------------------------------------------------------
            for line in frame:
                if np.nanmax(line) < floodlevel: line *= np.nan
                if abs(spectrace - line.size) < 36: ovszspc = True
                elif np.sum(np.isfinite(line)) < spectrace: line *= np.nan
                pass
            frame = [line for line in frame if not np.all(~np.isfinite(line))]
            # SCAN RATE CORRECTION -------------------------------------------------------
            template = []
            for col in np.array(frame).T:
                if not np.all(~np.isfinite(col)): template.append(np.nanmedian(col))
                else: template.append(np.nan)
                pass
            template = np.array(template)
            for line in frame:
                errref = np.sqrt(abs(line))/abs(template)
                line /= template
                refline = np.nanmedian(line)
                select = np.isfinite(line)
                minok = (abs(line[select] - refline) < 3e0*np.nanmin(errref[select]))
                if np.nansum(minok) > 0:
                    alpha = np.nansum(minok)/np.nansum(line[select][minok])
                    line *= alpha
                    line *= template
                    pass
                else: line *= np.nan
                pass
            frame = [line for line in frame if not np.all(~np.isfinite(line))]
            if debug:
                display = [np.array(line)/template for line in frame]
                plt.figure()
                plt.imshow(np.array(display))
                plt.colorbar()
                plt.clim(0.95, 1.05)
                plt.title(str(index))
                plt.show()
                pass
            spectrum = []
            specerr = []
            nspectrum = []
            vtemplate = []
            for row in np.array(frame):
                if not np.all(~np.isfinite(row)): vtemplate.append(np.nanmedian(row))
                else: vtemplate.append(np.nan)
                pass
            vtemplate = np.array(vtemplate)
            for col in np.array(frame).T:
                ignorecol = False
                if not np.all(~np.isfinite(col)):
                    errref = (np.sqrt(abs(np.nanmedian(col)))/abs(vtemplate))
                    ratio = col/vtemplate
                    refline = np.nanmedian(ratio)
                    select = np.isfinite(col)
                    ok = (abs(ratio[select] - refline) < 3e0*errref[select])
                    if np.nansum(ok) > 0:
                        alpha = np.nansum(ok)/np.nansum(ratio[select][ok])
                        valid = (abs(col[select]*alpha - vtemplate[select]) <
                                 3*np.sqrt(abs(vtemplate[select])))
                        pass
                    else: valid = [False]
                    if np.nansum(valid) > 0:
                        spectrum.append(np.nanmedian(col[select][valid]))
                        specerr.append(np.nanstd(col[select][valid]))
                        nspectrum.append(np.nansum(valid))
                        pass
                    else: ignorecol = True
                else: ignorecol = True
                if ignorecol:
                    spectrum.append(np.nan)
                    specerr.append(np.nan)
                    nspectrum.append(0)
                    pass
                pass
            spectrum = np.array(spectrum)
            spectrum -= np.nanmin(spectrum)
            # EXCLUDE RESIDUAL GLITCHES
            seloutlrs = np.isfinite(template) & np.isfinite(spectrum)
            if True in seloutlrs:
                nanme = (abs(spectrum[seloutlrs] - template[seloutlrs])/
                         template[seloutlrs]) > 1e0
                if True in nanme: spectrum[seloutlrs][nanme] = np.nan
                pass
            else:
                data['IGNORED'][index] = True
                data['TRIAL'][index] = 'Invalid Spectrum/Template'
                pass
            # TRUNCATED SPECTRUM
            # testspec = spectrum[np.isfinite(spectrum)]
            # if (np.all(testspec[-18:] > emptythr)) and not ovszspc:
            #    data['IGNORED'][index] = True
            #    data['TRIAL'][index] = 'Truncated Spectrum'
            #    pass
            data['SPECTRUM'][index] = np.array(spectrum)
            data['SPECERR'][index] = np.array(specerr)
            data['NSPEC'][index] = np.array(nspectrum)
            pass
        pass
    # PLOT -------------------------------------------------------------------------------
    if debug:
        for v in set(visits):
            plt.figure()
            for spec, vi in zip(data['SPECTRUM'], visits):
                if vi == v: plt.plot(spec)
                pass
            plt.ylabel('Stellar Spectra [Counts]')
            plt.xlabel('Pixel Number')
            plt.title('Visit ' + str(int(v)))
            plt.show()
            pass
        pass
    # WAVELENGTH CALIBRATION -------------------------------------------------------------
    wavett, tt = ag2ttf(flttype)
    if ovszspc:
        select = (wavett*1e-4 < 1.68) & (wavett*1e-4 > 1.09)
        wavett = wavett[select]
        tt = tt[select]
        pass
    scaleco = np.nanmax(tt) / np.nanmin(tt[tt > 0])
    data['PHT2CNT'] = [np.nan]*len(data['LOC'])
    data['WAVE'] = [np.array([np.nan]*maxwasize)]*len(data['LOC'])
    data['DISPERSION'] = [np.nan]*len(data['LOC'])
    data['SHIFT'] = [np.nan]*len(data['LOC'])
    data['BACKGROUND'] = [np.nan]*len(data['LOC'])
    spectralindex = []
    for index, loc in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        if not ignore:
            spectrum = data['SPECTRUM'][index].copy()
            cutoff = np.nanmax(spectrum)/scaleco
            finitespec = spectrum[np.isfinite(spectrum)]
            test = finitespec < cutoff
            if True in test:
                finitespec[test] = np.nan
                spectrum[np.isfinite(spectrum)] = finitespec
                pass
            wave, disp, shift, si, bck = wavesol(abs(spectrum), tt, wavett, disper,
                                                 ovszspc=ovszspc, bck=None, debug=debug)
            if (disp > ldisp) and (disp < udisp): spectralindex.append(si)
            pass
        pass
    siv = np.nanmedian(spectralindex)
    for index, loc in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        if not ignore:
            spectrum = data['SPECTRUM'][index].copy()
            cutoff = np.nanmax(spectrum)/scaleco
            finitespec = spectrum[np.isfinite(spectrum)]
            test = finitespec < cutoff
            if True in test:
                finitespec[test] = np.nan
                spectrum[np.isfinite(spectrum)] = finitespec
                pass
            wave, disp, shift, si, bck = wavesol(abs(spectrum), tt, wavett, disper,
                                                 siv=siv, ovszspc=ovszspc,
                                                 bck=None, debug=debug)
            if (disp < ldisp) or (disp > udisp):
                data['TRIAL'][index] = 'Dispersion Out Of Bounds'
                ignore = True
                pass
            if (abs(disp - disper) < 1e-7) and not ovszspc:
                data['TRIAL'][index] = 'Dispersion Fit Failure'
                ignore = True
                pass
            pass
        if not ignore:
            liref = itp.interp1d(wavett*1e-4, tt, bounds_error=False, fill_value=np.nan)
            phot2counts = liref(wave)
            data['PHT2CNT'][index] = phot2counts
            data['WAVE'][index] = wave  # MICRONS
            data['DISPERSION'][index] = disp  # ANGSTROMS/PIXEL
            data['SHIFT'][index] = shift*1e4/disp  # PIXELS
            data['BACKGROUND'][index] = bck
            data['SPECTRUM'][index] = data['SPECTRUM'][index] - bck
            pass
        else: data['WAVE'][index] = (data['SPECTRUM'][index])*np.nan
        data['IGNORED'][index] = ignore
        pass
    # PLOTS ------------------------------------------------------------------------------
    if verbose and (not np.all(data['IGNORED'])):
        alltime = np.array([d for d,i in zip(data['TIME'], data['IGNORED']) if not i])
        dispersion = np.array([d for d,i in zip(data['DISPERSION'], data['IGNORED'])
                               if not i])
        shift = np.array([d for d,i in zip(data['SHIFT'], data['IGNORED']) if not i])
        spec = np.array([d for d,i in zip(data['SPECTRUM'], data['IGNORED']) if not i])
        photoc = np.array([d for d,i in zip(data['PHT2CNT'], data['IGNORED']) if not i])
        wave = np.array([d for d,i in zip(data['WAVE'], data['IGNORED']) if not i])
        errspec = np.array([d for d,i in zip(data['SPECERR'], data['IGNORED']) if not i])
        allb = np.array([d for d,i in zip(data['BACKGROUND'], data['IGNORED']) if not i])
        torder = np.argsort(alltime)
        vrange = data['VRANGE']
        allerr = []
        for s, e, w in zip(spec, errspec, wave):
            select = (w > vrange[0]) & (w < vrange[1])
            allerr.extend(e[select]/np.sqrt(s[select]))
            pass
        allerr = np.array(allerr)
        select = np.isfinite(allerr)
        allerr = allerr[select]
        allerr = allerr[allerr > 0.9]

        plt.figure()
        for spectrum in data['SPECTRUM']: plt.plot(spectrum)
        plt.ylabel('Stellar Spectra [Counts]')
        plt.xlabel('Pixel Number')

        plt.figure()
        for w, p, s in zip(wave, photoc, spec):
            select = (w > vrange[0]) & (w < vrange[1])
            plt.plot(w[select], s[select]/p[select])
            pass
        plt.ylabel('Stellar Spectra [Photons]')
        plt.xlabel('Wavelength [microns]')

        plt.figure()
        plt.hist(allerr)
        plt.xlabel('Error Distribution [Noise Model Units]')

        plt.figure()
        plt.plot(dispersion[torder], 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Dispersion [Angstroms/Pixel]')
        plt.ylim(data['DISPLIM'][0], data['DISPLIM'][1])

        plt.figure()
        plt.plot(shift[torder] - np.nanmin(shift), 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Shift [Pixels]')

        plt.figure()
        plt.plot(allb[torder], 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Background [DN]')
        plt.show()
        pass
    allignore = data['IGNORED']
    allculprits = data['TRIAL']
    log.warning('>-- IGNORED: %s / %s', str(np.nansum(allignore)), str(len(allignore)))
    for index, ignore in enumerate(allignore):
        if ignore: log.warning('>-- %s: %s', str(index), str(allculprits[index]))
        pass
    data.pop('EXP', None)
    data.pop('EXPFLAG', None)
    for key in data: out['data'][key] = data[key]
    caled = not np.all(data['IGNORED'])
    if caled: out['STATUS'].append(True)
    return caled
# ------------------------- ------------------------------------------
# -- DETECTOR PLATE SCALE -- -----------------------------------------
def dps(flttype):
    '''
    G. ROUDIER: Detector plate scale

    http://www.stsci.edu/hst/wfc3/ins_performance/detectors
    http://www.stsci.edu/hst/stis/design/detectors
    http://www.stsci.edu/hst/stis/design/gratings
    '''
    detector = flttype.split('-')[2]
    fltr = flttype.split('-')[3]
    arcsec2pix = None
    if detector in ['IR']: arcsec2pix = 0.13
    if detector in ['UVIS']: arcsec2pix = 0.04
    if detector in ['CCD']: arcsec2pix = 0.05071
    if detector in ['FUV.MAMA']:
        if fltr in ['G140M']: arcsec2pix = np.sqrt(0.029*0.036)
        pass
    return arcsec2pix
# --------------------------------------------------------------------
# -- SCIENCE WAVELENGTH BAND -- --------------------------------------
def validrange(flttype):
    '''
    G. ROUDIER: Science wavelength band

    http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-18.pdf
    http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-17.pdf
    http://www.stsci.edu/hst/stis/design/gratings/documents/handbooks/currentIHB/c13_specref07.html
    http://www.stsci.edu/hst/stis/design/gratings/documents/handbooks/currentIHB/c13_specref16.html
    http://www.stsci.edu/hst/stis/design/gratings/documents/handbooks/currentIHB/c13_specref05.html
        '''
    fltr = flttype.split('-')[3]
    vrange = None
    if fltr in ['G141']: vrange = [1.12, 1.65]  # MICRONS
    if fltr in ['G102']: vrange = [0.80, 1.14]
    if fltr in ['G430L']: vrange = [0.30, 0.55]
    if fltr in ['G140M']: vrange = [0.12, 0.17]
    if fltr in ['G750L']: vrange = [0.55, 0.95]
    if fltr in ['3.6']: vrange = [3.1,3.92]
    if fltr in ['4.5']: vrange = [3.95,4.95]
    return vrange
# ----------------------------- --------------------------------------
# -- FILTERS AND GRISMS -- -------------------------------------------
def fng(flttype):
    '''
    G. ROUDIER: Filters and grisms

    http://www.stsci.edu/hst/wfc3/documents/handbooks/currentDHB/wfc3_dhb.pdf
    G102 http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-18.pdf
    G141 http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-17.pdf
    http://www.stsci.edu/hst/stis/design/gratings/documents/handbooks/currentIHB/c13_specref07.html
    http://www.stsci.edu/hst/stis/design/gratings/documents/handbooks/currentIHB/c13_specref16.html
    http://www.stsci.edu/hst/stis/design/gratings/documents/handbooks/currentIHB/c13_specref05.html
    '''
    fltr = flttype.split('-')[3]
    wvrng = None
    disp = None
    if fltr == 'G141':
        wvrng = [1085e1, 17e3]  # Angstroms
        disp = 46.5  # Angstroms/Pixel
        llim = 45
        ulim = 47.7  # HD 189733 TRANSIT VS 47.5 STSCI
        pass
    if fltr == 'G102':
        wvrng = [8e3, 115e2]  # Angstroms
        disp = 24.5  # Angstroms/Pixel
        llim = 23.5
        ulim = 25
        pass
    if fltr == 'G430L':
        wvrng = [29e2, 57e2]  # Angstroms
        disp = 2.73  # Angstroms/Pixel
        llim = 2.6
        ulim = 2.9
        pass
    if fltr == 'G750L':
        wvrng = [524e1, 1027e1]  # Angstroms
        disp = 4.92  # Angstroms/Pixel
        llim = 4.91
        ulim = 4.93
        pass
    # if fltr == '3.6':
    #     wvrng = [3100e1, 3950e1 ]  # Angstroms
    #     disp = 0  # Angstroms/Pixel
    #     llim = 0
    #     ulim = 0
    #     pass
    # if fltr == '4.5':
    #     wvrng = [3900e1, 4950e1 ]  # Angstroms
    #     disp = 0  # Angstroms/Pixel
    #     llim = 0
    #     ulim = 0
    #     pass
    return wvrng, disp, llim, ulim
# ------------------------ -------------------------------------------
# -- ISOLATE -- ------------------------------------------------------
def isolate(thisdiff, psmin, spectrace, scanwdw, targetn, floodlevel,
            axis=1, debug=False, stare=False):
    '''
    G. ROUDIER: Based on Minkowski functionnals decomposition algorithm
    '''
    valid = np.isfinite(thisdiff)
    if np.nansum(~valid) > 0: thisdiff[~valid] = 0
    select = thisdiff[valid] < psmin
    if np.nansum(select) > 0: thisdiff[valid][select] = 0
    flooded = thisdiff.copy()
    flooded[thisdiff < floodlevel] = 0
    eachprofile = np.nansum(flooded, axis=axis)
    eachpixels = np.arange(eachprofile.size)
    loc = eachpixels[eachprofile > 0]
    if loc.__len__() > 0:
        diffloc = [0]
        diffloc.extend(np.diff(loc))
        minlocs = [np.nanmin(loc)]
        maxlocs = []
        cvcount = 0
        if axis > 0:
            thr = int(scanwdw - 6)
            if thr <= 0: thr = scanwdw
            thrw = int(scanwdw - 6)
            if thrw <= 0: thrw = scanwdw
            pass
        else:
            thr = 6
            thrw = int(spectrace - 12)
            pass
        for dl,c in zip(diffloc, np.arange(len(loc))):
            if (dl > thr) and (cvcount <= thrw):
                minlocs.pop(-1)
                minlocs.append(loc[c])
                pass
            if (dl > thr) and (cvcount > thrw):
                maxlocs.append(loc[c - 1])
                minlocs.append(loc[c])
                cvcount = 0
                pass
            if dl < thr: cvcount += 1
            pass
        maxlocs.append(loc[-1])
        mn = minlocs[targetn]
        mx = maxlocs[targetn]
        if (mx - mn) < thrw:
            mn = np.nan
            mx = np.nan
            pass
        if stare:
            mn = minlocs[targetn] - 1
            mx = maxlocs[targetn] + 1
            pass
        pass
    else:
        mn = np.nan
        mx = np.nan
        pass
    if debug:
        show = thisdiff.copy()
        show[show < floodlevel] = np.nan
        plt.figure()
        plt.title('Isolate Flood Level')
        plt.imshow(show)
        plt.colorbar()
        plt.show()
        if np.isfinite(mn):
            plt.figure()
            plt.title('Isolation Level')
            plt.plot(diffloc)
            plt.plot(np.array(diffloc)*0 + thr)
            plt.show()
            pass
        pass
    return mn, mx
# ------------- ------------------------------------------------------
# -- APERTURE AND FILTER TO TOTAL TRANSMISSION FILTER -- -------------
def ag2ttf(flttype):
    '''
    G ROUDIER: Aperture and filter to total transmission filter
    '''
    detector = flttype.split('-')[2]
    grism = flttype.split('-')[3]
    lightpath = ag2lp(detector, grism)
    mu, ttp = bttf(lightpath)
    if grism == 'G141':
        select = (mu > 1.055e4) & (mu < 1.75e4)
        mu = mu[select]
        ttp = ttp[select]
        pass
    if grism == 'G102':
        select = mu < 1.16e4
        mu = mu[select]
        ttp = ttp[select]
        pass
    if grism == 'G430L':
        select = (mu > 2.9e3) & (mu < 5.7394e3)
        mu = mu[select]
        ttp = ttp[select]
        pass
    if grism == 'G750L':
        select = (mu > 5.24e3) & (mu < 1.27e4)
        mu = mu[select]
        ttp = ttp[select]
        pass
    return mu, ttp
# ------------------------------------------------------ -------------
# -- APERTURE AND GRISM TO .FITS FILES -- ----------------------------
def ag2lp(detector, grism):
    '''
    G. ROUDIER: The first element of the returned list defines the default
    interpolation grid, filter/grism file is suited.

    ['Grism source', 'Refractive correction plate', 'Cold mask', 'Mirror 1', 'Mirror 2',
    'Fold mirror', 'Channel select mechanism', 'Pick off mirror', 'OTA']

    http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2011-05.pdf
    ftp://ftp.stsci.edu/cdbs/comp/ota/
    ftp://ftp.stsci.edu/cdbs/comp/wfc3/
    '''
    lightpath = []
    if grism == 'G141': lightpath.append('WFC3/wfc3_ir_g141_src_004_syn.fits')
    if grism == 'G102': lightpath.append('WFC3/wfc3_ir_g102_src_003_syn.fits')
    if grism == 'G430L': lightpath.append('STIS/stis_g430l_009_syn.fits')
    if grism == 'G750L': lightpath.append('STIS/stis_g750l_009_syn.fits')
    if detector == 'IR':
        lightpath.extend(['WFC3/wfc3_ir_rcp_001_syn.fits',
                          'WFC3/wfc3_ir_mask_001_syn.fits',
                          'WFC3/wfc3_ir_mir1_001_syn.fits',
                          'WFC3/wfc3_ir_mir2_001_syn.fits',
                          'WFC3/wfc3_ir_fold_001_syn.fits',
                          'WFC3/wfc3_ir_csm_001_syn.fits',
                          'WFC3/wfc3_pom_001_syn.fits',
                          'WFC3/hst_ota_007_syn.fits'])
        pass
    return lightpath
# --------------------------------------- ----------------------------
# -- BUILD TOTAL TRANSMISSION FILTER -- ------------------------------
def bttf(lightpath, debug=False):
    '''
    G. ROUDIER: Builds total transmission filter
    '''
    ttp = 1e0
    muref = [np.nan]
    if debug: plt.subplot(211)
    for name in lightpath:
        muref, t = loadcalf(name, muref)
        ttp *= t
        if debug: plt.plot(muref, t, 'o--')
        pass
    if debug:
        plt.xlim([min(muref), max(muref)])
        plt.ylabel('Transmission Curves')

        plt.subplot(212)
        plt.plot(muref, ttp, 'o--')
        plt.xlabel('Wavelength [angstroms]')
        plt.ylabel('Total Throughput')
        plt.xlim([min(muref), max(muref)])
        plt.semilogy()
        plt.show()
        pass
    return muref, ttp
# ------------------------------------- ------------------------------
# -- WFC3 CAL FITS -- ------------------------------------------------
def loadcalf(name, muref, calloc=excalibur.context['data_cal']):
    '''
    G. ROUDIER: Loads optical element .fits calibration file
    '''
    # NOTEBOOK RUN
    # calloc = '/proj/sdp/data/cal'
    # ------------
    fitsfile = os.path.join(calloc, name)
    data = pyfits.getdata(fitsfile)
    muin = np.array(data.WAVELENGTH)
    tin = np.array(data.THROUGHPUT)
    if False in np.isfinite(muref): muref = muin
    f = itp.interp1d(muin, tin, bounds_error=False, fill_value=0)
    t = f(muref)
    return muref, t
# ------------------- ------------------------------------------------
# -- WAVELENGTH SOLUTION -- ------------------------------------------
def wavesol(spectrum, tt, wavett, disper, siv=None, fd=False, bck=None, fs=False,
            debug=False, ovszspc=False):
    '''
    G. ROUDIER: Wavelength calibration on log10 spectrum to emphasize the
    edges, approximating the log(stellar spectrum) with a linear model
    '''
    mutt = wavett.copy()
    mutt /= 1e4
    xdata = np.arange(spectrum.size)
    test = tt == 0
    if True in test: tt[test] = np.nan
    logtt = np.log10(tt)
    test = spectrum == 0
    if True in test: spectrum[test] = np.nan
    logspec = np.log10(spectrum)
    wave = xdata*disper/1e4
    select = np.isfinite(spectrum)
    minwave = np.nanmin(wave[select])
    select = np.isfinite(tt)
    reftt = np.nanmin(mutt[select])
    shift = reftt - minwave
    scale = np.nanmedian(logspec) - np.nanmedian(logtt)
    params = lm.Parameters()
    params.add('scale', value=scale)
    if bck is None: params.add('background', value=0e0, vary=False)
    else: params.add('background', value=bck)
    if siv is None: params.add('slope', value=1e-2)
    else: params.add('slope', value=siv, vary=False)
    if fd or ovszspc: params.add('disper', value=disper, vary=False)
    else: params.add('disper', value=disper)
    if fs: params.add('shift', value=shift, vary=False)
    else: params.add('shift', value=shift)
    out = lm.minimize(wcme, params, args=(logspec, mutt, logtt, False))
    disper = out.params['disper'].value
    shift = out.params['shift'].value
    slope = out.params['slope'].value
    scale = out.params['scale'].value
    background = out.params['background'].value
    wave = wcme(out.params, logspec, refmu=mutt, reftt=logtt)
    # PLOTS
    if debug:
        plt.figure()
        plt.plot(mutt, logtt, 'o--', label='Reference')
        plt.plot(wave, logspec - (scale+wave*slope), 'o--', label='Spectrum')
        plt.legend(loc=8)
        plt.show()
        pass
    return wave, disper, shift+minwave, slope, background
# ------------------------- ------------------------------------------
# -- WAVELENGTH FIT FUNCTION -- --------------------------------------
def wcme(params, data, refmu=None, reftt=None, forward=True):
    '''
    G. ROUDIER: Wavelength calibration function for LMfit
    '''
    slope = params['slope'].value
    scale = params['scale'].value
    disper = params['disper'].value
    shift = params['shift'].value
    background = params['background'].value
    liref = itp.interp1d(refmu, reftt, bounds_error=False, fill_value=np.nan)
    wave = np.arange(data.size)*disper*1e-4 + shift
    model = liref(wave) + scale + slope*wave
    select = (np.isfinite(model)) & (np.isfinite(data))
    d = np.log10(10**(data.copy()) - background)
    weights = np.ones(d.size)
    if np.sum(~select) > 0:
        model[~select] = 1e0
        d[~select] = 0e0
        weights[~select] = 1e2
        pass
    if forward: out = wave
    else: out = (d - model)/weights
    return out
# ----------------------------- --------------------------------------
# -- TIME TO Z -- ----------------------------------------------------
def time2z(time, ipct, tknot, sma, orbperiod, ecc, tperi=None, epsilon=1e-10):
    '''
    G. ROUDIER: Time samples in [Days] to separation in [R*]
    '''
    if tperi is not None:
        ft0 = (tperi - tknot) % orbperiod
        ft0 /= orbperiod
        if ft0 > 0.5: ft0 += -1e0
        M0 = 2e0*np.pi*ft0
        E0 = solveme(np.array([M0]), ecc, epsilon)
        realf = np.sqrt(1e0 - ecc)*np.cos(float(E0)/2e0)
        imagf = np.sqrt(1e0 + ecc)*np.sin(float(E0)/2e0)
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
# --------------- ----------------------------------------------------
# -- TRUE ANOMALY NEWTON RAPHSON SOLVER -- ---------------------------
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
# ---------------------------------------- ---------------------------
# -- CALIBRATE STARE DATA -- -----------------------------------------
def starecal(_fin, clc, tim, tid, flttype, out,
             emptythr=1e3, frame2png=False, verbose=False, debug=False):
    '''
    G. ROUDIER: WFC3 STARE Calibration
    '''
    calibrated = False
    # VISIT ----------------------------------------------------------
    for pkey in tim['data'].keys(): visits = np.array(tim['data'][pkey]['visits'])
    # DATA TYPE ------------------------------------------------------
    vrange = validrange(flttype)
    wvrng, disper, ldisp, udisp = fng(flttype)
    spectrace = np.round((np.max(wvrng) - np.min(wvrng))/disper)
    # LOAD DATA ------------------------------------------------------
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')
    data = {'LOC':[], 'EPS':[], 'DISPLIM':[ldisp, udisp],
            'SCANRATE':[], 'SCANLENGTH':[], 'SCANANGLE':[],
            'EXP':[], 'EXPERR':[], 'EXPFLAG':[], 'VRANGE':vrange,
            'TIME':[], 'EXPLEN':[], 'MIN':[], 'MAX':[], 'TRIAL':[]}
    for loc in sorted(clc['LOC']):
        fullloc = os.path.join(dbs, loc)
        with pyfits.open(fullloc) as hdulist:
            header0 = hdulist[0].header
            eps = False
            test = header0['UNITCORR']
            if (test in ['COMPLETE', 'PERFORM']): eps = True
            data['EPS'].append(eps)
            if 'SCAN_RAT' in header0: data['SCANRATE'].append(header0['SCAN_RAT'])
            else: data['SCANRATE'].append(np.nan)
            if 'SCAN_LEN' in header0: data['SCANLENGTH'].append(header0['SCAN_LEN'])
            else: data['SCANLENGTH'].append(np.nan)
            if 'SCAN_ANG' in header0: data['SCANANGLE'].append(header0['SCAN_ANG'])
            elif 'PA_V3' in header0: data['SCANANGLE'].append(header0['PA_V3'])
            else: data['SCANANGLE'].append(666)
            frame = []
            errframe = []
            dqframe = []
            ftime = []
            fmin = []
            fmax = []
            datahdu = [h for h in hdulist if h.size != 0]
            for fits in datahdu:
                if fits.header['EXTNAME'] in ['SCI']:
                    fitsdata = np.empty(fits.data.shape)
                    fitsdata[:] = fits.data[:]
                    if eps and ('DELTATIM' in fits.header):
                        frame.append(fitsdata*float(fits.header['DELTATIM']))
                        fmin.append(float(fits.header['GOODMIN'])*
                                    float(fits.header['DELTATIM']))
                        fmax.append(float(fits.header['GOODMAX'])*
                                    float(fits.header['DELTATIM']))
                        pass
                    elif eps and not'DELTATIM' in fits.header:
                        frame.append(fitsdata*np.nan)
                        fmin.append(np.nan)
                        fmax.append(np.nan)
                        pass
                    else:
                        frame.append(fitsdata)
                        fmin.append(float(fits.header['GOODMIN']))
                        fmax.append(float(fits.header['GOODMAX']))
                        pass
                    ftime.append(float(fits.header['ROUTTIME']))
                    del fits.data
                    pass
                if (fits.header['EXTNAME'] in ['ERR', 'DQ']):
                    fitsdata = np.empty(fits.data.shape)
                    fitsdata[:] = fits.data[:]
                    if fits.header['EXTNAME'] == 'ERR': errframe.append(fitsdata)
                    if fits.header['EXTNAME'] == 'DQ': dqframe.append(fitsdata)
                    del fits.data
                    pass
                pass
            data['LOC'].append(loc)
            data['EXP'].append(frame)
            data['EXPERR'].append(errframe)
            data['EXPFLAG'].append(dqframe)
            data['TIME'].append(ftime)
            data['EXPLEN'].append(header0['EXPTIME'])
            data['MIN'].append(fmin)
            data['MAX'].append(fmax)
            pass
        pass
    # MASK DATA ------------------------------------------------------
    data['MEXP'] = data['EXP'].copy()
    data['MASK'] = data['EXPFLAG'].copy()
    data['IGNORED'] = [False]*len(data['LOC'])
    data['TRUNCSPEC'] = [False]*len(data['LOC'])
    data['FLOODLVL'] = [np.nan]*len(data['LOC'])
    data['TRIAL'] = ['']*len(data['LOC'])
    # FLOOD LEVEL STABILIZATION --------------------------------------
    for index in enumerate(data['LOC']):
        ignore = data['IGNORED'][index[0]]
        # ISOLATE SCAN Y ---------------------------------------------
        sampramp = np.array(data['MEXP'][index[0]]).copy()
        for sutr in sampramp:
            select = ~np.isfinite(sutr)
            if True in select: sutr[select] = 0
            pass
        psdiff = sampramp[0] - sampramp[-1]
        psmin = np.nanmin(data['MIN'][index[0]])
        fldthr = np.nanpercentile(psdiff,
                                  1e2*(1e0 - spectrace/(psdiff.shape[0]*psdiff.shape[1])))
        data['FLOODLVL'][index[0]] = fldthr
        pass
    allfloodlvl = np.array(data['FLOODLVL'])
    for v in set(visits):
        select = visits == v
        allfloodlvl[select] = np.nanmedian(allfloodlvl[select])
        pass
    data['FLOODLVL'] = list(allfloodlvl)
    # DATA CUBE ------------------------------------------------------
    for index, nm in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        # ISOLATE SCAN Y ---------------------------------------------
        sampramp = np.array(data['MEXP'][index]).copy()
        for sutr in sampramp:
            select = ~np.isfinite(sutr)
            if True in select: sutr[select] = 0
            pass
        psdiff = sampramp[0] - sampramp[-1]
        psmin = np.nanmin(data['MIN'][index])
        targetn = 0
        scanwpi = 1
        minlocs = []
        maxlocs = []
        fldthr = data['FLOODLVL'][index]
        for de in [psdiff]:
            lmn, lmx = isolate(de, psmin, spectrace, scanwpi, targetn, fldthr,
                               debug=False, stare=True)
            minlocs.append(lmn)
            maxlocs.append(lmx)
            pass
        ignore = ignore or not((np.any(np.isfinite(minlocs))) and
                               (np.any(np.isfinite(maxlocs))))
        if not ignore:
            minl = np.nanmin(minlocs)
            maxl = np.nanmax(maxlocs)
            minl -= 12
            maxl += 12
            if minl < 0: minl = 0
            else: psdiff[:int(minl), :] = np.nan
            if maxl > (psdiff[0].shape[0] - 1): maxl = psdiff[0].shape[0] - 1
            else: psdiff[int(maxl):, :] = np.nan
            thispstamp = psdiff.copy()
            thispstamp[thispstamp <= psmin] = np.nan
            # PLOTS --------------------------------------------------
            if debug:
                show = thispstamp.copy()
                valid = np.isfinite(show)
                show[~valid] = 0
                show[show < fldthr] = np.nan
                plt.figure()
                plt.title('Isolate Flood Level')
                plt.imshow(show)
                plt.colorbar()

                plt.figure()
                plt.title('Diff Accum')
                plt.imshow(thispstamp)
                plt.colorbar()

                colorlim = np.nanmin(thispstamp)
                plt.figure()
                plt.title('Background')
                plt.imshow(thispstamp)
                plt.colorbar()
                plt.clim(colorlim, abs(colorlim))
                plt.show()
                pass
            # ISOLATE SCAN X -----------------------------------------
            targetn = 0
            minx, maxx = isolate(thispstamp.copy(), psmin, spectrace, scanwpi, targetn,
                                 fldthr, axis=0, stare=True, debug=False)
            if np.isfinite(minx*maxx):
                minx -= 1.5*12
                maxx += 1.5*12
                if minx < 0:
                    minx = 5
                    data['TRUNCSPEC'][index] = True
                    pass
                if maxx > (thispstamp.shape[1] - 1):
                    maxx = thispstamp.shape[1] - 5
                    data['TRUNCSPEC'][index] = True
                    pass
                thispstamp[:, :int(minx)] = np.nan
                thispstamp[:, int(maxx):] = np.nan
                pstamperr = np.sqrt(abs(thispstamp))
                pass
            else:
                garbage = data['EXP'][index][::-1].copy()
                thispstamp = np.sum(np.diff(garbage, axis=0), axis=0)
                pstamperr = thispstamp*np.nan
                data['TRIAL'][index] = 'Could Not Find X Edges'
                ignore = True
                pass
            pass
        else:
            garbage = data['EXP'][index][::-1].copy()
            thispstamp = np.sum(np.diff(garbage, axis=0), axis=0)
            pstamperr = thispstamp*np.nan
            data['TRIAL'][index] = 'Could Not Find Y Edges'
            ignore = True
            pass
        data['MEXP'][index] = thispstamp
        data['TIME'][index] = np.nanmax(data['TIME'][index].copy())
        data['IGNORED'][index] = ignore
        data['EXPERR'][index] = pstamperr
        log.warning('>-- %s / %s', str(index), str(len(data['LOC'])-1))
        # PLOTS ------------------------------------------------------
        if frame2png:
            if not os.path.exists('TEST'): os.mkdir('TEST')
            if not os.path.exists('TEST/'+tid): os.mkdir('TEST/'+tid)
            plt.figure()
            plt.title('Ignored = '+str(ignore))
            plt.imshow(thispstamp)
            plt.colorbar()
            plt.savefig('TEST/'+tid+'/'+nm+'.png')
            plt.close()
            pass
        pass
    # SPECTRUM EXTRACTION --------------------------------------------
    data['SPECTRUM'] = [np.nan]*len(data['LOC'])
    data['SPECERR'] = [np.nan]*len(data['LOC'])
    data['NSPEC'] = [np.nan]*len(data['LOC'])
    for index, loc in enumerate(data['LOC']):
        floodlevel = data['FLOODLVL'][index]
        if floodlevel < emptythr:
            data['IGNORED'][index] = True
            data['TRIAL'][index] = 'Empty Frame'
            pass
        ignore = data['IGNORED'][index]
        if not ignore:
            frame = data['MEXP'][index].copy()
            spectrum = []
            specerr = []
            nspectrum = []
            for col in np.array(frame).T:
                valid = np.isfinite(col)
                if True in valid:
                    spectrum.append(np.nansum(col[valid]))
                    specerr.append(np.sqrt(abs(np.nansum(col[valid]))))
                    nspectrum.append(np.nansum(valid))
                    pass
                else:
                    spectrum.append(np.nan)
                    specerr.append(np.nan)
                    nspectrum.append(0)
                    pass
                pass
            spectrum = np.array(spectrum)
            if np.all(~np.isfinite(spectrum)):
                data['IGNORED'][index] = True
                data['TRIAL'][index] = 'NaN Spectrum'
                pass
            data['SPECTRUM'][index] = np.array(spectrum)
            data['SPECERR'][index] = np.array(specerr)
            data['NSPEC'][index] = np.array(nspectrum)
            pass
        pass
    # WAVELENGTH CALIBRATION -----------------------------------------
    wavett, tt = ag2ttf(flttype)
    scaleco = np.nanmax(tt) / np.nanmin(tt[tt > 0])
    data['PHT2CNT'] = [np.nan]*len(data['LOC'])
    data['WAVE'] = [np.nan]*len(data['LOC'])
    data['DISPERSION'] = [np.nan]*len(data['LOC'])
    data['SHIFT'] = [np.nan]*len(data['LOC'])
    spectralindex = []
    for index, loc in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        ovszspc = False
        if data['TRUNCSPEC'][index]:
            ovszspc = True
            if 'G141' in flttype: wthr = 1.67*1e4
            select = wavett < wthr
            wavett = wavett[select]
            tt = tt[select]
            pass
        if not ignore:
            spectrum = data['SPECTRUM'][index].copy()
            cutoff = np.nanmax(spectrum)/scaleco
            spectrum[spectrum < cutoff] = np.nan
            spectrum = abs(spectrum)
            w, d, s, si, _bck = wavesol(spectrum, tt, wavett, disper, ovszspc=ovszspc,
                                        debug=False)
            if (d > ldisp) and (d < udisp): spectralindex.append(si)
            pass
        pass
    siv = np.nanmedian(spectralindex)
    for index, loc in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        ovszspc = False
        if data['TRUNCSPEC'][index]:
            ovszspc = True
            if 'G141' in flttype: wthr = 1.67*1e4
            select = wavett < wthr
            wavett = wavett[select]
            tt = tt[select]
            pass
        if not ignore:
            spectrum = data['SPECTRUM'][index].copy()
            cutoff = np.nanmax(spectrum)/scaleco
            spectrum[spectrum < cutoff] = np.nan
            spectrum = abs(spectrum)
            wave, disp, shift, si, _bck = wavesol(spectrum, tt, wavett, disper,
                                                  siv=siv, ovszspc=ovszspc)
            liref = itp.interp1d(wavett*1e-4, tt, bounds_error=False, fill_value=np.nan)
            phot2counts = liref(wave)
            data['PHT2CNT'][index] = phot2counts
            data['WAVE'][index] = wave  # MICRONS
            data['DISPERSION'][index] = disp  # ANGSTROMS/PIXEL
            data['SHIFT'][index] = shift*1e4/disp  # PIXELS
            pass
        else: data['WAVE'][index] = (data['SPECTRUM'][index])*np.nan
        data['IGNORED'][index] = ignore
        pass
    # PLOTS ----------------------------------------------------------
    if verbose and (not np.all(data['IGNORED'])):
        alltime = np.array([d for d,i in zip(data['TIME'], data['IGNORED']) if not i])
        dispersion = np.array([d for d,i in zip(data['DISPERSION'], data['IGNORED'])
                               if not i])
        shift = np.array([d for d,i in zip(data['SHIFT'], data['IGNORED']) if not i])
        spec = np.array([d for d,i in zip(data['SPECTRUM'], data['IGNORED']) if not i])
        photoc = np.array([d for d,i in zip(data['PHT2CNT'], data['IGNORED']) if not i])
        wave = np.array([d for d,i in zip(data['WAVE'], data['IGNORED']) if not i])
        errspec = np.array([d for d,i in zip(data['SPECERR'], data['IGNORED']) if not i])
        torder = np.argsort(alltime)
        vrange = data['VRANGE']
        allerr = []
        for s, e, w in zip(spec, errspec, wave):
            select = (w > vrange[0]) & (w < vrange[1])
            allerr.extend(e[select]/np.sqrt(s[select]))
            pass
        allerr = np.array(allerr)
        select = np.isfinite(allerr)
        allerr = allerr[select]
        allerr = allerr[allerr > 0.9]

        plt.figure()
        for spectrum in data['SPECTRUM']: plt.plot(spectrum)
        plt.ylabel('Stellar Spectra [Counts]')
        plt.xlabel('Pixel Number')

        plt.figure()
        for w, p, s in zip(wave, photoc, spec):
            select = (w > vrange[0]) & (w < vrange[1])
            plt.plot(w[select], s[select]/p[select])
            pass
        plt.ylabel('Stellar Spectra [Photons]')
        plt.xlabel('Wavelength [microns]')

        plt.figure()
        plt.hist(allerr)
        plt.xlabel('Error Distribution [Noise Model Units]')

        plt.figure()
        plt.plot(dispersion[torder], 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Dispersion [Angstroms/Pixel]')
        plt.ylim(data['DISPLIM'][0], data['DISPLIM'][1])

        plt.figure()
        plt.plot(shift[torder] - np.nanmin(shift), 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Shift [Pixels]')
        plt.show()
        pass
    allignore = data['IGNORED']
    allculprits = data['TRIAL']
    allindex = np.arange(len(data['LOC']))
    log.warning('>-- IGNORED: %s / %s', str(np.nansum(allignore)), str(len(allignore)))
    for index in allindex:
        if allculprits[index].__len__() > 0:
            log.warning('>-- %s: %s', str(index), str(allculprits[index]))
            pass
        pass
    data.pop('EXP', None)
    data.pop('EXPFLAG', None)
    for key in data: out['data'][key] = data[key]
    calibrated = not np.all(data['IGNORED'])
    if calibrated: out['STATUS'].append(True)
    return calibrated
# -------------------------- -----------------------------------------
# -- STIS CALIBRATION -- ---------------------------------------------
def stiscal_G750L(_fin, clc, tim, tid, flttype, out,
                  verbose=False, debug=False):
    '''
    R. ESTRELA: STIS .flt data extraction and wavelength calibration
    '''
    calibrated = False
    # VISIT NUMBERING --------------------------------------------------------------------
    for pkey in tim['data'].keys(): visits = np.array(tim['data'][pkey]['dvisits'])
    # PHASE ------------------------------------------------------------------------------
    # for pkey in tim['data'].keys(): phase = np.array(tim['data'][pkey]['phase'])
    # OPTICS AND FILTER ------------------------------------------------------------------
    vrange = validrange(flttype)
    _wvrng, disp, ldisp, udisp = fng(flttype)
    # DATA FORMAT ------------------------------------------------------------------------
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')
    data = {'LOC':[], 'EPS':[], 'DISPLIM':[ldisp, udisp],
            'SCANRATE':[], 'SCANLENGTH':[], 'SCANANGLE':[],
            'EXP':[], 'EXPERR':[], 'EXPFLAG':[], 'VRANGE':vrange,
            'TIME':[], 'EXPLEN':[], 'MIN':[], 'MAX':[], 'TRIAL':[], 'TIMEOBS':[], 'DATEOBS':[]}
    # LOAD DATA --------------------------------------------------------------------------
    for loc in sorted(clc['LOC']):
        fullloc = os.path.join(dbs, loc)
        with pyfits.open(fullloc) as hdulist:
            header0 = hdulist[0].header
            if 'SCAN_ANG' in header0: scanangle = header0['SCAN_ANG']
            elif 'PA_V3' in header0: scanangle = header0['PA_V3']
            else: scanangle = 666
            allloc = []
            alltime = []
            allexplen = []
            alleps = []
            allexp = []
            allexperr = []
            allmask = []
            allmin = []
            allmax = []
            alldate = []
            alltimeobs = []
            for fits in hdulist:
                if (fits.size != 0) and (fits.header['EXTNAME']=='SCI'):
                    allloc.append(fits.header['EXPNAME'])
                    alltime.append(float(fits.header['EXPEND']))
                    allexplen.append(float(fits.header['EXPTIME']))
                    alldate.append(header0['TDATEOBS'])
                    alltimeobs.append(header0['TTIMEOBS'])
                    fitsdata = np.empty(fits.data.shape)
                    fitsdata[:] = fits.data[:]
                    test = fits.header['BUNIT']
                    eps = False
                    if test != 'COUNTS': eps = True
                    alleps.append(eps)
                    allmin.append(float(fits.header['GOODMIN']))
                    allmax.append(float(fits.header['GOODMAX']))
                    # BINARIES
                    # GMR: Let's put that in the mask someday
                    nam = 'MIDPOINT'
                    if tid in ['HAT-P-1'] and nam in header0['TARGNAME']: allexp.append(fitsdata[200:380, :])
                    else: allexp.append(fitsdata)
                    del fits.data
                    pass
                if 'EXTNAME' in fits.header:
                    if (fits.header['EXTNAME'] in ['ERR', 'DQ']):
                        fitsdata = np.empty(fits.data.shape)
                        fitsdata[:] = fits.data[:]
                        if fits.header['EXTNAME'] == 'ERR': allexperr.append(fitsdata)
                        if fits.header['EXTNAME'] == 'DQ': allmask.append(fitsdata)
                        del fits.data
                        pass
                    if eps:
                        eps2count = allexplen[-1]*float(header0['CCDGAIN'])
                        allexp[-1] = allexp[-1]*eps2count
                        allexperr[-1] = allexperr[-1]*eps2count
                        pass
                    pass
                pass
            allscanangle = [scanangle]*len(allloc)
            allscanlength = [1e0]*len(allloc)
            allscanrate = [0e0]*len(allloc)
            data['LOC'].extend(allloc)
            data['EPS'].extend(alleps)
            data['SCANRATE'].extend(allscanrate)
            data['SCANLENGTH'].extend(allscanlength)
            data['SCANANGLE'].extend(allscanangle)
            data['EXP'].extend(allexp)
            data['EXPERR'].extend(allexperr)
            data['TIME'].extend(alltime)
            data['EXPLEN'].extend(allexplen)
            data['MIN'].extend(allmin)
            data['MAX'].extend(allmax)
            data['TIMEOBS'].extend(alltimeobs)
            data['DATEOBS'].extend(alldate)
            pass
        pass
    data['MEXP'] = data['EXP'].copy()
    data['MASK'] = data['EXPFLAG'].copy()
    data['ALLDATEOBS'] = data['DATEOBS'].copy()
    data['ALLTIMEOBS'] = data['TIMEOBS'].copy()
    data['IGNORED'] = np.array([False]*len(data['LOC']))
    data['FLOODLVL'] = [np.nan]*len(data['LOC'])
    data['TRIAL'] = ['']*len(data['LOC'])
    data['SPECTRUM'] = [np.array([np.nan])]*len(data['LOC'])
    data['SPECERR'] = [np.array([np.nan])]*len(data['LOC'])
    data['TEMPLATE'] = [np.array([np.nan])]*len(data['LOC'])
    data['NSPEC'] = [1e0]*len(data['LOC'])
    # REJECT OUTLIERS IN EXPOSURE LENGTH -------------------------------------------------
    for v in set(visits):
        select = visits == v
        visitexplength = np.array(data['EXPLEN'])[select]
        visitignore = data['IGNORED'][select]
        ref = np.nanmedian(visitexplength)
        visitignore[visitexplength != ref] = True
        data['IGNORED'][select] = visitignore
        pass
    for index, ignore in enumerate(data['IGNORED']):
        # SELECT DATE AND TIME OF THE EXPOSURE FOR FLAT FRINGE SELECTION
        frame = data['MEXP'][index].copy()
        dateobs_exp = data['ALLDATEOBS'][index]
        timeobs_exp = data['ALLTIMEOBS'][index]
        tog_exp = dateobs_exp +' '+ timeobs_exp
        time_exp = raissatime.mktime(datetime.datetime.strptime(tog_exp, "%Y-%m-%d %H:%M:%S").timetuple())
        # LOAD FRINGE FLAT -------------------------------------------------------------------
        obs_name = clc['ROOTNAME'][0]
        name_sel = obs_name[:-5]
        lightpath_fringe = ('STIS/CCDFLAT/')
        calloc = excalibur.context['data_cal']
        filefringe = os.path.join(calloc,lightpath_fringe)
        if tid in ['HD 209458']:
            lightpath_fringe = ('STIS/CCDFLAT/h230851ao_pfl.fits')
            calloc = excalibur.context['data_cal']
            filefringe = os.path.join(calloc,lightpath_fringe)
            hdu = pyfits.open(filefringe)
            data_fringe = hdu[1].data
            err_fringe = hdu[2].data
            pass
        else:
            diff_list = []
            all_infile = []
            for infile in glob.glob('%s/%s*_flt.fits' % (filefringe,name_sel)):
                hdu = pyfits.open(infile)
                all_infile.append(infile)
                header_flat = hdu[0].header
                date_time=header_flat['TDATEOBS']
                hour_time=header_flat['TTIMEOBS']
                tog = date_time +' '+ hour_time
                time_flat_s = raissatime.mktime(datetime.datetime.strptime(tog, "%Y-%m-%d %H:%M:%S").timetuple())
                diff = abs(time_exp-time_flat_s)
                diff_list.append(diff)
                pass
            cond_win = np.where(diff_list == np.min(diff_list))
            all_infile = np.array(all_infile)
            sel_flatfile = all_infile[cond_win][0]
            hdulist = pyfits.open(sel_flatfile)
            data_fringe = hdulist[4].data
            err_fringe = hdulist[5].data
            pass
        smooth_fringe = scipy.signal.medfilt(data_fringe, 7)
        sigma_fringe = np.median(err_fringe)
        bad_fringe = (np.abs(data_fringe - smooth_fringe) / sigma_fringe) > 2
        img_fringe = data_fringe.copy()
        img_fringe[bad_fringe] = smooth_fringe[bad_fringe]
        if debug:
            plt.figure()
            for i in range(0,len(data_fringe)):
                plt.plot(img_fringe[i,:])
                pass
            pass
        cont_data = img_fringe.copy()
        div_list=[]
        for i in range(508,515):
            pixels = np.arange(0,1024,1)
            coefs = poly.polyfit(pixels,cont_data[i,:], 11)
            ffit = poly.polyval(pixels, coefs)
            div = cont_data[i,:]/ffit
            div_list.append(div)
            if debug:
                plt.figure()
                plt.plot(pixels, ffit,color='red')
                plt.plot(cont_data[i,:],color='blue')
                plt.xlabel('Pixels')
                plt.title('Contemporaneous Flat Fringe - Polynomial fit')
                pass
            pass
        #############
        # COSMIC RAY REJECTION IN THE 2D IMAGE
        img_cr = frame.copy()
        allframe_list = []
        for i in range(0,len(frame)):
            img_sm = scipy.signal.medfilt(img_cr[i,:], 9)
            # std = np.std(img_cr[i,:] - img_sm)
            std = np.std(img_sm)
            bad = np.abs(img_cr[i,:] - img_sm) > 3*std
            line = img_cr[i,:]
            line[bad] = img_sm[bad]
            allframe_list.append(line)
            pass
        allframe = np.array(allframe_list)
        # APPLY FLAT FRINGE
        # plt.figure()
        if not ignore:
            find_spec = np.where(allframe == np.max(allframe))
            spec_idx = find_spec[0][0]
            spec_idx_up = spec_idx+4
            spec_idx_dwn = spec_idx-3
            spec_idx_all = np.arange(spec_idx_dwn,spec_idx_up,1)
            frame2 = allframe.copy()
            for i,flatnorm in zip(spec_idx_all,div_list):
                frame_sel = allframe[i,:]
                coefs_f = poly.polyfit(pixels,frame_sel, 12)
                ffit_f = poly.polyval(pixels, coefs_f)
                frame2[i,400:1023] = frame2[i,400:1023]/flatnorm[400:1023]
                if debug:
                    plt.subplot(2, 1, 1)
                    plt.plot(pixels,frame_sel,color='blue')
                    plt.plot(pixels, ffit_f,color='red')
                    plt.subplot(2, 1, 2)
                    norm = frame_sel/ffit_f
                    plt.plot(norm, color='orange',label='Observed spectrum')
                    plt.plot(flatnorm,color='blue',label='Contemporaneous Flat fringe')
                    plt.xlabel('pixels')
                    plt.ylabel('Normalized flux')
                    plt.legend(loc='lower right', shadow=False, frameon=False, fontsize='7', scatterpoints=1)
                    pass
                pass
                data['SPECTRUM'][index] = np.nansum(frame2, axis=0)
                data['SPECERR'][index] = np.sqrt(np.nansum(frame2, axis=0))
                pass
        else:
            data['SPECTRUM'][index] = np.nansum(frame, axis=0)*np.nan
            data['SPECERR'][index] = np.nansum(frame, axis=0)*np.nan
            data['TRIAL'][index] = 'Exposure Length Outlier'
            pass
        pass
    ####
    # MASK BAD PIXELS IN SPECTRUM --------------------------------------------------------
    for v in set(visits):
        select = (visits == v) & ~(data['IGNORED'])
        # for index, valid in enumerate(select):
        #    spec_ind = data['SPECTRUM'][index]
        #    if valid and 'G750' in flttype:
        #            b, a = signal.butter(3., 0.05)
        #            zi = signal.lfilter_zi(b, a)
        #            spec0 = spec_ind[500:1024]
        #            spec_ind[500:1024], _ = signal.lfilter(b, a, spec_ind[500:1024], zi=zi*spec0[0])
        specarray = np.array([s for s, ok in zip(data['SPECTRUM'], select) if ok])
        trans = np.transpose(specarray)
        template = np.nanmedian(trans, axis=1)
        # TEMPLATE MEDIAN 5 POINTS LOW PASS FILTER ---------------------------------------
        smootht = []
        smootht.extend([template[0]]*2)
        smootht.extend(template)
        smootht.extend([template[-1]]*2)
        for index in np.arange(len(template)):
            medianvalue = np.nanmedian(template[index:index+5])
            smootht[2+index] = medianvalue
            pass
        smootht = smootht[2:-2]
        if debug:
            plt.figure()
            plt.plot(template, 'o')
            plt.plot(smootht)
            plt.show()
            pass
        template = np.array(smootht)
        for vindex, valid in enumerate(select):
            if valid: data['TEMPLATE'][vindex] = template
            pass
        pass
    wavett, tt = ag2ttf(flttype)
    # COSMIC RAYS REJECTION --------------------------------------------------------------
    data['PHT2CNT'] = [np.nan]*len(data['LOC'])
    data['WAVE'] = [np.array([np.nan])]*len(data['LOC'])
    data['DISPERSION'] = [np.nan]*len(data['LOC'])
    data['SHIFT'] = [np.nan]*len(data['LOC'])
    allsi = []
    for index, rejected in enumerate(data['IGNORED']):
        if not rejected:
            template = data['TEMPLATE'][index]
            spec = data['SPECTRUM'][index]
            temp_spec = spec/template
            ht25 = np.nanpercentile(temp_spec,25)
            lt75 = np.nanpercentile(temp_spec,75)
            std = np.std(temp_spec[(temp_spec > ht25) & (temp_spec < lt75)])
            # BAD PIXEL THRESHOLD --------------------------------------------------------
            bpthr = temp_spec > np.nanmedian(temp_spec) + 3e0*std
            if True in bpthr: spec[bpthr] = np.nan
            # if 'G430' in flttype:
            #    cond_max = np.where(spec == np.nanmax(spec))
            #    spec[cond_max] = np.nan
            #    data['SPECTRUM'][index] = spec
            # else:
            #    data['SPECTRUM'][index] = spec

            # FIRST WAVESOL --------------------------------------------------------------
            # scaleco = np.nanmax(tt) / np.nanmin(tt[tt > 0])
            scaleco = 1e1
            if np.sum(np.isfinite(spec)) > (spec.size/2) and 'G750' in flttype:
                wavecalspec = spec.copy()
                finitespec = np.isfinite(spec)
                nanme = spec[finitespec] < (np.nanmax(wavecalspec)/scaleco)
                if True in nanme:
                    finwavecalspec = wavecalspec[finitespec]
                    finwavecalspec[nanme] = np.nan
                    wavecalspec[finitespec] = finwavecalspec
                    pass
                selref = tt > (np.nanmax(tt)/scaleco)
                # dispersion is fixed, shift is not
                w, d, s, si, _bck = wavesol(abs(wavecalspec), tt[selref], wavett[selref], disp,
                                            fd=True, fs=False, debug=False)
                data['WAVE'][index] = w
                allsi.append(si)
                pass
            else:
                data['IGNORED'][index] = True
                data['TRIAL'][index] = 'Not Enough Valid Points In Extracted Spectrum'
                pass
            if 'G430' in flttype:
                pixel = np.arange(len(spec))
                w = (pixel + 1079.96)/0.37
                data['WAVE'][index] = w*1e-4
                liref = itp.interp1d(wavett*1e-4, tt, bounds_error=False, fill_value=np.nan)
                phot2counts = liref(w*1e-4)
                data['PHT2CNT'][index] = phot2counts
                data['DISPERSION'][index] = 2.70
                data['SHIFT'][index] = -1079.96
                pass
            pass
        pass
    # Plot first wavelength calibration
    if debug:
        for v in set(visits):
            select = (visits == v) & ~(data['IGNORED'])
            plt.figure()
            for index, valid in enumerate(select):
                if valid: plt.plot(data['WAVE'][index], data['SPECTRUM'][index], 'o')
                pass
            plt.xlabel('Wavelength [microns]')
            plt.ylabel('Counts')
            plt.title('Visit number: '+str(int(v)))
            pass
        plt.show()
        pass
    # SECOND Wavesol
    # SECOND SIGMA CLIPPING
    for index, rejected in enumerate(data['IGNORED']):
        if not rejected:
            # template = data['TEMPLATE'][index]
            # spec = data['SPECTRUM'][index]
            # temp_spec = spec/template
            # ht25 = np.nanpercentile(temp_spec,25)
            # lt75 = np.nanpercentile(temp_spec,75)
            # selfin = np.isfinite(temp_spec)
            # std1 = np.nanstd(temp_spec[selfin][(temp_spec[selfin] > ht25) & (temp_spec[selfin] < lt75)])
            # BAD PIXEL THRESHOLD --------------------------------------------------------
            # bpthr = temp_spec[selfin] > np.nanmedian(temp_spec) + 2e0*std1
            # temp_cut=template.copy()
            # spec_cut = spec.copy()
            # data['SPECTRUM'][index] = spec_cut

            # SECOND WAVESOL --------------------------------------------------------------
            # scaleco = np.nanmax(tt) / np.nanmin(tt[tt > 0])
            scaleco = 1e1

            if np.sum(np.isfinite(spec)) > (spec.size/2) and 'G750' in flttype:
                wavecalspec = spec.copy()
                selfinspec = np.isfinite(spec)
                nanme = spec[selfinspec] < (np.nanmax(wavecalspec)/scaleco)
                if True in nanme:
                    wavefin = wavecalspec[selfinspec]
                    wavefin[nanme] = np.nan
                    wavecalspec[selfinspec] = wavefin
                    pass
                selref = tt > (np.nanmax(tt)/scaleco)
                # dispersion is fixed, shift is not
                w, d, s, si, _bck = wavesol(abs(wavecalspec), tt[selref], wavett[selref], disp,
                                            siv=np.nanmedian(allsi), fd=True, fs=False, debug=False)
                liref = itp.interp1d(wavett*1e-4, tt,
                                     bounds_error=False, fill_value=np.nan)
                phot2counts = liref(w)
                data['PHT2CNT'][index] = phot2counts
                data['WAVE'][index] = w
                data['DISPERSION'][index] = d
                data['SHIFT'][index] = s
                pass
            # else:
                # data['IGNORED'][index] = True
                # data['TRIAL'][index] = 'Not Enough Valid Points In Extracted Spectrum'
                pass

            # WAVELENGTH CALIBRATION - G430L
            if 'G430' in flttype:
                pixel = np.arange(len(spec))
                w = (pixel + 1079.96)/0.37
                data['WAVE'][index] = w*1e-4
                liref = itp.interp1d(wavett*1e-4, tt, bounds_error=False, fill_value=np.nan)
                phot2counts = liref(w*1e-4)
                data['PHT2CNT'][index] = phot2counts
                data['DISPERSION'][index] = 2.70
                data['SHIFT'][index] = -1079.96
            pass
        pass
    # PLOTS ------------------------------------------------------------------------------
    if verbose and (not np.all(data['IGNORED'])):
        alltime = np.array([d for d,i in zip(data['TIME'], data['IGNORED']) if not i])
        dispersion = np.array([d for d,i in zip(data['DISPERSION'], data['IGNORED'])
                               if not i])
        shift = np.array([d for d,i in zip(data['SHIFT'], data['IGNORED']) if not i])
        spec = np.array([d for d,i in zip(data['SPECTRUM'], data['IGNORED']) if not i])
        photoc = np.array([d for d,i in zip(data['PHT2CNT'], data['IGNORED']) if not i])
        wave = np.array([d for d,i in zip(data['WAVE'], data['IGNORED']) if not i])
        errspec = np.array([d for d,i in zip(data['SPECERR'], data['IGNORED']) if not i])
        torder = np.argsort(alltime)
        vrange = data['VRANGE']
        allerr = []
        for s, e, w in zip(spec, errspec, wave):
            select = (w > vrange[0]) & (w < vrange[1])
            allerr.extend(e[select]/np.sqrt(s[select]))
            pass
        allerr = np.array(allerr)
        select = np.isfinite(allerr)
        allerr = allerr[select]
        allerr = allerr[allerr > 0.9]

        plt.figure()
        for spectrum in data['SPECTRUM']: plt.plot(spectrum)
        plt.ylabel('Stellar Spectra [Counts]')
        plt.xlabel('Pixel Number')

        plt.figure()
        for w, p, s in zip(wave, photoc, spec):
            select = (w > vrange[0]) & (w < vrange[1])
            plt.plot(w[select], s[select]/p[select])
            pass
        plt.ylabel('Stellar Spectra [Photons]')
        plt.xlabel('Wavelength [microns]')

        plt.figure()
        plt.hist(allerr)
        plt.xlabel('Error Distribution [Noise Model Units]')

        plt.figure()
        plt.plot(dispersion[torder], 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Dispersion [Angstroms/Pixel]')
        plt.ylim(data['DISPLIM'][0], data['DISPLIM'][1])

        plt.figure()
        plt.plot(shift[torder] - np.nanmin(shift), 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Shift [Pixels]')
        plt.show()
        pass
    allignore = data['IGNORED']
    allculprits = data['TRIAL']
    log.warning('>-- IGNORED: %s / %s', str(np.nansum(allignore)), str(len(allignore)))
    for index, ignore in enumerate(allignore):
        if ignore: log.warning('>-- %s: %s', str(index), str(allculprits[index]))
        pass
    data.pop('EXP', None)
    for key in data: out['data'][key] = data[key]
    calibrated = not np.all(data['IGNORED'])
    if calibrated: out['STATUS'].append(True)
    return calibrated
# ---------------------- ---------------------------------------------
# -------------------------- -----------------------------------------
# -- STIS CALIBRATION -- ---------------------------------------------
def stiscal_G430L(fin, clc, tim, tid, flttype, out,
                  verbose=False, debug=False):
    '''
    R. ESTRELA: STIS .flt data extraction and wavelength calibration
    '''
    calibrated = False
    # VISIT NUMBERING --------------------------------------------------------------------
    for pkey in tim['data'].keys(): visits = np.array(tim['data'][pkey]['dvisits'])
    # PHASE ------------------------------------------------------------------------------
    for pkey in tim['data'].keys(): phase = np.array(tim['data'][pkey]['phase'])
    # OPTICS AND FILTER ------------------------------------------------------------------
    vrange = validrange(flttype)
    _wvrng, _disp, ldisp, udisp = fng(flttype)
    # DATA FORMAT ------------------------------------------------------------------------
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')
    data = {'LOC':[], 'EPS':[], 'DISPLIM':[ldisp, udisp],
            'SCANRATE':[], 'SCANLENGTH':[], 'SCANANGLE':[],
            'EXP':[], 'EXPERR':[], 'EXPFLAG':[], 'VRANGE':vrange,
            'TIME':[], 'EXPLEN':[], 'MIN':[], 'MAX':[], 'TRIAL':[]}
    # LOAD DATA --------------------------------------------------------------------------
    for loc in sorted(clc['LOC']):
        fullloc = os.path.join(dbs, loc)
        with pyfits.open(fullloc) as hdulist:
            header0 = hdulist[0].header
            if 'SCAN_ANG' in header0: scanangle = header0['SCAN_ANG']
            elif 'PA_V3' in header0: scanangle = header0['PA_V3']
            else: scanangle = 666
            allloc = []
            alltime = []
            allexplen = []
            alleps = []
            allexp = []
            allexperr = []
            allmask = []
            allmin = []
            allmax = []
            for fits in hdulist:
                if (fits.size != 0) and (fits.header['EXTNAME']=='SCI'):
                    allloc.append(fits.header['EXPNAME'])
                    alltime.append(float(fits.header['EXPEND']))
                    allexplen.append(float(fits.header['EXPTIME']))
                    fitsdata = np.empty(fits.data.shape)
                    fitsdata[:] = fits.data[:]
                    test = fits.header['BUNIT']
                    eps = False
                    if test != 'COUNTS': eps = True
                    alleps.append(eps)
                    allmin.append(float(fits.header['GOODMIN']))
                    allmax.append(float(fits.header['GOODMAX']))
                    # BINARIES
                    # GMR: Let's put that in the mask someday
                    if tid in ['HAT-P-1']: allexp.append(fitsdata[0:120, :])
                    else: allexp.append(fitsdata)
                    del fits.data
                    pass
                if 'EXTNAME' in fits.header:
                    if (fits.header['EXTNAME'] in ['ERR', 'DQ']):
                        fitsdata = np.empty(fits.data.shape)
                        fitsdata[:] = fits.data[:]
                        if fits.header['EXTNAME'] == 'ERR': allexperr.append(fitsdata)
                        if fits.header['EXTNAME'] == 'DQ': allmask.append(fitsdata)
                        del fits.data
                        pass
                    if eps:
                        eps2count = allexplen[-1]*float(header0['CCDGAIN'])
                        allexp[-1] = allexp[-1]*eps2count
                        allexperr[-1] = allexperr[-1]*eps2count
                        pass
                    pass
                pass
            allscanangle = [scanangle]*len(allloc)
            allscanlength = [1e0]*len(allloc)
            allscanrate = [0e0]*len(allloc)
            data['LOC'].extend(allloc)
            data['EPS'].extend(alleps)
            data['SCANRATE'].extend(allscanrate)
            data['SCANLENGTH'].extend(allscanlength)
            data['SCANANGLE'].extend(allscanangle)
            data['EXP'].extend(allexp)
            data['EXPERR'].extend(allexperr)
            data['TIME'].extend(alltime)
            data['EXPLEN'].extend(allexplen)
            data['MIN'].extend(allmin)
            data['MAX'].extend(allmax)
            pass
        pass
    data['MEXP'] = data['EXP'].copy()
    data['MASK'] = data['EXPFLAG'].copy()
    data['IGNORED'] = np.array([False]*len(data['LOC']))
    data['FLOODLVL'] = [np.nan]*len(data['LOC'])
    data['TRIAL'] = ['']*len(data['LOC'])
    data['SPECTRUM0'] = [np.array([np.nan])]*len(data['LOC'])
    data['SPECTRUM'] = [np.array([np.nan])]*len(data['LOC'])
    data['SPECTRUM_CLEAN'] = [np.array([np.nan])]*len(data['LOC'])
    data['SPECERR'] = [np.array([np.nan])]*len(data['LOC'])
    data['TEMPLATE'] = [np.array([np.nan])]*len(data['LOC'])
    data['PHT2CNT'] = [np.array([np.nan])]*len(data['LOC'])
    data['NSPEC'] = [1e0]*len(data['LOC'])
    # REJECT OUTLIERS IN EXPOSURE LENGTH -------------------------------------------------
    for v in set(visits):
        select = visits == v
        visitexplength = np.array(data['EXPLEN'])[select]
        visitignore = data['IGNORED'][select]
        ref = np.nanmedian(visitexplength)
        visitignore[visitexplength != ref] = True
        data['IGNORED'][select] = visitignore
        pass
    # COSMIC RAYS REJECTION - MEDIAN FILTER + SIGMA CLIPPING
    for index, ignore in enumerate(data['IGNORED']):
        # COSMIC RAY REJECTION IN THE 2D IMAGE
        frame = data['MEXP'][index].copy()
        img_cr = frame.copy()
        allframe_list = []
        for i in range(0,len(frame)):
            img_sm = scipy.signal.medfilt(img_cr[i,:], 9)
            std = np.std(img_cr[i,:] - img_sm)
            # std = np.std(img_sm)
            bad = np.abs(img_cr[i,:] - img_sm) > 2*std
            line = img_cr[i,:]
            line[bad] = img_sm[bad]
            img_sm2 = scipy.signal.medfilt(line, 9)
            std2 = np.std(line - img_sm2)
            bad2 = np.abs(line - img_sm2) > 2*std2
            line2 = line.copy()
            line2[bad2] = img_sm2[bad2]
            allframe_list.append(line2)
            pass
        allframe = np.array(allframe_list)
        if not ignore:
            data['SPECTRUM'][index] = np.nansum(allframe, axis=0)
            data['SPECERR'][index] = np.sqrt(np.nansum(allframe, axis=0))
            data['PHT2CNT'][index] = [np.nan]*len(frame[0])
            pass
        else:
            data['SPECTRUM'][index] = np.nansum(allframe, axis=0)*np.nan
            data['SPECERR'][index] = np.nansum(allframe, axis=0)*np.nan
            data['TRIAL'][index] = 'Exposure Length Outlier'
            data['PHT2CNT'][index] = [np.nan]*len(frame[0])
            pass
        pass
    # if debug:
    #    for v in set(visits):
    #        select = (visits == v) & ~(data['IGNORED'])
    #        plt.figure()
    #        for index, valid in enumerate(select):
    #            if valid: plt.plot(data['SPECTRUM'][index], 'o')
    #            pass
    #            plt.xlabel('Wavelength [microns]')
    #            plt.ylabel('Counts')
    #            plt.title('Visit number: '+str(int(v)))
    #            pass
    #        plt.show()
    pass
    wavett, tt = ag2ttf(flttype)
    if 'G430' in flttype:
        select = wavett > 0.29e4
        wavett = wavett[select]
        tt = tt[select]
        pass
    if verbose and (not np.all(data['IGNORED'])):
        plt.figure()
        for spectrum in data['SPECTRUM']: plt.plot(spectrum)
        plt.ylabel('Stellar Spectra [Counts]')
        plt.xlabel('Pixel Number')
        plt.show()
        # MASK BAD PIXELS IN SPECTRUM --------------------------------------------------------
    for v in set(visits):
        select = (visits == v) & ~(data['IGNORED'])
        specarray = np.array([s for s, ok in zip(data['SPECTRUM'], select) if ok])
        trans = np.transpose(specarray)
        template = np.nanmedian(trans, axis=1)
        # TEMPLATE MEDIAN 5 POINTS LOW PASS FILTER ---------------------------------------
        smootht = []
        smootht.extend([template[0]]*2)
        smootht.extend(template)
        smootht.extend([template[-1]]*2)
        for index in np.arange(len(template)):
            medianvalue = np.nanmedian(template[index:index+5])
            smootht[2+index] = medianvalue
            pass
        smootht = smootht[2:-2]
        template = np.array(smootht)
        for vindex, valid in enumerate(select):
            if valid: data['TEMPLATE'][vindex] = template
            pass
        pass
    # COSMIC RAYS REJECTION
    # data['PHT2CNT'] = [np.nan]*len(data['LOC'])
    data['WAVE'] = [np.array([np.nan])]*len(data['LOC'])
    data['DISPERSION'] = [np.nan]*len(data['LOC'])
    data['SHIFT'] = [np.nan]*len(data['LOC'])

    set_wav = np.array([290,570])

    def phoenix(set_wav):
        # PHOENIX MODELS
        filters = [BoxcarFilter('a', 300, 550)]  # Define your passbands
        feherr=np.sqrt(abs(fin['priors']['FEH*_uperr']*fin['priors']['FEH*_lowerr']))
        loggerr = np.sqrt(abs(fin['priors']['LOGG*_uperr']*
                              fin['priors']['LOGG*_lowerr']))
        terr = np.sqrt(abs(fin['priors']['T*_uperr']*fin['priors']['T*_lowerr']))
        sc = LDPSetCreator(teff=(fin['priors']['T*'], terr),
                           logg=(fin['priors']['b']['logg'], loggerr),
                           z=(fin['priors']['FEH*'], feherr),
                           filters=filters)
        list_diff = []
        for i in range(0,len(sc.files)):
            hdul = pyfits.open(sc.files[i])
            teff = hdul[0].header['PHXTEFF']
            zz = hdul[0].header['PHXM_H']
            logg_str = hdul[0].header['PHXLOGG']
            diff1 = abs(fin['priors']['T*'] - teff)
            diff2 = abs(fin['priors']['FEH*'] - zz)
            diff3 = abs(fin['priors']['b']['logg'] - logg_str)
            diff_total = diff1 + diff2 + diff3
            list_diff.append(diff_total)
        cond_win = np.where(list_diff == np.min(list_diff))
        hdul2 = pyfits.open(sc.files[cond_win[0][0]])  # 1 for HAT-p-26 and Hat-P-11, 3 for Hat-p-18, 1 for WASP-52, 4 for WASP-80
        data_all = hdul2[0].data
        wl0 = hdul[0].header['crval1']*1e-1  # defines the wavelength at pixel CRPIX1
        dwl = hdul[0].header['cdelt1']*1e-1  # Delta wavelength     [nm]
        nwl = hdul[0].header['naxis1']  # Number of wl samples
        wl = wl0 + np.arange(nwl)*dwl
        model = data_all[77]  # take only the last spectra of each fits
        # Average the spectra to get 1 spectrum model
        # new_spec=[]
        # trans_listdata = np.transpose(list_models)
        # for i in range(0,len(trans_listdata)):
        #    med_wav = np.mean(trans_listdata[i])
        #    new_spec.append(med_wav)
        # f_spec = itp.interp1d(wl, new_spec, bounds_error=False)
        # spec_sel = f_spec(wavett*0.1)
        # spec_sel_norm = spec_sel/np.max(spec_sel)
        cond_wav = np.where((wl > set_wav[0]) & (wl < set_wav[1]))
        wl_sel = wl[cond_wav]
        new_spec = np.array(model)
        spec_sm = scipy.signal.medfilt(new_spec, 9)
        new_spec_sel = spec_sm[cond_wav]
        mid, low, high, binsz = binnagem(wl_sel,1024)
        # func_spec = scipy.interpolate.interp1d(wl_sel,new_spec_sel)
        # BINNING PHOENIX MODEL
        bin_spec=[]
        for w_low, w_hi in zip(low, high):
            select = np.where((wl_sel > w_low) & (wl_sel < w_hi))
            # inte = scipy.integrate.quad(lambda x: func_spec(x), w_low, w_hi)
            inte = np.sum(new_spec_sel[select]*(wl_sel[select[0]+1]-wl_sel[select[0]]))
            databin=inte/binsz[0]  # inte[0] if scipy.integrate
            bin_spec.append(databin)
        bin_spec=np.array(bin_spec)
        bin_spec = scipy.signal.medfilt(bin_spec, 5)
        return mid, bin_spec

    def chisqfunc(args):
        avar, bvar= args
        chisq = np.sum(((g_wav*bin_spec_norm[cond_mid]) - f(bvar+(avar)*mid_ang[cond_mid]))**2)
        return chisq

    dispersion_list = []
    for index, rejected in enumerate(data['IGNORED']):
        if not rejected:
            spec = data['SPECTRUM'][index]
            template = data['TEMPLATE'][index]
            temp_spec = spec/template
            ht25 = np.nanpercentile(temp_spec,25)
            lt75 = np.nanpercentile(temp_spec,75)
            std = np.std(temp_spec[(temp_spec > ht25) & (temp_spec < lt75)])
            # BAD PIXEL THRESHOLD --------------------------------------------------------
            bpthr = temp_spec > np.nanmedian(temp_spec) + 2e0*std
            if True in bpthr: spec[bpthr] = np.nan
            # second
            temp_spec2 = spec/template
            ht25 = np.nanpercentile(temp_spec2,25)
            lt75 = np.nanpercentile(temp_spec2,75)
            selfin = np.isfinite(temp_spec2)
            std1 = np.nanstd(temp_spec2[selfin][(temp_spec2[selfin] > ht25) & (temp_spec2[selfin] < lt75)])
            # BAD PIXEL THRESHOLD --------------------------------------------------------
            bpthr = temp_spec2[selfin] > np.nanmedian(temp_spec2) + 2e0*std1
            spec_cut = spec.copy()
            data['SPECTRUM'][index] = spec_cut
            data['SPECTRUM_CLEAN'][index] = spec_cut
            # if debug:
            #    plt.plot(temp_spec)
            #    cd = np.nanmedian(temp_spec2) + 2e0*std1
            #    plt.axhline(y=cd, xmin=0, xmax=1,color='red')
            #    plt.show()
            # WAVELENGTH CALIBRATION -----------------------------------------------------
            disp_all=[]
            if np.sum(np.isfinite(spec)) > (spec.size/2):
                # wavecalspec = spec.copy()
                wavecalspec = spec[:-1]
                finitespec = np.isfinite(wavecalspec)
                spec_norm=wavecalspec[finitespec]/np.max(wavecalspec[finitespec])
                phoenix_model = phoenix(set_wav)
                bin_spec = phoenix_model[1]
                mid = phoenix_model[0]
                bin_spec_norm = bin_spec/np.max(bin_spec)
                # select=spec_norm > 1e-1
                x=np.arange(len(wavecalspec))
                x_finite=x[finitespec]
                th_norm=tt/np.max(tt)
                f = itp.interp1d(x_finite, spec_norm, bounds_error=False, fill_value=0)
                # f_x = f(x)
                mid_ang = mid*10
                g = itp.interp1d(wavett, th_norm, bounds_error=False, fill_value=0)
                cond_mid = np.where((mid_ang >= wavett[0]) & (mid_ang <= wavett[-1]))
                g_wav= g(mid_ang[cond_mid])
                # model = scipy.signal.medfilt(g_wav*bin_spec_norm[cond_mid], 5)
                # wave = np.arange(spec.size)*disper*1e-4 + shift
                x0 = (1./2.72,-1000,1.)
                result = opt.minimize(chisqfunc,x0,method='Nelder-Mead')
                d_frc = result.x[0]
                d = 1./result.x[0]
                dispersion_list.append(d)
                s = result.x[1]
                calib_spec=f(s+(d_frc)*mid_ang[cond_mid])
                data['SPECTRUM'][index] = calib_spec*np.max(wavecalspec[finitespec])
                if debug:
                    plt.plot(mid[cond_mid],calib_spec,'o',label='calibrated spec')
                    plt.plot(mid[cond_mid],g_wav*bin_spec_norm[cond_mid],'o',label='calibrated spec')
                    plt.legend(loc='lower right', shadow=False, fontsize='16', frameon=True,scatterpoints=1)
                    plt.xlabel('Wavelength [nm]')
                    plt.ylabel('Normalized Flux')
                    plt.show()
                    pass
                liref = itp.interp1d(wavett, tt,
                                     bounds_error=False, fill_value=np.nan)
                phot2counts = liref(mid_ang[cond_mid])
                data['PHT2CNT'][index] = phot2counts
                data['WAVE'][index] = mid[cond_mid]*0.001
                data['DISPERSION'][index] = d
                data['SHIFT'][index] = s
                err = data['SPECERR'][index]
                data['SPECERR'][index] = err[cond_mid]
                disp_all.append(np.median(dispersion_list))
                pass
            pass

    if debug:
        for v in set(visits):
            select = (visits == v) & ~(data['IGNORED'])
            plt.figure()
            for index, valid in enumerate(select):
                if valid: plt.plot(mid,data['SPECTRUM'][index], 'o')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Counts')
                plt.title('Visit number: '+str(int(v)))
                pass
            plt.show()
            pass
    if debug:
        plt.figure(figsize=[6,6])
        spec_all=[]
        wave_all=[]
        for v in set(visits):
            select = (visits == v) & ~(data['IGNORED'])
            for raissaindex, valid in enumerate(select):
                if valid:
                    spec_valid = data['SPECTRUM'][raissaindex]
                    wave_valid = data['WAVE'][raissaindex]
                    wave_all.append(wave_valid)
                    spec_all.append(spec_valid)
        template = np.nanmean(spec_all,0)
        flats = []
        for spectrum, w in zip(spec_all, wave_all):
            flat = spectrum/template
            flats.append(flat)
        plt.imshow(flats, cmap='jet',vmin=0, vmax=1.2, extent=(290,570,0,len(flats)))
        plt.colorbar()
        plt.title('Flattened 1D spectrum - all exposures')

    for v in set(visits):
        select = (visits == v) & ~(data['IGNORED'])
        plt.figure()
        for index, valid in enumerate(select):
            spec_all = np.array(data['SPECTRUM'][index])
            wave_all = np.array(data['WAVE'][index])
            phot_all = np.array(data['PHT2CNT'][index])
            err_all = np.array(data['SPECERR'][index])
            cond_wavcut = np.where(wave_all > 0.45)
            data['SPECTRUM'][index] = spec_all[cond_wavcut]
            data['WAVE'][index] = wave_all[cond_wavcut]
            data['PHT2CNT'][index] = phot_all[cond_wavcut]
            data['SPECERR'][index] = err_all[cond_wavcut]
            pass
        plt.show()
    pass

    if debug:
        inte_res=[]
        phase_all=[]
        for v in set(visits):
            select = (visits == 1) & ~(data['IGNORED'])
            for raissaindex, valid in enumerate(select):
                if valid:
                    phase_sel = phase[raissaindex]
                    phase_all.append(phase_sel)
                    wav = np.array(data['WAVE'][raissaindex])
                    spec = np.array(data['SPECTRUM'][raissaindex])
                    fin = np.isfinite(spec)
                    wav_fin = wav[fin]
                    spec_fin = spec[fin]
                    cond = np.where((wav_fin > 0.3) & (wav_fin < 0.5))
                    # pixels = np.arange(0,1024,1)
                    # func_spec = itp.interp1d(wav_fin,spec_fin, kind='linear')
                    # func_teste = itp.interp1d(pixels,spec, kind='linear')
                    # inte = integrate.quad(lambda x: func_teste(x), pixels[0],pixels[-1])
                    # inte = integrate.quad(lambda x: func_spec(x), 0.3, 0.54)
                    inte = np.sum(spec_fin[cond]*(wav_fin[cond[0]+1]-wav_fin[cond[0]]))
                    inte_res.append(inte)
                    # inte_res = np.array(inte_res)
                    # phase_all = np.array(phase_all)
                    # cond_out = np.where((phase_all > 0.01) | (phase_all < -0.01))
                    # oot = inte_res[cond_out]
                    # norm = inte_res/np.mean(oot)
                    pass
                pass
            pass
        pass
        # PLOTS ------------------------------------------------------------------------------
    if verbose and (not np.all(data['IGNORED'])):
        alltime = np.array([d for d,i in zip(data['TIME'], data['IGNORED']) if not i])
        dispersion = np.array([d for d,i in zip(data['DISPERSION'], data['IGNORED']) if not i])
        shift = np.array([d for d,i in zip(data['SHIFT'], data['IGNORED']) if not i])
        spec = np.array([d for d,i in zip(data['SPECTRUM'], data['IGNORED']) if not i])
        photoc = np.array([d for d,i in zip(data['PHT2CNT'], data['IGNORED']) if not i])
        wave = np.array([d for d,i in zip(data['WAVE'], data['IGNORED']) if not i])
        errspec = np.array([d for d,i in zip(data['SPECERR'], data['IGNORED']) if not i])
        torder = np.argsort(alltime)
        vrange = data['VRANGE']
        allerr = []
        for s, e, w in zip(spec, errspec, wave):
            select = (w > vrange[0]) & (w < vrange[1])
            allerr.extend(e[select]/np.sqrt(s[select]))
            pass
        allerr = np.array(allerr)
        select = np.isfinite(allerr)
        allerr = allerr[select]
        allerr = allerr[allerr > 0.9]

        plt.figure()
        for spectrum in data['SPECTRUM_CLEAN']: plt.plot(spectrum)
        plt.ylabel('Stellar Spectra [Counts]')
        plt.xlabel('Pixel Number')
        plt.figure()
        for w, p, s in zip(wave, photoc, spec):
            select = (w > vrange[0]) & (w < vrange[1])
            plt.plot(w[select], s[select]/p[select])
            pass
        plt.ylabel('Stellar Spectra [Photons]')
        plt.xlabel('Wavelength [microns]')

        plt.figure()
        plt.hist(allerr)
        plt.xlabel('Error Distribution [Noise Model Units]')

        plt.figure()
        plt.plot(dispersion[torder], 'o')
        disp_1sig = np.median(dispersion[torder])+3*np.std(dispersion[torder])
        plt.axhline(y=disp_1sig, xmin=0, xmax=1,color='red')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Dispersion [Angstroms/Pixel]')
        plt.ylim(data['DISPLIM'][0], data['DISPLIM'][1])

        plt.figure()
        plt.plot(shift[torder] - np.nanmin(shift), 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Shift [Pixels]')
        plt.show()
        pass
    allignore = data['IGNORED']
    allculprits = data['TRIAL']
    log.warning('>-- IGNORED: %s / %s', str(np.nansum(allignore)), str(len(allignore)))
    for index, ignore in enumerate(allignore):
        if ignore: log.warning('>-- %s: %s', str(index), str(allculprits[index]))
        pass
    data.pop('EXP', None)
    for key in data: out['data'][key] = data[key]
    calibrated = not np.all(data['IGNORED'])
    if calibrated: out['STATUS'].append(True)
    return calibrated

# ---------------------- ---------------------------------------------
# -------------------------- -----------------------------------------

def stiscal_unified(fin, clc, tim, tid, flttype, out,
                    verbose=False, debug=False):
    '''
    R. ESTRELA: STIS .flt data extraction and wavelength calibration FILTERS G430L and G750L
    '''
    calibrated = False
    # VISIT NUMBERING --------------------------------------------------------------------
    for pkey in tim['data'].keys(): visits = np.array(tim['data'][pkey]['dvisits'])
    # PHASE ------------------------------------------------------------------------------
    for pkey in tim['data'].keys(): phase = np.array(tim['data'][pkey]['phase'])
    # OPTICS AND FILTER ------------------------------------------------------------------
    vrange = validrange(flttype)
    _wvrng, _disp, ldisp, udisp = fng(flttype)
    # DATA FORMAT ------------------------------------------------------------------------
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')
    data = {'LOC':[], 'EPS':[], 'DISPLIM':[ldisp, udisp],
            'SCANRATE':[], 'SCANLENGTH':[], 'SCANANGLE':[],
            'EXP':[], 'EXPERR':[], 'EXPFLAG':[], 'VRANGE':vrange,
            'TIME':[], 'EXPLEN':[], 'MIN':[], 'MAX':[], 'TRIAL':[], 'TIMEOBS':[], 'DATEOBS':[]}
    # LOAD DATA --------------------------------------------------------------------------
    for loc in sorted(clc['LOC']):
        fullloc = os.path.join(dbs, loc)
        with pyfits.open(fullloc) as hdulist:
            header0 = hdulist[0].header
            if 'SCAN_ANG' in header0: scanangle = header0['SCAN_ANG']
            elif 'PA_V3' in header0: scanangle = header0['PA_V3']
            else: scanangle = 666
            allloc = []
            alltime = []
            allexplen = []
            alleps = []
            allexp = []
            allexperr = []
            allmask = []
            allmin = []
            allmax = []
            alldate = []
            alltimeobs = []
            for fits in hdulist:
                if (fits.size != 0) and (fits.header['EXTNAME']=='SCI'):
                    allloc.append(fits.header['EXPNAME'])
                    alltime.append(float(fits.header['EXPEND']))
                    allexplen.append(float(fits.header['EXPTIME']))
                    alldate.append(header0['TDATEOBS'])
                    alltimeobs.append(header0['TTIMEOBS'])
                    fitsdata = np.empty(fits.data.shape)
                    fitsdata[:] = fits.data[:]
                    test = fits.header['BUNIT']
                    eps = False
                    if test != 'COUNTS': eps = True
                    alleps.append(eps)
                    allmin.append(float(fits.header['GOODMIN']))
                    allmax.append(float(fits.header['GOODMAX']))
                    # BINARIES
                    # GMR: Let's put that in the mask someday
                    if tid in ['HAT-P-1']: allexp.append(fitsdata[0:120, :])
                    else: allexp.append(fitsdata)
                    del fits.data
                    pass
                if 'EXTNAME' in fits.header:
                    if (fits.header['EXTNAME'] in ['ERR', 'DQ']):
                        fitsdata = np.empty(fits.data.shape)
                        fitsdata[:] = fits.data[:]
                        if fits.header['EXTNAME'] == 'ERR': allexperr.append(fitsdata)
                        if fits.header['EXTNAME'] == 'DQ': allmask.append(fitsdata)
                        del fits.data
                        pass
                    if eps:
                        eps2count = allexplen[-1]*float(header0['CCDGAIN'])
                        allexp[-1] = allexp[-1]*eps2count
                        allexperr[-1] = allexperr[-1]*eps2count
                        pass
                    pass
                pass
            allscanangle = [scanangle]*len(allloc)
            allscanlength = [1e0]*len(allloc)
            allscanrate = [0e0]*len(allloc)
            data['LOC'].extend(allloc)
            data['EPS'].extend(alleps)
            data['SCANRATE'].extend(allscanrate)
            data['SCANLENGTH'].extend(allscanlength)
            data['SCANANGLE'].extend(allscanangle)
            data['EXP'].extend(allexp)
            data['EXPERR'].extend(allexperr)
            data['TIME'].extend(alltime)
            data['EXPLEN'].extend(allexplen)
            data['MIN'].extend(allmin)
            data['MAX'].extend(allmax)
            data['TIMEOBS'].extend(alltimeobs)
            data['DATEOBS'].extend(alldate)
            pass
        pass
    data['MEXP'] = data['EXP'].copy()
    data['MASK'] = data['EXPFLAG'].copy()
    data['ALLDATEOBS'] = data['DATEOBS'].copy()
    data['ALLTIMEOBS'] = data['TIMEOBS'].copy()
    data['IGNORED'] = np.array([False]*len(data['LOC']))
    data['FLOODLVL'] = [np.nan]*len(data['LOC'])
    data['TRIAL'] = ['']*len(data['LOC'])
    data['SPECTRUM0'] = [np.array([np.nan])]*len(data['LOC'])
    data['SPECTRUM'] = [np.array([np.nan])]*len(data['LOC'])
    data['SPECTRUM_CLEAN'] = [np.array([np.nan])]*len(data['LOC'])
    data['SPECERR'] = [np.array([np.nan])]*len(data['LOC'])
    data['TEMPLATE'] = [np.array([np.nan])]*len(data['LOC'])
    data['PHT2CNT'] = [np.array([np.nan])]*len(data['LOC'])
    data['NSPEC'] = [1e0]*len(data['LOC'])
    # REJECT OUTLIERS IN EXPOSURE LENGTH -------------------------------------------------
    for v in set(visits):
        select = visits == v
        visitexplength = np.array(data['EXPLEN'])[select]
        visitignore = data['IGNORED'][select]
        ref = np.nanmedian(visitexplength)
        visitignore[visitexplength != ref] = True
        data['IGNORED'][select] = visitignore
        pass
#     # COSMIC RAYS REJECTION - MEDIAN FILTER + SIGMA CLIPPING
#     for index, ignore in enumerate(data['IGNORED']):
#         # COSMIC RAY REJECTION IN THE 2D IMAGE
#         frame = data['MEXP'][index].copy()
#         img_cr = frame.copy()
#         allframe_list = []
#         for i in range(0,len(frame)):
#             img_sm = scipy.signal.medfilt(img_cr[i,:], 9)
#             std = np.std(img_cr[i,:] - img_sm)
#             # std = np.std(img_sm)
#             bad = np.abs(img_cr[i,:] - img_sm) > 2*std
#             line = img_cr[i,:]
#             line[bad] = img_sm[bad]
#             img_sm2 = scipy.signal.medfilt(line, 9)
#             std2 = np.std(line - img_sm2)
#             bad2 = np.abs(line - img_sm2) > 2*std2
#             line2 = line.copy()
#             line2[bad2] = img_sm2[bad2]
#             allframe_list.append(line2)
#             pass
#         allframe = np.array(allframe_list)

        # FLAT FRINGE G750L
    for index, ignore in enumerate(data['IGNORED']):
        if 'G750L' in flttype:
            # SELECT DATE AND TIME OF THE EXPOSURE FOR FLAT FRINGE SELECTION
            frame = data['MEXP'][index].copy()
            dateobs_exp = data['ALLDATEOBS'][index]
            timeobs_exp = data['ALLTIMEOBS'][index]
            tog_exp = dateobs_exp +' '+ timeobs_exp
            time_exp = raissatime.mktime(datetime.datetime.strptime(tog_exp, "%Y-%m-%d %H:%M:%S").timetuple())
            # LOAD FRINGE FLAT -------------------------------------------------------------------
            obs_name = clc['ROOTNAME'][0]
            name_sel = obs_name[:-5]
            lightpath_fringe = ('STIS/CCDFLAT/')
            calloc = excalibur.context['data_cal']
            filefringe = os.path.join(calloc,lightpath_fringe)
            if tid in ['HD 209458']:
                lightpath_fringe = ('STIS/CCDFLAT/h230851ao_pfl.fits')
                calloc = excalibur.context['data_cal']
                filefringe = os.path.join(calloc,lightpath_fringe)
                hdu = pyfits.open(filefringe)
                data_fringe = hdu[1].data
                err_fringe = hdu[2].data
                pass
            else:
                diff_list = []
                all_infile = []
                for infile in glob.glob('%s/%s*_flt.fits' % (filefringe,name_sel)):
                    hdu = pyfits.open(infile)
                    all_infile.append(infile)
                    header_flat = hdu[0].header
                    date_time=header_flat['TDATEOBS']
                    hour_time=header_flat['TTIMEOBS']
                    tog = date_time +' '+ hour_time
                    time_flat_s = raissatime.mktime(datetime.datetime.strptime(tog, "%Y-%m-%d %H:%M:%S").timetuple())
                    diff = abs(time_exp-time_flat_s)
                    diff_list.append(diff)
                    pass
                cond_win = np.where(diff_list == np.min(diff_list))
                all_infile = np.array(all_infile)
                sel_flatfile = all_infile[cond_win][0]
                hdulist = pyfits.open(sel_flatfile)
                data_fringe = hdulist[4].data
                err_fringe = hdulist[5].data
                pass
            smooth_fringe = scipy.signal.medfilt(data_fringe, 7)
            sigma_fringe = np.median(err_fringe)
            bad_fringe = (np.abs(data_fringe - smooth_fringe) / sigma_fringe) > 2
            img_fringe = data_fringe.copy()
            img_fringe[bad_fringe] = smooth_fringe[bad_fringe]
            if debug:
                plt.figure()
                for i in range(0,len(data_fringe)):
                    plt.plot(img_fringe[i,:])
                    pass
                pass
            cont_data = img_fringe.copy()
            div_list=[]
            for ll in range(508,515):
                pixels = np.arange(0,1024,1)
                coefs = poly.polyfit(pixels,cont_data[ll,:], 11)
                ffit = poly.polyval(pixels, coefs)
                div = cont_data[ll,:]/ffit
                div_list.append(div)
                # COSMIC RAY REJECTION IN THE 2D IMAGE
            img_cr = frame.copy()
            allframe_list = []
            for i in range(0,len(frame)):
                img_sm = scipy.signal.medfilt(img_cr[i,:], 9)
                # std = np.std(img_cr[i,:] - img_sm)
                std = np.std(img_sm)
                bad = np.abs(img_cr[i,:] - img_sm) > 3*std
                line = img_cr[i,:]
                line[bad] = img_sm[bad]
                allframe_list.append(line)
                pass
            allframe = np.array(allframe_list)
            # APPLY FLAT FRINGE
            # plt.figure()
            if not ignore:
                find_spec = np.where(allframe == np.max(allframe))
                spec_idx = find_spec[0][0]
                spec_idx_up = spec_idx+4
                spec_idx_dwn = spec_idx-3
                spec_idx_all = np.arange(spec_idx_dwn,spec_idx_up,1)
                frame2 = allframe.copy()
                for i,flatnorm in zip(spec_idx_all,div_list):
                    frame_sel = allframe[i,:]
                    coefs_f = poly.polyfit(pixels,frame_sel, 12)
                    ffit_f = poly.polyval(pixels, coefs_f)
                    frame2[i,400:1023] = frame2[i,400:1023]/flatnorm[400:1023]
                    if debug:
                        plt.subplot(2, 1, 1)
                        plt.plot(pixels,frame_sel,color='blue')
                        plt.plot(pixels, ffit_f,color='red')
                        plt.subplot(2, 1, 2)
                        norm = frame_sel/ffit_f
                        plt.plot(norm, color='orange',label='Observed spectrum')
                        plt.plot(flatnorm,color='blue',label='Contemporaneous Flat fringe')
                        plt.xlabel('pixels')
                        plt.ylabel('Normalized flux')
                        plt.legend(loc='lower right', shadow=False, frameon=False, fontsize='7', scatterpoints=1)
                        pass
                    pass
                    data['SPECTRUM'][index] = np.nansum(frame2, axis=0)
                    data['SPECERR'][index] = np.sqrt(np.nansum(frame2, axis=0))
                    pass
                pass
            else:
                data['SPECTRUM'][index] = np.nansum(frame, axis=0)*np.nan
                data['SPECERR'][index] = np.nansum(frame, axis=0)*np.nan
                data['TRIAL'][index] = 'Exposure Length Outlier'
                pass
            pass
        if 'G430' in flttype:
            if not ignore:
                data['SPECTRUM'][index] = np.nansum(allframe, axis=0)
                data['SPECERR'][index] = np.sqrt(np.nansum(allframe, axis=0))
                data['PHT2CNT'][index] = [np.nan]*len(frame[0])
                pass
            else:
                data['SPECTRUM'][index] = np.nansum(allframe, axis=0)*np.nan
                data['SPECERR'][index] = np.nansum(allframe, axis=0)*np.nan
                data['TRIAL'][index] = 'Exposure Length Outlier'
                data['PHT2CNT'][index] = [np.nan]*len(frame[0])
                pass
            pass
    # if debug:
    #    for v in set(visits):
    #        select = (visits == v) & ~(data['IGNORED'])
    #        plt.figure()
    #        for index, valid in enumerate(select):
    #            if valid: plt.plot(data['SPECTRUM'][index], 'o')
    #            pass
    #            plt.xlabel('Wavelength [microns]')
    #            plt.ylabel('Counts')
    #            plt.title('Visit number: '+str(int(v)))
    #            pass
    #        plt.show()
    pass
    wavett, tt = ag2ttf(flttype)
    if 'G430' in flttype:
        select = wavett > 0.29e4
        wavett = wavett[select]
        tt = tt[select]
        pass
    if verbose and (not np.all(data['IGNORED'])):
        plt.figure()
        for spectrum in data['SPECTRUM']: plt.plot(spectrum)
        plt.ylabel('Stellar Spectra [Counts]')
        plt.xlabel('Pixel Number')
        plt.show()
        # MASK BAD PIXELS IN SPECTRUM --------------------------------------------------------
    for v in set(visits):
        select = (visits == v) & ~(data['IGNORED'])
        specarray = np.array([s for s, ok in zip(data['SPECTRUM'], select) if ok])
        trans = np.transpose(specarray)
        template = np.nanmedian(trans, axis=1)
        # TEMPLATE MEDIAN 5 POINTS LOW PASS FILTER ---------------------------------------
        smootht = []
        smootht.extend([template[0]]*2)
        smootht.extend(template)
        smootht.extend([template[-1]]*2)
        for index in np.arange(len(template)):
            medianvalue = np.nanmedian(template[index:index+5])
            smootht[2+index] = medianvalue
            pass
        smootht = smootht[2:-2]
        template = np.array(smootht)
        for vindex, valid in enumerate(select):
            if valid: data['TEMPLATE'][vindex] = template
            pass
        pass
    # COSMIC RAYS REJECTION
    # data['PHT2CNT'] = [np.nan]*len(data['LOC'])
    data['WAVE'] = [np.array([np.nan])]*len(data['LOC'])
    data['DISPERSION'] = [np.nan]*len(data['LOC'])
    data['SHIFT'] = [np.nan]*len(data['LOC'])

    if 'G430' in flttype:
        set_wav = np.array([290,570])
    if 'G750' in flttype:
        set_wav = np.array([524,1027])

    def phoenix(set_wav):
        # PHOENIX MODELS
        filters = [BoxcarFilter('a', 550, 950)]  # Define your passbands
        feherr=np.sqrt(abs(fin['priors']['FEH*_uperr']*fin['priors']['FEH*_lowerr']))
        loggerr = np.sqrt(abs(fin['priors']['LOGG*_uperr']*
                              fin['priors']['LOGG*_lowerr']))
        terr = np.sqrt(abs(fin['priors']['T*_uperr']*fin['priors']['T*_lowerr']))
        sc = LDPSetCreator(teff=(fin['priors']['T*'], terr),
                           logg=(fin['priors']['b']['logg'], loggerr),
                           z=(fin['priors']['FEH*'], feherr),
                           filters=filters)
        list_diff = []
        for i in range(0,len(sc.files)):
            hdul = pyfits.open(sc.files[i])
            teff = hdul[0].header['PHXTEFF']
            zz = hdul[0].header['PHXM_H']
            logg_str = hdul[0].header['PHXLOGG']
            diff1 = abs(fin['priors']['T*'] - teff)
            diff2 = abs(fin['priors']['FEH*'] - zz)
            diff3 = abs(fin['priors']['b']['logg'] - logg_str)
            diff_total = diff1 + diff2 + diff3
            list_diff.append(diff_total)
        cond_win = np.where(list_diff == np.min(list_diff))
        hdul2 = pyfits.open(sc.files[cond_win[0][0]])  # 1 for HAT-p-26 and Hat-P-11, 3 for Hat-p-18, 1 for WASP-52, 4 for WASP-80
        data_all = hdul2[0].data
        wl0 = hdul[0].header['crval1']*1e-1  # defines the wavelength at pixel CRPIX1
        dwl = hdul[0].header['cdelt1']*1e-1  # Delta wavelength     [nm]
        nwl = hdul[0].header['naxis1']  # Number of wl samples
        wl = wl0 + np.arange(nwl)*dwl
        model = data_all[77]  # take only the last spectra of each fits
        # Average the spectra to get 1 spectrum model
        # new_spec=[]
        # trans_listdata = np.transpose(list_models)
        # for i in range(0,len(trans_listdata)):
        #    med_wav = np.mean(trans_listdata[i])
        #    new_spec.append(med_wav)
        # f_spec = itp.interp1d(wl, new_spec, bounds_error=False)
        # spec_sel = f_spec(wavett*0.1)
        # spec_sel_norm = spec_sel/np.max(spec_sel)
        cond_wav = np.where((wl > set_wav[0]) & (wl < set_wav[1]))
        wl_sel = wl[cond_wav]
        new_spec = np.array(model)
        if 'G430' in flttype:
            spec_sm = scipy.signal.medfilt(new_spec, 9)
        else:
            spec_sm = new_spec
        new_spec_sel = spec_sm[cond_wav]
        mid, low, high, binsz = binnagem(wl_sel,1024)
        # func_spec = scipy.interpolate.interp1d(wl_sel,new_spec_sel)
        # BINNING PHOENIX MODEL
        bin_spec=[]
        for w_low, w_hi in zip(low, high):
            select = np.where((wl_sel > w_low) & (wl_sel < w_hi))
            # inte = scipy.integrate.quad(lambda x: func_spec(x), w_low, w_hi)
            inte = np.sum(new_spec_sel[select]*(wl_sel[select[0]+1]-wl_sel[select[0]]))
            databin=inte/binsz[0]  # inte[0] if scipy.integrate
            bin_spec.append(databin)
        bin_spec=np.array(bin_spec)
        if 'G430' in flttype:
            window = 5
        else:
            window = 5
        bin_spec = scipy.signal.medfilt(bin_spec, window)
        return mid, bin_spec

    def chisqfunc(args):
        avar, bvar, scvar = args
        chisq = np.sum(((g_wav*bin_spec_norm[cond_mid])*scvar - f(bvar+(avar)*mid_ang[cond_mid]))**2)
        return chisq

    # CALIBRATION
    dispersion_list = []
    for index, rejected in enumerate(data['IGNORED']):
        if not rejected:
            spec = data['SPECTRUM'][index]
            template = data['TEMPLATE'][index]
            temp_spec = spec/template
            ht25 = np.nanpercentile(temp_spec,25)
            lt75 = np.nanpercentile(temp_spec,75)
            std = np.std(temp_spec[(temp_spec > ht25) & (temp_spec < lt75)])
            # BAD PIXEL THRESHOLD --------------------------------------------------------
            bpthr = temp_spec > np.nanmedian(temp_spec) + 2e0*std
            if True in bpthr: spec[bpthr] = np.nan
            # second
            temp_spec2 = spec/template
            ht25 = np.nanpercentile(temp_spec2,25)
            lt75 = np.nanpercentile(temp_spec2,75)
            selfin = np.isfinite(temp_spec2)
            std1 = np.nanstd(temp_spec2[selfin][(temp_spec2[selfin] > ht25) & (temp_spec2[selfin] < lt75)])
            # BAD PIXEL THRESHOLD --------------------------------------------------------
            bpthr = temp_spec2[selfin] > np.nanmedian(temp_spec2) + 2e0*std1
            spec_cut = spec.copy()
            data['SPECTRUM'][index] = spec_cut
            data['SPECTRUM_CLEAN'][index] = spec_cut
            # if debug:
            #    plt.plot(temp_spec)
            #    cd = np.nanmedian(temp_spec2) + 2e0*std1
            #    plt.axhline(y=cd, xmin=0, xmax=1,color='red')
            #    plt.show()
            # WAVELENGTH CALIBRATION -----------------------------------------------------
            disp_all=[]
            if np.sum(np.isfinite(spec)) > (spec.size/2):
                wavecalspec = spec
                # wavecalspec = spec.copy()
                if 'G430' in flttype:
                    wavecalspec = spec[:-1]
                finitespec = np.isfinite(wavecalspec)
                spec_norm=wavecalspec[finitespec]/np.max(wavecalspec[finitespec])
                phoenix_model = phoenix(set_wav)
                bin_spec = phoenix_model[1]
                mid = phoenix_model[0]
                bin_spec_norm = bin_spec/np.max(bin_spec)
                # select=spec_norm > 1e-1
                x=np.arange(len(wavecalspec))
                x_finite=x[finitespec]
                th_norm=tt/np.max(tt)
                f = itp.interp1d(x_finite, spec_norm, bounds_error=False, fill_value=0)
                # f_x = f(x)
                mid_ang = mid*10
                g = itp.interp1d(wavett, th_norm, bounds_error=False, fill_value=0)
                cond_mid = np.where((mid_ang >= wavett[0]) & (mid_ang <= wavett[-1]))
                g_wav= g(mid_ang[cond_mid])
                # model = scipy.signal.medfilt(g_wav*bin_spec_norm[cond_mid], 5)
                # wave = np.arange(spec.size)*disper*1e-4 + shift
                if 'G750' in flttype:
                    x0 = (1./4.72,-1000,1.)
                else:
                    x0 = (1./2.72,-1000,1.)
                result = opt.minimize(chisqfunc,x0,method='Nelder-Mead')
                d_frc = result.x[0]
                d = 1./result.x[0]
                dispersion_list.append(d)
                s = result.x[1]
                sc = result.x[2]
                calib_spec=f(s+(d_frc)*mid_ang[cond_mid])
                data['SPECTRUM'][index] = calib_spec*np.max(wavecalspec[finitespec])
                if debug:
                    plt.plot(mid[cond_mid],calib_spec,'o',label='calibrated spec')
                    plt.plot(mid[cond_mid],g_wav*bin_spec_norm[cond_mid]*sc,'o',label='calibrated spec')
                    plt.legend(loc='lower right', shadow=False, fontsize='16', frameon=True,scatterpoints=1)
                    plt.xlabel('Wavelength [nm]')
                    plt.ylabel('Normalized Flux')
                    plt.show()
                    pass
                liref = itp.interp1d(wavett, tt,
                                     bounds_error=False, fill_value=np.nan)
                phot2counts = liref(mid_ang[cond_mid])
                data['PHT2CNT'][index] = phot2counts
                data['WAVE'][index] = mid[cond_mid]*0.001
                data['DISPERSION'][index] = d
                data['SHIFT'][index] = s
                err = data['SPECERR'][index]
                data['SPECERR'][index] = err[cond_mid]
                disp_all.append(np.median(dispersion_list))
                pass
            pass

    if verbose:
        for v in set(visits):
            select = (visits == v) & ~(data['IGNORED'])
            plt.figure()
            for index, valid in enumerate(select):
                if valid: plt.plot(mid[cond_mid],data['SPECTRUM'][index], 'o')
                plt.xlabel('Wavelength [nm]')
                plt.ylabel('Counts')
                plt.title('Visit number: '+str(int(v)))
                pass
            plt.show()
            pass
    if debug:
        plt.figure(figsize=[6,6])
        spec_all=[]
        wave_all=[]
        for v in set(visits):
            select = (visits == v) & ~(data['IGNORED'])
            for raissaindex, valid in enumerate(select):
                if valid:
                    spec_valid = data['SPECTRUM'][raissaindex]
                    wave_valid = data['WAVE'][raissaindex]
                    wave_all.append(wave_valid)
                    spec_all.append(spec_valid)
        template = np.nanmean(spec_all,0)
        flats = []
        for spectrum, w in zip(spec_all, wave_all):
            flat = spectrum/template
            flats.append(flat)
        plt.imshow(flats, cmap='jet',vmin=0, vmax=1.2, extent=(290,570,0,len(flats)))
        plt.colorbar()
        plt.title('Flattened 1D spectrum - all exposures')

    if 'G430' in flttype:
        for v in set(visits):
            select = (visits == v) & ~(data['IGNORED'])
            for index, valid in enumerate(select):
                spec_all = np.array(data['SPECTRUM'][index])
                wave_all = np.array(data['WAVE'][index])
                phot_all = np.array(data['PHT2CNT'][index])
                err_all = np.array(data['SPECERR'][index])
                cond_wavcut = np.where(wave_all > 0.45)
                data['SPECTRUM'][index] = spec_all[cond_wavcut]
                data['WAVE'][index] = wave_all[cond_wavcut]
                data['PHT2CNT'][index] = phot_all[cond_wavcut]
                data['SPECERR'][index] = err_all[cond_wavcut]
                pass
            pass

    if debug:
        inte_res=[]
        phase_all=[]
        for v in set(visits):
            select = (visits == 1) & ~(data['IGNORED'])
            for raissaindex, valid in enumerate(select):
                if valid:
                    phase_sel = phase[raissaindex]
                    phase_all.append(phase_sel)
                    wav = np.array(data['WAVE'][raissaindex])
                    spec = np.array(data['SPECTRUM'][raissaindex])
                    fin = np.isfinite(spec)
                    wav_fin = wav[fin]
                    spec_fin = spec[fin]
                    cond = np.where((wav_fin > 0.3) & (wav_fin < 0.5))
                    # pixels = np.arange(0,1024,1)
                    # func_spec = itp.interp1d(wav_fin,spec_fin, kind='linear')
                    # func_teste = itp.interp1d(pixels,spec, kind='linear')
                    # inte = integrate.quad(lambda x: func_teste(x), pixels[0],pixels[-1])
                    # inte = integrate.quad(lambda x: func_spec(x), 0.3, 0.54)
                    inte = np.sum(spec_fin[cond]*(wav_fin[cond[0]+1]-wav_fin[cond[0]]))
                    inte_res.append(inte[0])
                    pass
                pass
            pass
        pass
        # PLOTS ------------------------------------------------------------------------------
    if verbose and (not np.all(data['IGNORED'])):
        alltime = np.array([d for d,i in zip(data['TIME'], data['IGNORED']) if not i])
        dispersion = np.array([d for d,i in zip(data['DISPERSION'], data['IGNORED']) if not i])
        shift = np.array([d for d,i in zip(data['SHIFT'], data['IGNORED']) if not i])
        spec = np.array([d for d,i in zip(data['SPECTRUM'], data['IGNORED']) if not i])
        photoc = np.array([d for d,i in zip(data['PHT2CNT'], data['IGNORED']) if not i])
        wave = np.array([d for d,i in zip(data['WAVE'], data['IGNORED']) if not i])
        errspec = np.array([d for d,i in zip(data['SPECERR'], data['IGNORED']) if not i])
        torder = np.argsort(alltime)
        vrange = data['VRANGE']
        allerr = []
        for s, e, w in zip(spec, errspec, wave):
            select = (w > vrange[0]) & (w < vrange[1])
            allerr.extend(e[select]/np.sqrt(s[select]))
            pass
        allerr = np.array(allerr)
        select = np.isfinite(allerr)
        allerr = allerr[select]
        allerr = allerr[allerr > 0.9]

        plt.figure()
        for spectrum in data['SPECTRUM0']: plt.plot(spectrum)
        plt.ylabel('Stellar Spectra [Counts]')
        plt.xlabel('Pixel Number')
        plt.figure()
        for w, p, s in zip(wave, photoc, spec):
            select = (w > vrange[0]) & (w < vrange[1])
            plt.plot(w[select], s[select]/p[select])
            pass
        plt.ylabel('Stellar Spectra [Photons]')
        plt.xlabel('Wavelength [microns]')

        plt.figure()
        plt.hist(allerr)
        plt.xlabel('Error Distribution [Noise Model Units]')

        plt.figure()
        plt.plot(dispersion[torder], 'o')
        disp_1sig = np.median(dispersion[torder])+3*np.std(dispersion[torder])
        plt.axhline(y=disp_1sig, xmin=0, xmax=1,color='red')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Dispersion [Angstroms/Pixel]')
        # plt.ylim(data['DISPLIM'][0], data['DISPLIM'][1])

        plt.figure()
        plt.plot(shift[torder] - np.nanmin(shift), 'o')
        plt.xlabel('Time Ordered Frame Number')
        plt.ylabel('Shift [Pixels]')
        plt.show()
        pass
    allignore = data['IGNORED']
    allculprits = data['TRIAL']
    log.warning('>-- IGNORED: %s / %s', str(np.nansum(allignore)), str(len(allignore)))
    for index, ignore in enumerate(allignore):
        if ignore: log.warning('>-- %s: %s', str(index), str(allculprits[index]))
        pass
    data.pop('EXP', None)
    for key in data: out['data'][key] = data[key]
    calibrated = not np.all(data['IGNORED'])
    if calibrated: out['STATUS'].append(True)
    return calibrated


# ---------------------- ---------------------------------------------
# -------------------------- -----------------------------------------
def binnagem(t, nbins):
    tmax = t[-1]
    tmin = t[0]
    tbin = (tmax-tmin)*np.arange(nbins+1)/nbins
    tbin = tbin + tmin
    lower = np.resize(tbin, len(tbin)-1)
    tmid = lower + 0.5*np.diff(tbin)
    higher = tmid + 0.5*np.diff(tbin)
    binsize = np.diff(tbin)
    return tmid, lower, higher, binsize
# ---------------------- ---------------------------------------------
# -------------------------- -----------------------------------------
# -- SPITZER CALIBRATION -- ------------------------------------------
def spitzercal(clc, out):
    '''
    K. PEARSON: SPITZER data extraction
    '''
    calibrated = False
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')

    data = {
        'LOC':[], 'EXPLEN':[], 'EXP':[], 'EXPC':[], 'TIME':[],
        'FRAME':[], 'NOISEPIXEL': [], 'FAIL':[],
        'PHOT':[], 'WX':[], 'WY':[], 'BG': [],  # Aperture photometry
    }
    c = 0
    # LOAD DATA --------------------------------------------------------------------------
    for loc in sorted(clc['LOC']):
        fullloc = os.path.join(dbs, loc)
        with pyfits.open(fullloc) as hdulist:
            alltime = []
            allexplen = []
            allloc = []
            allframes = []
            allfail = []
            # photometry
            allbg = []              # background flux
            allwx = []; allwy = []  # flux weighted centroids
            allphot = []            # aperture flux
            allnp = []              # noise pixel

            for fits in hdulist:
                if (fits.size != 0) and (fits.header.get('exptype')=='sci'):
                    start = fits.header.get('MJD_OBS') + 2400000.5

                    if fits.data.ndim == 2:  # full frame data
                        # simulate subframe
                        xs = fits.data.shape[1]
                        ys = fits.data.shape[0]
                        subdata = fits.data[int(ys/2)-16:int(ys/2)+16, int(xs/2)-16:int(xs/2)+16]
                        dcube = np.array([subdata])
                    elif fits.data.ndim == 3:
                        dcube = fits.data.copy()
                    # convert from ADU to e/s
                    dcube *= float(fits.header.get('FLUXCONV',0.1257))
                    dcube *= float(fits.header.get('GAIN',3.7))

                    idur = fits.header.get('ATIMEEND') - fits.header.get('AINTBEG')
                    nimgs = dcube.shape[0]
                    dt = idur/nimgs/(24*60*60)
                    dcube[np.isnan(dcube)] = 0
                    dcube[np.isinf(dcube)] = 0
                    template = np.median(dcube,0)

                    template = template-template.min()
                    template /= np.max(template)

                    residual = dcube.copy()

                    for i in range(nimgs):

                        N = template.flatten().shape[0]
                        A = np.vstack([template.flatten(), np.ones(N)]).T
                        aa, bb = np.linalg.lstsq(A, dcube[i].flatten())[0]
                        residual[i] = dcube[i] - (aa*template+bb)

                        # allamp_temp.append(aa)
                        # allbg_temp.append(bb)
                        # save template ??

                        alltime.append(start+dt*i)
                        allexplen.append(dt*24*60)  # exposure time [s]
                        allloc.append(loc)
                        allframes.append(i)
                        pass
                    pass

                    # 3sigma clip on standard deviation of residual typically removes value from star
                    # stds = [np.std(residual[i]) for i in range(residual.shape[0])]
                    # mask = np.zeros(residual.shape).astype(bool)
                    # for i in range(mask.shape[0]): mask[i] = np.abs(residual[i])>4*stds[i]

                    mask = np.abs(residual)>4*np.std(residual,0)  # compute standard deviation in time

                    # replace bad pixels
                    for i in range(nimgs):
                        labels, nlabel = label(mask[i])
                        for j in range(1,nlabel+1):
                            mmask = labels==j  # mini mask
                            smask = binary_dilation(mmask)  # dilated mask
                            bmask = np.logical_xor(smask,mmask)  # bounding pixels
                            dcube[i][mmask] = np.mean(dcube[i][bmask])  # replace
                            pass
                        fail = False
                        try:
                            # quickly estimate brightest pixel on detector
                            # lets hope there are no cosmic rays
                            yc, xc = np.unravel_index(np.argmax(dcube[i],axis=None), dcube[i].shape)

                            # estimate priors
                            xv,yv = mesh_box([xc,yc],5)
                            wx = np.sum(np.unique(xv)*dcube[i][yv,xv].sum(0))/np.sum(dcube[i][yv,xv].sum(0))
                            wy = np.sum(np.unique(yv)*dcube[i][yv,xv].sum(1))/np.sum(dcube[i][yv,xv].sum(1))

                        except (ValueError, IndexError):  # todo find the correct exception
                            fail = True

                        try:
                            area2, bg2, np2 = phot(dcube[i], wx, wy, r=2, dr=8)
                            area25, bg25, np25 = phot(dcube[i], wx, wy, r=2.5, dr=8)
                            area3, bg3, np3 = phot(dcube[i], wx, wy, r=3, dr=8)
                            area35, bg35, np35 = phot(dcube[i], wx, wy, r=3.5, dr=8)
                            area4, bg4, np4 = phot(dcube[i], wx, wy, r=4, dr=8)
                        except (ValueError, IndexError):
                            fail = True
                            area2, area25, area3, area35, area4 = 0,0,0,0,0
                            bg2, bg25, bg3, bg35, bg4 = 0,0,0,0,0
                            np2, np25, np3, np35, np4 = 0,0,0,0,0

                        # aperture photometry
                        allwx.append(wx)
                        allwy.append(wy)
                        allbg.append([bg2, bg25, bg3, bg35, bg4])
                        allphot.append([area2, area25, area3, area35, area4])
                        allnp.append([np2, np25, np3, np35, np4])

                        allfail.append(fail)
                        pass

            data['TIME'].extend(alltime)       # MJD of observation
            data['EXPLEN'].extend(allexplen)   # exposure time
            data['LOC'].extend(allloc)        # file path on disk
            data['FRAME'].extend(allframes)   # frame number in data cube

            data['NOISEPIXEL'].extend(allnp)  # noise pixel parameter, Beta
            data['PHOT'].extend(allphot)      # aperture size
            data['BG'].extend(allbg)          # background value
            data['WX'].extend(allwx)          # flux weighted centroid
            data['WY'].extend(allwy)          #

            data['FAIL'].extend(allfail)      # fail flag - usually can't fit centroid or do aperture phot
            c+=1

    for key in data: out['data'][key] = data[key]
    calibrated = not np.all(data['FAIL'])
    if calibrated: out['STATUS'].append(True)

    return calibrated


def mixed_psf(x,y,x0,y0,a,sigx,sigy,rot,b, w):
    '''
    K. PEARSON: weighted sum of Gaussian + Lorentz PSF
    '''
    gaus = gaussian_psf(x,y,x0,y0,a,sigx,sigy,rot, 0)
    lore = lorentz_psf(x,y,x0,y0,a,sigx,sigy,rot, 0)
    return (1-w)*gaus + w*lore + b

def gaussian_psf(x,y,x0,y0,a,sigx,sigy,rot, b):
    '''
    K. PEARSON: Gaussian PSF
    '''
    rx = (x-x0)*np.cos(rot) - (y-y0)*np.sin(rot)
    ry = (x-x0)*np.sin(rot) + (y-y0)*np.cos(rot)
    gausx = np.exp(-(rx)**2 / (2*sigx**2))
    gausy = np.exp(-(ry)**2 / (2*sigy**2))
    return a*gausx*gausy + b

def lorentz_psf(x,y,x0,y0,a,sigx,sigy,rot, b):
    '''
    K. PEARSON: Lorentz PSF
    '''
    rx = (x-x0)*np.cos(rot) - (y-y0)*np.sin(rot)
    ry = (x-x0)*np.sin(rot) + (y-y0)*np.cos(rot)
    lorex = sigx**2 / (rx**2 + sigx**2)
    lorey = sigy**2 / (ry**2 + sigy**2)
    return a*lorex*lorey + b

def mesh_box(pos,box):
    '''
    K. PEARSON: array indices for box extraction
    '''
    pos = [int(np.round(pos[0])),int(np.round(pos[1]))]
    x = np.arange(pos[0]-box, pos[0]+box+1)
    y = np.arange(pos[1]-box, pos[1]+box+1)
    xv, yv = np.meshgrid(x, y)
    return xv.astype(int),yv.astype(int)

def fit_psf(data,pos,init,lo,up,psf_function=gaussian_psf,lossfn='linear',box=15):
    '''
    K. PEARSON: fitting routine for different PSFs
    '''
    xv,yv = mesh_box(pos, box)

    def fcn2min(pars):
        model = psf_function(xv,yv,*pars)
        return (data[yv,xv]-model).flatten()
    res = least_squares(fcn2min,x0=[*pos,*init],bounds=[lo,up],loss=lossfn,jac='3-point')
    return res.x

def phot(data,xc,yc,r=5,dr=4):
    '''
    K. PEARSON: Aperture photometry
    '''
    if dr>0:
        bgflux = skybg_phot(data,xc,yc,r,dr)
    else:
        bgflux = 0
    data = data-bgflux
    data[data<0] = 0

    # interpolate to high res grid for subpixel precision
    xvh,yvh = mesh_box([xc,yc], (np.round(r)+1)*10)
    rvh = ((xvh-xc)**2 + (yvh-yc)**2)**0.5
    maskh = (rvh<r*10)
    # downsize to native resolution to get pixel weights
    xv,yv = mesh_box([xc,yc], (np.round(r)+1))

    mask = np.array(pilimage.fromarray(maskh.astype(float)).resize(xv.shape, pilimage.BILINEAR))
    mask = mask / mask.max()
    flux = np.sum(data[yv,xv] * mask)
    # fmask = flux*mask  # should this be used for NP calculation?
    noisepixel = data[yv,xv].sum()**2 / np.sum(data[yv,xv]**2)
    return flux, bgflux, noisepixel

def skybg_phot(data,xc,yc,r=5,dr=5):
    '''
    K. PEARSON: Aperture photometry for sky annulus
    '''
    # create a crude annulus to mask out bright background pixels
    xv,yv = mesh_box([xc,yc], np.round(r+dr))
    rv = ((xv-xc)**2 + (yv-yc)**2)**0.5
    mask = (rv>r) & (rv<(r+dr))
    cutoff = np.percentile(data[yv,xv][mask], 50)
    dat = np.copy(data)
    dat[dat>cutoff] = cutoff  # ignore bright pixels like stars
    return min(np.mean(data[yv,xv][mask]), np.median(data[yv,xv][mask]))

class psf():
    '''
    K. PEARSON: Interface for different PSF models
    '''
    def __init__(self,pars,psf_function):
        self.pars = pars
        self.fn = psf_function

    def eval(self,x,y):
        return self.fn(x,y,*self.pars)

    def pylint(self):
        print('pylint is stupid',self.pars)

def estimate_sigma(x,maxidx=-1):
    '''
    K. PEARSON: estimates width of PSF
    '''
    if maxidx == -1:
        maxidx = np.argmax(x)
    lower = np.abs(x-0.5*np.max(x))[:maxidx].argmin()
    upper = np.abs(x-0.5*np.max(x))[maxidx:].argmin()+maxidx
    FWHM = upper-lower
    return FWHM/(2*np.sqrt(2*np.log(2)))
