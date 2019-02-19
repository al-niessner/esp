# -- IMPORTS -- ------------------------------------------------------
import os
import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context

import excalibur
import excalibur.system.core as syscore

import numpy as np
import lmfit as lm
import scipy.interpolate as itp
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
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
def timing(force, ext, clc, out, verbose=False):
    '''
G. ROUDIER: Uses system orbital parameters to guide the dataset towards transit, eclipse or phasecurve tasks
    '''
    chunked = False
    priors = force['priors'].copy()
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')
    data = {'LOC':[], 'SCANANGLE':[], 'TIME':[], 'EXPLEN':[]}
    # LOAD DATA ------------------------------------------------------
    if ('WFC3' in ext) and ('SCAN' in ext):
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
    scanangle = np.array(data['SCANANGLE'].copy())
    exposlen = np.array(data['EXPLEN'].copy())
    ordt = np.argsort(time)
    ignto = ignore.copy()[ordt]
    scato = scanangle.copy()[ordt]
    exlto = exposlen.copy()[ordt]
    tmeto = time.copy()[ordt]
    ssc = syscore.ssconstants()
    if tmeto.size > 1:
        for p in priors['planets']:
            out['data'][p] = {}
            smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
            tmjd = priors[p]['t0']
            if tmjd > 2400000.5: tmjd -= 2400000.5
            z, phase = time2z(time, priors[p]['inc'], tmjd, smaors,
                              priors[p]['period'], priors[p]['ecc'])
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
            for v in set(visto):
                selv = (visto == v)
                trlim = 1e0
                posphsto = phsto.copy()
                posphsto[posphsto < 0] = posphsto[posphsto < 0] + 1e0
                pcconde = False
                tecrit = abs(np.arcsin(trlim/smaors))/(2e0*np.pi)
                if (np.max(posphsto[selv]) - np.min(posphsto[selv])) > (1e0 - 2e0*tecrit):
                    pcconde = True
                    pass
                pccondt = False
                if (np.max(phsto[selv]) - np.min(phsto[selv])) > (1e0 - 2e0*tecrit):
                    pccondt = True
                    pass
                if pcconde and pccondt: out['phasecurve'].append(int(v))
                select = (abs(zto[selv]) < trlim)
                if np.any(select)and(np.min(abs(posphsto[selv][select] - 0.5)) < tecrit):
                    out['eclipse'].append(int(v))
                    pass
                if np.any(select)and(np.min(abs(posphsto[selv][select])) < tecrit):
                    out['transit'].append(int(v))
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
                pcconde = False
                tecrit = abs(np.arcsin(trlim/smaors))/(2e0*np.pi)
                if (np.max(posphsto[selv]) - np.min(posphsto[selv])) > (1e0 - 2e0*tecrit):
                    pcconde = True
                    pass
                pccondt = False
                if (np.max(phsto[selv]) - np.min(phsto[selv])) > (1e0 - 2e0*tecrit):
                    pccondt = True
                    pass
                if pcconde and pccondt: out['data'][p]['phasecurve'].append(int(v))
                select = (abs(zto[selv]) < trlim)
                if np.any(select)and(np.min(abs(posphsto[selv][select] - 0.5)) < tecrit):
                    out['data'][p]['eclipse'].append(int(v))
                    pass
                if np.any(select) and (np.min(abs(posphsto[selv][select])) < tecrit):
                    out['data'][p]['transit'].append(int(v))
                    pass
                pass
            vis[ordt] = visto.astype(int)
            orb[ordt] = orbto.astype(int)
            dvis[ordt] = dvisto.astype(int)
            ignore[ordt] = ignto
            log.warning('>-- Planet: %s', p)
            log.warning('>-- Transit: %s', str(out['data'][p]['transit']))
            log.warning('>-- Eclipse: %s', str(out['data'][p]['eclipse']))
            log.warning('>-- Phase Curve: %s', str(out['data'][p]['phasecurve']))
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
        pass
    if out['transit'] or out['eclipse'] or out['phasecurve']: chunked = True
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
    # DATA CUBE --------------------------------------------------------------------------
    for index, nm in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        # MINKOWSKI FLOOD LEVEL ----------------------------------------------------------
        psdiff = np.diff(data['MEXP'][index][::-1].copy(), axis=0)
        psmin = np.nansum(data['MIN'][index])
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
    for index, nm in enumerate(data['LOC']):
        ignore = data['IGNORED'][index]
        # ISOLATE SCAN Y -----------------------------------------------------------------
        psdiff = np.diff(data['MEXP'][index][::-1].copy(), axis=0)
        psminsel = np.array(data['MIN'][index]) < 0
        if True in psminsel: psmin = np.nansum(np.array(data['MIN'][index])[psminsel])
        else: psmin = np.nanmin(data['MIN'][index])
        floatsw = data['SCANLENGTH'][index]/arcsec2pix
        if (scanwdw > psdiff[0].shape[0]) or (len(psdiff) < 2):
            scanwpi = np.round(floatsw/(len(psdiff)))
            pass
        else: scanwpi = np.round(floatsw/(len(psdiff) - 1))
        if scanwpi < 1:
            data['TRIAL'][index] = 'Subexposure Scan Length < 1 Pixel'
            ignore = True
            pass
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
                    bcke = np.nanmedian(eachcol[eachcol < np.nanpercentile(eachcol, thr)])
                    background.append(bcke)
                    pass
                background = np.array([np.array(background)]*eachdiff.shape[0])
                eachdiff -= background
                pass
            targetn = 0
            if tid in ['XO-2', 'HAT-P-1']: targetn = -1
            minlocs = []
            maxlocs = []
            fldthr = data['FLOODLVL'][index]
            for de, md in zip(psdiff.copy()[::-1], data['MIN'][index][::-1]):
                lmn, lmx = isolate(de, md, spectrace, scanwpi, targetn, fldthr)
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
                    lmn, lmx = isolate(de, md, spectrace, redscanwpi, targetn, fldthr)
                    minlocs.append(lmn)
                    maxlocs.append(lmx)
                    pass
                pass
            pass
        else:
            minlocs = [np.nan]
            maxlocs = [np.nan]
            pass
        ignore = ignore or not((np.any(np.isfinite(minlocs))) and
                               (np.any(np.isfinite(maxlocs))))
        if not ignore:
            minl = np.nanmin(minlocs)
            maxl = np.nanmax(maxlocs)
            # CONTAMINATION FROM ANOTHER SOURCE IN THE UPPER FRAME -----------------------
            if (tid in ['HAT-P-41']) and ((maxl - minl) > 15): minl = maxl - 15
            if minl < 10: minl = 10
            if maxl > (psdiff[0].shape[0] - 10): maxl = psdiff[0].shape[0] - 10
            for eachdiff in psdiff:
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
            minx, maxx = isolate(mltord, psmin, spectrace, scanwpi, targetn, fldthr,
                                 axis=0, debug=False)
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
            testspec = spectrum[np.isfinite(spectrum)]
            if (np.all(testspec[-18:] > emptythr)) and not ovszspc:
                data['IGNORED'][index] = True
                data['TRIAL'][index] = 'Truncated Spectrum'
                pass
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
        select = (mu > 2.92e3) & (mu < 5.7394e3)
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
def starecal(clc, tim, tid, flttype, out,
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
def stiscal(clc, tim, tid, flttype, out,
            verbose=False, debug=True):
    '''
R. ESTRELA: STIS .flt data extraction and wavelength calibration
    '''
    calibrated = False
    # VISIT NUMBERING --------------------------------------------------------------------
    for pkey in tim['data'].keys(): visits = np.array(tim['data'][pkey]['dvisits'])
    # OPTICS AND FILTER ------------------------------------------------------------------
    vrange = validrange(flttype)
    _wvrng, disp, ldisp, udisp = fng(flttype)
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
        frame = data['MEXP'][index].copy()
        if not ignore:
            data['SPECTRUM'][index] = np.nansum(frame, axis=0)
            data['SPECERR'][index] = np.sqrt(np.nansum(frame, axis=0))
            pass
        else:
            data['SPECTRUM'][index] = np.nansum(frame, axis=0)*np.nan
            data['SPECERR'][index] = np.nansum(frame, axis=0)*np.nan
            data['TRIAL'][index] = 'Exposure Length Outlier'
            pass
        pass
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
            data['SPECTRUM'][index] = spec
            # FIRST WAVESOL --------------------------------------------------------------
            # scaleco = np.nanmax(tt) / np.nanmin(tt[tt > 0])
            scaleco = 1e1
            if np.sum(np.isfinite(spec)) > (spec.size/2):
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
    for index, rejected in enumerate(data['IGNORED']):
        if not rejected:
            template = data['TEMPLATE'][index]
            spec = data['SPECTRUM'][index]
            temp_spec = spec/template
            ht25 = np.nanpercentile(temp_spec,25)
            lt75 = np.nanpercentile(temp_spec,75)
            selfin = np.isfinite(temp_spec)
            std1 = np.nanstd(temp_spec[selfin][(temp_spec[selfin] > ht25) & (temp_spec[selfin] < lt75)])
            # BAD PIXEL THRESHOLD --------------------------------------------------------
            bpthr = temp_spec[selfin] > np.nanmedian(temp_spec) + 3e0*std1
            temp_cut=template.copy()
            spec_cut = spec.copy()
            if True in bpthr:
                temptrash = spec_cut[selfin]
                temptrash[bpthr] = np.nan
                spec_cut[selfin] = temptrash
                temptrash = temp_cut[selfin]
                temptrash[bpthr] = np.nan
                temp_cut[selfin] = temptrash
                pass
            # SECOND SIGMA CLIPPING
            temp_spec2 = spec_cut/temp_cut
            ht25 = np.nanpercentile(temp_spec2,25)
            lt75 = np.nanpercentile(temp_spec2,75)
            selfin = np.isfinite(temp_spec2)
            std2 = np.nanstd(temp_spec2[selfin][(temp_spec2[selfin] > ht25) & (temp_spec2[selfin] < lt75)])
            bpthr2 = temp_spec2[selfin] > np.nanmedian(temp_spec2) + 5e0*std2
            spec_cut2 = spec_cut.copy()
            if True in bpthr2:
                temptrash = spec_cut2[selfin]
                temptrash[bpthr2] = np.nan
                spec_cut2[selfin] = temptrash
                pass
            data['SPECTRUM'][index] = spec_cut2
            if debug:
                plt.figure()
                plt.plot(spec_cut,'o',markersize='0.5',color='green')
                plt.plot(spec_cut2,'o',markersize='0.5',color='red')
                plt.show()
                pass
            # FIRST WAVESOL --------------------------------------------------------------
            # scaleco = np.nanmax(tt) / np.nanmin(tt[tt > 0])
            scaleco = 1e1
            if np.sum(np.isfinite(spec)) > (spec.size/2):
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
            else:
                data['IGNORED'][index] = True
                data['TRIAL'][index] = 'Not Enough Valid Points In Extracted Spectrum'
                pass
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
