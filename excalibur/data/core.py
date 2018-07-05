# -- IMPORTS -- ------------------------------------------------------
import os
import pdb

import dawgie

import excalibur.system.core as syscore

import numpy as np
import lmfit as lm
import scipy.interpolate as itp
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
# ------------- ------------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- COLLECT DATA -- -------------------------------------------------
def collect(name, scrape, out,
            threshold=0, verbose=False, debug=False):
    collected = False
    obs, ins, det, fil, mod = name.split('-')
    for rootname in scrape['name'].keys():
        oc = scrape['name'][rootname]['observatory'] in [obs.strip()]
        ic = scrape['name'][rootname]['instrument'] in [ins.strip()]
        dc = scrape['name'][rootname]['detector'] in [det.strip()]
        fc = scrape['name'][rootname]['filter'] in [fil.strip()]
        mc = scrape['name'][rootname]['mode'] in [mod.strip()]
        cond = oc and ic and dc and fc and mc
        if cond:
            out['activefilters'][name]['ROOTNAME'].append(rootname)
            sha = scrape['name'][rootname]['sha']
            md5 = scrape['name'][rootname]['md5']
            loc = md5+'_'+sha
            out['activefilters'][name]['LOC'].append(loc)            
            out['activefilters'][name]['TOTAL'].append(True)            
            collected = True
            pass
        pass
    if collected:
        if len(out['activefilters'][name]['TOTAL']) < threshold:
            out['activefilters'].pop(name, None)
            collected = False
            pass
        pass
    else: out['activefilters'].pop(name, None)
    if collected: out['STATUS'].append(True)
    return collected
# ------------------ -------------------------------------------------
# -- CALIBRATE SCAN DATA -- ------------------------------------------
def scancal(collect, tid, flttype, out,
            dtlist=['XO-2', 'WASP-19'],
            frame2png=False, verbose=False, debug=False):
    # DATA TYPE ------------------------------------------------------
    arcsec2pix = dps(flttype)
    vrange = validrange(flttype)
    wvrng, disper, ldisp, udisp = fng(flttype)
    spectrace = np.round((np.max(wvrng) - np.min(wvrng))/disper)
    # LOAD DATA ------------------------------------------------------
    dbs = os.path.join(dawgie.context.data_dbs, 'mast')
    data = {'LOC':[], 'EPS':[], 'DISPLIM':[ldisp, udisp], 
            'SCANRATE':[], 'SCANLENGTH':[], 'SCANANGLE':[],
            'EXP':[], 'EXPERR':[], 'EXPFLAG':[], 'VRANGE':vrange,
            'TIME':[], 'EXPLEN':[], 'MIN':[], 'MAX':[], 'TRIAL':[]}
    for loc in sorted(collect['LOC']):
        fullloc = os.path.join(dbs, loc)
        with pyfits.open(fullloc) as hdulist:
            header0 = hdulist[0].header
            test = header0['UNITCORR']
            eps = False
            if (test in ['COMPLETE','PERFORM']): eps = True
            data['EPS'].append(eps)
            if 'SCAN_RAT' in header0:
                data['SCANRATE'].append(header0['SCAN_RAT'])
                pass
            else: data['SCANRATE'].append(np.nan)
            if 'SCAN_LEN' in header0:
                data['SCANLENGTH'].append(header0['SCAN_LEN'])
                pass
            else: data['SCANLENGTH'].append(np.nan)
            if 'SCAN_ANG' in header0:
                data['SCANANGLE'].append(header0['SCAN_ANG'])
                pass
            elif 'PA_V3' in header0:
                data['SCANANGLE'].append(header0['PA_V3'])
                pass
            else: scanang.append(np.nan)
            pass
            frame = []
            errframe = []
            dqframe = []
            ftime = []
            fmin = []
            fmax = []
            for fits in hdulist:
                if ((fits.size != 0) and
                    ('DELTATIM' in fits.header.keys())):
                    fitsdata = np.empty(fits.data.shape)
                    fitsdata[:] = fits.data[:]
                    frame.append(fitsdata)
                    ftime.append(float(fits.header['ROUTTIME']))
                    fmin.append(float(fits.header['GOODMIN']))
                    fmax.append(float(fits.header['GOODMAX']))
                    del fits.data
                    pass
                if ('EXTNAME' in fits.header):
                    if (fits.header['EXTNAME'] == 'ERR'):
                        fitsdata = np.empty(fits.data.shape)
                        fitsdata[:] = fits.data[:]
                        errframe.append(fitsdata)
                        del fits.data
                        pass
                    pass
                if (eps and ('EXTNAME' in fits.header)):
                    if (fits.header['EXTNAME'] == 'TIME'):
                        frame[-1] = (frame[-1]*
                                     np.array(float(
                                         fits.header['PIXVALUE'])))
                        errframe[-1] = (errframe[-1]*
                                        np.array(float(
                                            fits.header['PIXVALUE'])))
                        pass
                    pass
                if ('EXTNAME' in fits.header):
                    if (fits.header['EXTNAME'] == 'DQ'):
                        fitsdata = np.empty(fits.data.shape)
                        fitsdata[:] = fits.data[:]
                        dqframe.append(fitsdata)
                        del fits.data
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
    # MASK DATA ------------------------------------------------------    
    data['MEXP'] = data['EXP'].copy()
    data['MASK'] = data['EXPFLAG'].copy()
    data['IGNORED'] = [False]*len(data['LOC'])
    data['FLOODLVL'] = [np.nan]*len(data['LOC'])
    data['TRIAL'] = ['']*len(data['LOC'])
    for nm in data['LOC']:
        index = data['LOC'].index(nm)
        maskedexp = []
        masks = []
        ignore = False
        for dd, ff in zip(data['EXP'][index].copy(),
                          data['EXPFLAG'][index].copy()):
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
    # DATA CUBE ------------------------------------------------------
    for nm in data['LOC']:
        index = data['LOC'].index(nm)
        ignore = data['IGNORED'][index]
        # ISOLATE SCAN Y ---------------------------------------------
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
        if not(ignore):
            targetn = 0
            if tid in dtlist:
                if tid in ['XO-2']: targetn = -1
                pass
            minlocs = []
            maxlocs = []
            floodlist = []
            for de, md in zip(psdiff.copy()[::-1],
                              data['MIN'][index][::-1]):
                valid = np.isfinite(de)
                if np.nansum(~valid) > 0: de[~valid] = 0
                select = de[valid] < md
                if np.nansum(select) > 0: de[valid][select] = 0
                sqplate = de.shape[0]*de.shape[1]
                srcprct = spectrace*scanwpi/sqplate
                if tid in dtlist: srcprct *= 2
                fldlvl = np.nanpercentile(de, 1e2*(1e0 - srcprct))
                floodlist.append(fldlvl)
                pass
            fldthr = np.nanmedian(floodlist)
            for de, md in zip(psdiff.copy()[::-1],
                              data['MIN'][index][::-1]):
                lmn, lmx = isolate(de, md, spectrace, scanwpi,
                                   targetn, tid, dtlist, fldthr,
                                   verbose=verbose, debug=debug)
                minlocs.append(lmn)
                maxlocs.append(lmx)
                pass
            data['FLOODLVL'][index] = fldthr
            pass
        else:
            minlocs = [np.nan]
            maxlocs = [np.nan]
            pass
        ignore = ignore or not((np.any(np.isfinite(minlocs))) and
                               (np.any(np.isfinite(maxlocs))))
        if not(ignore):
            minl = np.nanmin(minlocs)
            maxl = np.nanmax(maxlocs)
            if minl < 0: minl = 10
            if maxl > (psdiff[0].shape[0] - 1): maxl = -10
            for eachdiff in psdiff:
                eachdiff[:int(minl),:] = 0
                eachdiff[int(maxl):,:] = 0            
                pass
            # DIFF ACCUM ---------------------------------------------
            thispstamp = np.nansum(psdiff, axis=0)
            thispstamp[thispstamp <= psmin] = np.nan
            thispstamp[thispstamp == 0] = np.nan
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
                plt.title('Background Sub')
                plt.imshow(thispstamp)
                plt.colorbar()
                plt.clim(colorlim, abs(colorlim))
                plt.show()
                pass
            # ISOLATE SCAN X -----------------------------------------
            mltord = thispstamp.copy()
            targetn = 0
            if tid in dtlist:
                if tid in ['XO-2']: targetn = -1
                pass
            minx, maxx = isolate(mltord, psmin, spectrace, scanwpi,
                                 targetn, tid, dtlist, fldthr,
                                 axis=0, verbose=verbose, debug=debug)
            if np.isfinite(minx*maxx):
                minx -= 1.5*12
                maxx += 1.5*12
                if minx < 0: minx = 10
                if maxx > (thispstamp.shape[1] - 1): maxx = -10
                if (maxx - minx) < spectrace:
                    data['TRIAL'][index] = 'Could Not Find Full Spectrum'
                    ignore = True
                    pass
                thispstamp[:,:int(minx)] = np.nan
                thispstamp[:,int(maxx):] = np.nan
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
            data['TRIAL'][index] = 'Could Not Find Y Edges'
            ignore = True
            pass
        data['MEXP'][index] = thispstamp
        data['TIME'][index] = np.nanmean(data['TIME'][index].copy())
        data['IGNORED'][index] = ignore
        data['EXPERR'][index] = pstamperr
        # PLOTS ------------------------------------------------------
        if debug or frame2png:
            plt.figure()
            plt.title('Ignored = '+str(ignore))
            plt.imshow(thispstamp)
            plt.colorbar()
            if frame2png:
                if not(os.path.exists('TEST')): os.mkdir('TEST')
                if not(os.path.exists('TEST/'+tid)):
                    os.mkdir('TEST/'+tid)
                    pass
                fname = 'TEST/'+tid+'/'+nm+'.png'
                plt.savefig(fname)
                plt.close()
                pass
            else: plt.show()
            pass
        pass
    # SPECTRUM EXTRACTION --------------------------------------------
    data['SPECTRUM'] = [np.nan]*len(data['LOC'])
    data['SPECERR'] = [np.nan]*len(data['LOC'])
    data['NSPEC'] = [np.nan]*len(data['LOC'])
    for loc in data['LOC']:
        index = data['LOC'].index(loc)
        ignore =  data['IGNORED'][index]
        floodlevel = data['FLOODLVL'][index]
        if not(ignore):
            frame = data['MEXP'][index].copy()
            frame = [line for line in frame if
                     not(np.all(~np.isfinite(line)))]
            # OVERSIZED MASK -----------------------------------------
            for line in frame:
                if np.nanmax(line) < floodlevel: line *= np.nan
                pass
            frame = [line for line in frame if
                     not(np.all(~np.isfinite(line)))]
            # SCAN RATE CORRECTION -----------------------------------
            template = []
            for col in np.array(frame).T:
                if not(np.all(~np.isfinite(col))):
                    template.append(np.nanmedian(col))
                    pass
                else: template.append(np.nan)
                pass
            template = np.array(template)
            for line in frame:                
                errref = np.sqrt(abs(line))/abs(template)
                line /= template
                refline = np.nanmedian(line)
                select = np.isfinite(line)
                minok = (abs(line[select] - refline) <
                         3*np.nanmin(errref[select]))
                ok = (abs(line[select] - refline) <
                      3*errref[select])
                if np.nansum(minok) > 0:
                    alpha = (np.nansum(minok)/
                             np.nansum(line[select][minok]))
                    line *= alpha
                    line *= template
                    line[select][~ok] = np.nan
                    pass
                else: line *= np.nan
                pass
            frame = [line for line in frame if
                     not(np.all(~np.isfinite(line)))]
            spectrum = []
            specerr = []
            nspectrum = []
            vtemplate = []
            for row in np.array(frame):
                if not(np.all(~np.isfinite(row))):
                    vtemplate.append(np.nanmedian(row))
                    pass
                else: vtemplate.append(np.nan)
                pass
            vtemplate = np.array(vtemplate)
            for col in np.array(frame).T:
                ignorecol = False
                if not(np.all(~np.isfinite(col))):
                    errref = (np.sqrt(abs(np.nanmedian(col)))/
                              abs(vtemplate))
                    ratio = col/vtemplate
                    refline = np.nanmedian(ratio)
                    select = np.isfinite(col)
                    ok = (abs(ratio[select] - refline) <
                          3e0*errref[select])
                    if np.nansum(ok) > 0:
                        alpha = (np.nansum(ok)/
                                 np.nansum(ratio[select][ok]))
                        valid = (abs(col[select]*alpha -
                                     vtemplate[select]) <
                                 3*np.sqrt(abs(vtemplate[select])))
                        pass
                    else: valid = [False]
                    if np.nansum(valid) > 0:
                        spectrum.append(np.mean(col[select][valid]))
                        specerr.append(np.std(col[select][valid]))
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
            spectrum = np.array(spectrum) - np.nanmin(spectrum)
            data['SPECTRUM'][index] = np.array(spectrum)
            data['SPECERR'][index] = np.array(specerr)
            data['NSPEC'][index] = np.array(nspectrum)
            pass            
        pass
    # PLOT -----------------------------------------------------------
    if debug:
        plt.figure()
        for spec in data['SPECTRUM']: plt.plot(spec)
        plt.ylabel('Stellar Spectra [Counts]')
        plt.xlabel('Pixel Number')
        plt.show()
        pass
    # WAVELENGTH CALIBRATION -----------------------------------------
    wavett, tt = ag2ttf(flttype, verbose=verbose, debug=debug)
    scaleco = np.nanmax(tt) / np.nanmin(tt[tt > 0])
    data['PHT2CNT'] = [np.nan]*len(data['LOC'])
    data['WAVE'] = [np.nan]*len(data['LOC']) 
    data['DISPERSION'] = [np.nan]*len(data['LOC']) 
    data['SHIFT'] = [np.nan]*len(data['LOC'])
    spectralindex = []
    for loc in data['LOC']:
        index = data['LOC'].index(loc)
        ignore =  data['IGNORED'][index]
        if not ignore:
            spectrum = data['SPECTRUM'][index].copy()
            cutoff = np.nanmax(spectrum)/scaleco
            spectrum[spectrum < cutoff] = np.nan
            spectrum = abs(spectrum)
            w, d, s, si = wavesol(spectrum, tt, wavett, disper,
                                  verbose=False, debug=False)
            if (d > ldisp) and (d < udisp): spectralindex.append(si)
            pass
        pass
    siv = np.nanmedian(spectralindex)
    for loc in data['LOC']:
        index = data['LOC'].index(loc)
        ignore =  data['IGNORED'][index]
        if not ignore:
            spectrum = data['SPECTRUM'][index].copy()
            cutoff = np.nanmax(spectrum)/scaleco
            spectrum[spectrum < cutoff] = np.nan
            spectrum = abs(spectrum)
            wave, disp, shift, si = wavesol(spectrum, tt, wavett,
                                            disper, siv=siv,
                                            verbose=verbose,
                                            debug=debug)
            if (disp < ldisp) or (disp > udisp):
                data['TRIAL'][index] = 'Dispersion Out Of Bounds'
                ignore = True
                pass
            if abs(disp - disper) < 1e-5:
                data['TRIAL'][index] = 'Dispersion Fit Failure'
                ignore = True
                pass
            pass
        if not ignore:
            liref = itp.interp1d(wavett*1e-4, tt,
                                 bounds_error=False,
                                 fill_value=np.nan)
            phot2counts = liref(wave)
            data['PHT2CNT'][index] = phot2counts
            data['WAVE'][index] = wave # MICRONS
            data['DISPERSION'][index] = disp # ANGSTROMS/PIXEL
            data['SHIFT'][index] = shift*1e4/disp # PIXELS
            pass
        data['IGNORED'][index] = ignore
        pass
    # PLOTS ----------------------------------------------------------
    if verbose:
        timing = np.array([d for d,i in zip(data['TIME'],
                                            data['IGNORED'])
                           if not(i)])
        dispersion = np.array([d for d,i in zip(data['DISPERSION'],
                                                data['IGNORED'])
                               if not(i)])
        shift = np.array([d for d,i in zip(data['SHIFT'],
                                           data['IGNORED'])
                          if not(i)])
        spec = np.array([d for d,i in zip(data['SPECTRUM'],
                                          data['IGNORED'])
                         if not(i)])
        photoc = np.array([d for d,i in zip(data['PHT2CNT'],
                                            data['IGNORED'])
                           if not(i)])
        wave = np.array([d for d,i in zip(data['WAVE'],
                                          data['IGNORED'])
                         if not(i)])
        errspec = np.array([d for d,i in zip(data['SPECERR'],
                                             data['IGNORED'])
                            if not(i)])
        torder = np.argsort(timing)
        allignore = data['IGNORED']
        allculprits = data['TRIAL']
        allindex = np.arange(len(data['LOC']))
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
        plt.xlabel('Wavelength [$\mu$m]')

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

        print('>-- IGNORED:', np.nansum(allignore),
              '/', len(allignore))
        for index in allindex:
            if len(allculprits[index]) > 0:
                print('Frame: ', index, allculprits[index])
                pass
            pass
        pass
    data.pop('EXP', None)
    data.pop('EXPFLAG', None)
    for key in data: out['data'][key] = data[key]        
    caled = not(np.all(data['IGNORED']))
    if caled: out['STATUS'].append(True)
    return caled
# ------------------------- ------------------------------------------
# -- DETECTOR PLATE SCALE -- -----------------------------------------
def dps(flttype):
    '''
http://www.stsci.edu/hst/wfc3/ins_performance/detectors
    '''
    detector = flttype.split('-')[2]
    arcsec2pix = None
    if detector == 'IR': arcsec2pix = 0.13 
    if detector == 'UVIS': arcsec2pix = 0.04
    return arcsec2pix
# --------------------------------------------------------------------
# -- DETECTOR PLATE SCALE -- -----------------------------------------
def validrange(flttype):
    '''
G102 http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-18.pdf
G141 http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-17.pdf
    '''
    fltr = flttype.split('-')[3]
    vrange = None
    if fltr == 'G141': vrange=[1.10, 1.65]
    if fltr == 'G102': vrange=[0.8, 1.14]    
    return vrange
# --------------------------------------------------------------------
# -- FILTERS AND GRISMS -- -------------------------------------------
def fng(flttype):
    '''
http://www.stsci.edu/hst/wfc3/documents/handbooks/currentDHB/wfc3_dhb.pdf
G102 http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-18.pdf
G141 http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-17.pdf
    '''
    fltr = flttype.split('-')[3]
    wvrng = None
    disp = None
    if fltr == 'G141':
        wvrng = [1085e1, 17e3] # Angstroms
        disp = 46.5 # Angstroms/Pixel
        llim = 45
        ulim = 47.5
        pass
    if fltr == 'G102':
        wvrng = [8e3, 115e2] # Angstroms
        disp = 24.5 # Angstroms/Pixel
        llim = 23.5
        ulim = 25
        pass
    return wvrng, disp, llim, ulim
# ------------------------ -------------------------------------------
# -- ISOLATE -- ------------------------------------------------------
def isolate(thisdiff, psmin, spectrace, scanwdw,
            targetn, tid, dtlist, floodlevel,
            axis=1, verbose=False, debug=False):
    '''
G ROUDIER: Based on Minkowski functionnals decomposition algorithm
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
    if len(loc) > 0:
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
            if ((dl > thr) and not((cvcount > thrw))):
                poop = minlocs.pop(-1)
                minlocs.append(loc[c])
                pass
            if ((dl > thr) and (cvcount > thrw)):
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

        plt.figure()
        plt.title('Isolation Level')
        plt.plot(diffloc)
        plt.plot(np.array(diffloc)*0 + thr)
        plt.show()
        pass
    return mn, mx
# ------------- ------------------------------------------------------
# -- APERTURE AND FILTER TO TOTAL TRANSMISSION FILTER -- -------------
def ag2ttf(flttype, verbose=False, debug=False):
    detector = flttype.split('-')[2]
    grism = flttype.split('-')[3]
    wvrng, disp, llim, ulim = fng(flttype)
    lightpath = ag2lp(detector, grism)
    mu, ttp = bttf(lightpath, verbose=verbose, debug=debug)
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
    return mu, ttp
# ------------------------------------------------------ -------------
# -- APERTURE AND GRISM TO .FITS FILES -- ----------------------------
def ag2lp(detector, grism):
    '''
http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2011-05.pdf
ftp://ftp.stsci.edu/cdbs/comp/ota/
ftp://ftp.stsci.edu/cdbs/comp/wfc3/

G ROUDIER: The first element of the returned list defines the default
interpolation grid, filter/grism file is suited.
    '''
    lightpath = []
    if grism == 'G141':
        lightpath.append('WFC3/wfc3_ir_g141_src_004_syn.fits')
        pass
    if grism == 'G102':
        lightpath.append('WFC3/wfc3_ir_g102_src_003_syn.fits')
        pass
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
    references = ['Grism source',
                  'Refractive correction plate',
                  'Cold mask',
                  'Mirror 1',
                  'Mirror 2',
                  'Fold mirror',
                  'Channel select mechanism',
                  'Pick off mirror',
                  'OTA']
    return lightpath
# --------------------------------------- ----------------------------
# -- BUILD TOTAL TRANSMISSION FILTER -- ------------------------------
def bttf(lightpath, verbose=False, debug=False):
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
        plt.ylim([0.5, 1])
        plt.ylabel('Transmission Curves')
        
        plt.subplot(212)
        plt.plot(muref, ttp, 'o--')
        plt.xlabel('$\lambda$ [Angstroms]')
        plt.ylabel('Total Throughput')
        plt.xlim([min(muref), max(muref)])
        plt.show()
        pass
    return muref, ttp
# ------------------------------------- ------------------------------
# -- WFC3 CAL FITS -- ------------------------------------------------
def loadcalf(name, muref, calloc='/proj/sdp/data/cal'):
    fitsfile = os.path.join(calloc, name)
    data = pyfits.getdata(fitsfile)
    muin = np.array(data.WAVELENGTH)
    tin = np.array(data.THROUGHPUT)
    if (False in np.isfinite(muref)): muref = muin
    f = itp.interp1d(muin, tin, bounds_error=False, fill_value=0)
    t = f(muref)
    return muref, t
# ------------------- ------------------------------------------------
# -- WAVELENGTH SOLUTION -- ------------------------------------------
def wavesol(spectrum, tt, wavett, disper, siv=None,
            verbose=False, debug=False):
    '''
G ROUDIER: Wavelength calibration on log10 spectrum to emphasize the
edges, approximating the log(stellar spectrum) with a linear model
    '''
    mutt = wavett.copy()
    mutt /= 1e4
    xdata = np.arange(spectrum.size)
    select = (tt == 0)
    tt[select] = np.nan    
    select = np.isfinite(spectrum)
    logspec = np.log10(spectrum)
    logtt = np.log10(tt)
    wave = xdata*disper/1e4
    minwave = np.nanmin(wave[select])
    select = np.isfinite(tt)
    reftt = np.nanmin(mutt[select])
    shift = reftt - minwave
    scale = np.nanmedian(logspec) - np.nanmedian(logtt)
    params = lm.Parameters()
    if siv is None: params.add('slope', value=1e-2)
    else: params.add('slope', value=siv, vary=False)
    params.add('scale', value=scale)
    params.add('disper', value=disper)
    params.add('shift', value=shift)
    out = lm.minimize(wcme, params,
                      args=(logspec, mutt, logtt,False))
    disper = out.params['disper'].value
    shift = out.params['shift'].value
    slope = out.params['slope'].value
    scale = out.params['scale'].value
    wave = wcme(out.params, logspec, refmu=mutt, reftt=logtt)
    # PLOTS
    if debug:
        plt.figure()
        plt.plot(mutt, logtt, 'o--')
        plt.plot(wave, logspec - (scale+wave*slope), 'o--')
        plt.show()
        pass
    return wave, disper, shift+minwave, slope
# ------------------------- ------------------------------------------
# -- WAVELENGTH FIT FUNCTION -- --------------------------------------
def wcme(params, data, refmu=None, reftt=None, forward=True):
    slope = params['slope'].value
    scale = params['scale'].value
    disper = params['disper'].value
    shift = params['shift'].value
    liref = itp.interp1d(refmu, reftt,
                         bounds_error=False, fill_value=np.nan)
    wave = np.arange(data.size)*disper*1e-4 + shift
    model = liref(wave) + scale + slope*wave
    select = (np.isfinite(model)) & (np.isfinite(data))
    d = data.copy()
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
# -- TIMING -- -------------------------------------------------------
def timing(force, cal, out, verbose=False, debug=False):
    chunked = False
    priors = force['priors'].copy()
    planets = priors['planets']
    time = np.array(cal['data']['TIME'].copy())
    ignore = np.array(cal['data']['IGNORED'].copy())
    scanangle = np.array(cal['data']['SCANANGLE'].copy())
    exposlen = np.array(cal['data']['EXPLEN'].copy())
    ordt = np.argsort(time)
    ignto = ignore.copy()[ordt]
    scato = scanangle.copy()[ordt]
    exlto = exposlen.copy()[ordt]
    tmeto = time.copy()[ordt]
    ssc = syscore.ssconstants()
    for p in planets:
        out['data'][p] = {}
        smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
        rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
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
        thro = np.percentile(tmetod[tmetod > 3*thrs], 75)
        # VISIT NUMBERING --------------------------------------------
        whereo = np.where(tmetod > 3*thrs)[0]
        wherev = np.where(tmetod > 3*thro)[0]
        visto = np.ones(tmetod.size)
        dvis = np.ones(tmetod.size)
        vis = np.ones(tmetod.size)
        for index in wherev: visto[index:] += 1        
        # DOUBLE SCAN VISIT RE NUMBERING -----------------------------
        dvisto = visto.copy()
        for v in set(visto):
            selv = (visto == v)
            vordsa = scato[selv].copy()
            if (len(set(vordsa)) > 1):
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
                select = np.where(tmetod[selv] > 3*thrs)[0]
                incorb = orbto[selv]
                for indice in select:
                    incorb[indice:] = incorb[indice:] + 1
                    pass
                orbto[selv] = incorb
                for o in set(orbto[selv]):
                    selo = (orbto[selv] == o)
                    if len(~ignto[selv][selo]) < 4:
                        ignto[selv][selo] = True
                        pass
                    ref = np.median(exlto[selv][selo])
                    if len(set(exlto[selv][selo])) > 1:
                        rej = (exlto[selv][selo] != ref)
                        ignto[selv][selo][rej] = True
                        pass
                    pass
                pass
            pass
        # TRANSIT VISIT PHASECURVE -----------------------------------
        for v in set(visto):
            selv = (visto == v)
            trlim = 1e0 - rpors
            posphsto = phsto.copy()
            posphsto[posphsto < 0] = posphsto[posphsto < 0] + 1e0
            pcconde = False
            if ((np.max(posphsto[selv]) - np.min(posphsto[selv])) >
                (1e0 - 2e0*abs(np.arcsin(trlim/smaors))/(2e0*np.pi))):
                pcconde = True
                pass
            pccondt = False
            if ((np.max(phsto[selv]) - np.min(phsto[selv])) >
                (1e0 - 2e0*abs(np.arcsin(trlim/smaors))/(2e0*np.pi))):
                pccondt = True
                pass
            if pcconde and pccondt: out['phasecurve'].append(int(v))
            select = (abs(zto[selv]) < trlim)
            if (np.any(select) and
                (np.min(abs(posphsto[selv][select] - 0.5)) <
                 abs(np.arcsin(trlim/smaors))/(2e0*np.pi))):
                out['eclipse'].append(int(v))
                pass
            if (np.any(select) and
                (np.min(abs(posphsto[selv][select])) <
                 abs(np.arcsin(trlim/smaors))/(2e0*np.pi))):
                out['transit'].append(int(v))
                pass
            pass
        out['data'][p]['transit'] = []
        out['data'][p]['eclipse'] = []
        out['data'][p]['phasecurve'] = []        
        for v in set(dvisto):
            selv = (dvisto == v)
            trlim = 1e0 - rpors
            posphsto = phsto.copy()
            posphsto[posphsto < 0] = posphsto[posphsto < 0] + 1e0
            pcconde = False
            if ((np.max(posphsto[selv]) - np.min(posphsto[selv])) >
                (1e0 - 2e0*abs(np.arcsin(trlim/smaors))/(2e0*np.pi))):
                pcconde = True
                pass
            pccondt = False
            if ((np.max(phsto[selv]) - np.min(phsto[selv])) >
                (1e0 - 2e0*abs(np.arcsin(trlim/smaors))/(2e0*np.pi))):
                pccondt = True
                pass
            if pcconde and pccondt: out['data'][p]['phasecurve'].append(int(v))
            select = (abs(zto[selv]) < trlim)
            if (np.any(select) and
                (np.min(abs(posphsto[selv][select] - 0.5)) <
                 abs(np.arcsin(trlim/smaors))/(2e0*np.pi))):
                out['data'][p]['eclipse'].append(int(v))
                pass
            if (np.any(select) and
                (np.min(abs(posphsto[selv][select])) <
                 abs(np.arcsin(trlim/smaors))/(2e0*np.pi))):
                out['data'][p]['transit'].append(int(v))
                pass
            pass
        vis[ordt] = visto.astype(int)
        orb[ordt] = orbto.astype(int)
        dvis[ordt] = dvisto.astype(int)
        ignore[ordt] = ignto
        # PLOTS ------------------------------------------------------
        if verbose:
            print('>-- TRANSIT:', out['transit'])
            print('>-- ECLIPSE:', out['eclipse'])
            print('>-- PHASE CURVE:', out['phasecurve'])
            
            plt.figure()
            plt.plot(phsto, 'k.')
            plt.plot(np.arange(phsto.size)[~ignto],
                     phsto[~ignto], 'bo')
            for i in wherev: plt.axvline(i, ls='--', color='r')
            for i in whereo: plt.axvline(i, ls='-.', color='g')
            plt.xlim(0, tmetod.size - 1)
            plt.ylim(-0.5, 0.5)
            plt.xlabel('Time index')
            plt.ylabel('Orbital Phase [2pi rad]')

            plt.figure()
            plt.plot(tmetod, 'o')
            plt.plot(tmetod*0+3*thro, 'r--')
            plt.plot(tmetod*0+3*thrs, 'g-.')
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
        out['data'][p]['thrs'] = thrs
        out['data'][p]['thro'] = thro
        out['data'][p]['visits'] = vis
        out['data'][p]['orbits'] = orb
        out['data'][p]['dvisits'] = dvis
        out['data'][p]['z'] = z
        out['data'][p]['phase'] = phase
        out['data'][p]['ordt'] = ordt
        out['data'][p]['ignore'] = ignore        
        out['STATUS'].append(True)        
        pass
    if ((len(out['transit']) > 0) or
        (len(out['eclipse']) > 0) or
        (len(out['phasecurve']) > 0)): chunked = True    
    return chunked
# ------------ -------------------------------------------------------
# -- TIME TO Z -- ----------------------------------------------------
def time2z(time, ipct, tknot, sma, orbperiod, ecc,
           tperi=None, epsilon=1e-10):
    ipctrad = ipct*np.pi/180e0
    if (tperi is not None):
        ft0 = (tperi - tknot)%orbperiod
        ft0 /= orbperiod
        if (ft0 > 0.5): ft0 += -1e0
        M0 = 2e0*np.pi*ft0
        E0 = solveme(M0, ecc, epsilon)
        realf = np.sqrt(1e0 - ecc)*np.cos(E0/2e0)
        imagf = np.sqrt(1e0 + ecc)*np.sin(E0/2e0)
        w = np.angle(np.complex(realf, imagf))
        if (abs(ft0) < epsilon):
            w = np.pi/2e0
            tperi = tknot
            pass
        pass
    else:
        w = np.pi/2e0
        tperi = tknot
        pass
    ft = (time - tperi)%orbperiod
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
    z = r*np.sqrt(1e0**2 - (np.sin(w+f)**2)*(np.sin(ipctrad))**2)
    z[sft < 0] *= -1e0
    return z, sft
# --------------- ----------------------------------------------------
# -- TRUE ANOMALY NEWTON RAPHSON SOLVER -- ---------------------------
def solveme(M, e, eps):
    E = np.copy(M)
    for i in np.arange(M.shape[0]):
        while(abs(E[i] - e*np.sin(E[i]) - M[i]) > eps):
            num = E[i] - e*np.sin(E[i]) - M[i]
            den = 1. - e*np.cos(E[i])
            E[i] = E[i] - num/den
            pass
        pass
    return E
# ---------------------------------------- ---------------------------
