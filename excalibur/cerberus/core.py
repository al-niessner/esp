# -- IMPORTS -- ------------------------------------------------------
import excalibur
# pylint: disable=import-self
import excalibur.cerberus.core
import excalibur.system.core as syscore

import logging
log = logging.getLogger(__name__)
pymc3log = logging.getLogger('pymc3')
pymc3log.setLevel(logging.ERROR)

import os
import pymc3 as pm
import numpy as np
# import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

import theano.tensor as tt
import theano.compile.ops as tco

import scipy.constants as cst
from scipy.interpolate import interp1d as itp
# -- GLOBAL CONTEXT FOR PYMC3 DETERMINISTICS ---------------------------------------------
from collections import namedtuple
CONTEXT = namedtuple('CONTEXT', ['cleanup', 'model', 'p', 'solidr', 'orbp', 'tspectrum',
                                 'xsl', 'spc', 'modparlbl', 'hzlib'])
ctxt = CONTEXT(cleanup=None, model=None, p=None, solidr=None, orbp=None, tspectrum=None,
               xsl=None, spc=None, modparlbl=None, hzlib=None)
def ctxtupdt(cleanup=None, model=None, p=None, solidr=None, orbp=None, tspectrum=None,
             xsl=None, spc=None, modparlbl=None, hzlib=None):
    '''
G. ROUDIER: Update context
    '''
    excalibur.cerberus.core.ctxt = CONTEXT(cleanup=cleanup, model=model, p=p,
                                           solidr=solidr, orbp=orbp,
                                           tspectrum=tspectrum, xsl=xsl, spc=spc,
                                           modparlbl=modparlbl, hzlib=hzlib)
    return
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
# -- X SECTIONS LIBRARY -- -------------------------------------------
def myxsecsversion():
    '''
    Alya Al-Kibbi:111
    Changed CH4 line list to HITEMP
    Alya Al-Kibbi:112
    Built interpolator on cross sections assuming constant broadening and
    shifting effects (like exomol)
    Done to speed up processing of CH4 HITEMP line list
    '''
    import dawgie
    return dawgie.VERSION(1,1,3)

def myxsecs(spc, out,
            hitemp=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HITEMP'),
            tips=os.path.join(excalibur.context['data_dir'], 'CERBERUS/TIPS'),
            ciadir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HITRAN/CIA'),
            exomoldir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/EXOMOL'),
            knownspecies=['NO', 'CH4','OH', 'C2H2', 'N2', 'N2O', 'O3', 'O2'].copy(),
            cialist=['H2-H', 'H2-H2', 'H2-He', 'He-H'].copy(),
            xmspecies=['TIO', 'H2O', 'H2CO', 'HCN', 'CO', 'CO2', 'NH3'].copy(),
            verbose=False):
    '''
G. ROUDIER: Builds Cerberus cross section library
    '''
    cs = False
    for p in spc['data'].keys():
        out['data'][p] = {}
        wgrid = np.array(spc['data'][p]['WB'])
#         cond_grid = (wgrid < 0.56) | (wgrid > 1.02)
        qtgrid = gettpf(tips, knownspecies)
        library = {}
        nugrid = (1e4/np.copy(wgrid))[::-1]
        dwnu = np.concatenate((np.array([np.diff(nugrid)[0]]), np.diff(nugrid)))
        for myexomol in xmspecies:
            log.warning('>-- %s', str(myexomol))
            library[myexomol] = {'I':[], 'nu':[], 'T':[],
                                 'Itemp':[], 'nutemp':[], 'Ttemp':[],
                                 'SPL':[], 'SPLNU':[]}
            thisxmdir = os.path.join(exomoldir, myexomol)
            myfiles = [f for f in os.listdir(thisxmdir) if f.endswith('K')]
            for mf in myfiles:
                xmtemp = float(mf.split('K')[0])
                with open(os.path.join(thisxmdir, mf), 'r') as fp:
                    data = fp.readlines()
                    fp.close()
                    for line in data:
                        line = np.array(line.split(' '))
                        line = line[line != '']
                        library[myexomol]['nutemp'].append(float(line[0]))
                        library[myexomol]['Itemp'].append(float(line[1]))
                        library[myexomol]['Ttemp'].append(xmtemp)
                        pass
                    pass
                pass
            for mytemp in set(library[myexomol]['Ttemp']):
                select = np.array(library[myexomol]['Ttemp']) == mytemp
                bini = []
                matnu = np.array(library[myexomol]['nutemp'])[select]
                sigma2 = np.array(library[myexomol]['Itemp'])[select]
                for nubin, mydw in zip(nugrid, dwnu):
                    select = ((matnu > (nubin-mydw/2.)) & (matnu <= nubin+mydw/2.))
                    bini.append(np.sum(sigma2[select]))
                    pass
                bini = np.array(bini)/dwnu
                library[myexomol]['nu'].extend(list(nugrid))
                library[myexomol]['I'].extend(list(bini))
                library[myexomol]['T'].extend(list(np.ones(nugrid.size)*mytemp))
                pass
            for iline in set(library[myexomol]['nu']):
                select = np.array(library[myexomol]['nu']) == iline
                y = np.array(library[myexomol]['I'])[select]
                x = np.array(library[myexomol]['T'])[select]
                sortme = np.argsort(x)
                x = x[sortme]
                y = y[sortme]
                myspl = itp(x, y, bounds_error=False, fill_value=0)
                library[myexomol]['SPL'].append(myspl)
                library[myexomol]['SPLNU'].append(iline)
                if verbose:
                    plt.plot(x, y, 'o')
                    xp = np.arange(101)/100.*(3000. - np.min(x))+np.min(x)
                    plt.plot(xp, myspl(xp))
                    plt.show()
                    pass
                pass
            if verbose:
                fts = 20
                plt.figure(figsize=(16,12))
                haha = [huhu for huhu in set(library[myexomol]['T'])]
                haha = np.sort(np.array(haha))
                haha = haha[::-1]
                for temp in haha:
                    select = np.array(library[myexomol]['T']) == temp
                    plt.semilogy(1e4/(np.array(library[myexomol]['nu'])[select]),
                                 np.array(library[myexomol]['I'])[select],
                                 label=str(int(temp))+'K')
                    pass
                plt.title(myexomol)
                plt.xlabel('Wavelength $\\lambda$[$\\mu m$]',
                           fontsize=fts+4)
                plt.ylabel('Cross Section [$cm^{2}.molecule^{-1}$]',
                           fontsize=fts+4)
                plt.tick_params(axis='both', labelsize=fts)
                plt.legend(bbox_to_anchor=(0.95, 0., 0.12, 1),
                           loc=5, ncol=1, mode='expand', numpoints=1,
                           borderaxespad=0., frameon=True)
                plt.savefig(myexomol+'_xslib.png', dpi=200)
                plt.show()
                pass
            pass
        for mycia in cialist:
            log.warning('>-- %s', str(mycia))
            myfile = '_'.join((os.path.join(ciadir, mycia), '2011.cia'))
            library[mycia] = {'I':[], 'nu':[], 'T':[],
                              'Itemp':[], 'nutemp':[], 'Ttemp':[],
                              'SPL':[], 'SPLNU':[]}
            with open(myfile, 'r') as fp:
                data = fp.readlines()
                fp.close()
                # Richard et Al. 2012
                for line in data:
                    line = np.array(line.split(' '))
                    line = line[line != '']
                    if line.size > 2: tmprtr = float(line[4])
                    if line.size == 2:
                        library[mycia]['nutemp'].append(float(line[0]))
                        library[mycia]['Itemp'].append(float(line[1]))
                        library[mycia]['Ttemp'].append(tmprtr)
                        pass
                    pass
                for mytemp in set(library[mycia]['Ttemp']):
                    select = np.array(library[mycia]['Ttemp']) == mytemp
                    bini = []
                    matnu = np.array(library[mycia]['nutemp'])[select]
                    sigma2 = np.array(library[mycia]['Itemp'])[select]
                    for nubin, mydw in zip(nugrid, dwnu):
                        select = (matnu > (nubin-mydw/2.)) & (matnu <= nubin+mydw/2.)
                        bini.append(np.sum(sigma2[select]))
                        pass
                    bini = np.array(bini)/dwnu
                    library[mycia]['nu'].extend(list(nugrid))
                    library[mycia]['I'].extend(list(bini))
                    library[mycia]['T'].extend(list(np.ones(nugrid.size)*mytemp))
                    pass
                for iline in set(library[mycia]['nu']):
                    select = np.array(library[mycia]['nu']) == iline
                    y = np.array(library[mycia]['I'])[select]
                    x = np.array(library[mycia]['T'])[select]
                    sortme = np.argsort(x)
                    x = x[sortme]
                    y = y[sortme]
                    myspl = itp(x, y, bounds_error=False, fill_value=0)
                    library[mycia]['SPL'].append(myspl)
                    library[mycia]['SPLNU'].append(iline)
                    if verbose:
                        plt.plot(x, y, 'o')
                        xp = np.arange(101)/100.*(np.max(x) - np.min(x))+np.min(x)
                        plt.plot(xp, myspl(xp))
                        plt.show()
                        pass
                    pass
                pass
            if verbose:
                for temp in set(library[mycia]['T']):
                    select = np.array(library[mycia]['T']) == temp
                    plt.semilogy(1e4/(np.array(library[mycia]['nu'])[select]),
                                 np.array(library[mycia]['I'])[select])
                    pass
                plt.title(mycia)
                plt.xlabel('Wavelength $\\lambda$[$\\mu m$]')
                plt.ylabel('Line intensity $S(T)$ [$cm^{5}.molecule^{-2}$]')
                plt.show()
                pass
            pass
        for ks in knownspecies:
            log.warning('>-- %s', str(ks))
            library[ks] = {'MU':[], 'I':[], 'nu':[], 'S':[],
                           'g_air':[], 'g_self':[],
                           'Epp':[], 'eta':[], 'delta':[]}
            myfiles = [f for f in os.listdir(os.path.join(hitemp, ks))
                       if f.endswith('.par')]
            dwmin = abs(wgrid[1] - wgrid[0])/2.
            dwmax = abs(wgrid[-1] - wgrid[-2])/2.
            for fdata in myfiles:
                weqname = fdata.split('_')
                readit = True
                if len(weqname) > 2:
                    if float(weqname[1].split('-')[0]) != 0:
                        maxweq = 1e4/float(weqname[1].split('-')[0])
                    else: maxweq = 1e20
                    if maxweq < (np.min(wgrid)-dwmin): readit = False
                    minweq = 1e4/float(weqname[1].split('-')[1])
                    if minweq > (np.max(wgrid)+dwmax): readit = False
                    pass
                if readit:
                    with open(os.path.join(hitemp, ks, fdata), 'r') as fp:
                        data = fp.readlines()
                        fp.close()
                        # Rothman et Al. 2010
                        for line in data:
                            waveeq = (1e4)/float(line[3:3+12])
                            cotest = True
                            if ks == 'H2O': cotest = float(line[15:15+10]) > 1e-27
                            if ks == 'CO2': cotest = float(line[15:15+10]) > 1e-29
                            cmintest = waveeq < (np.max(wgrid)+dwmax)
                            cmaxtest = waveeq > (np.min(wgrid)-dwmin)
                            if cmintest and cmaxtest and cotest:
                                library[ks]['MU'].append(waveeq)
                                library[ks]['I'].append(int(line[2:2+1]))
                                library[ks]['nu'].append(float(line[3:3+12]))
                                library[ks]['S'].append(float(line[15:15+10]))
                                library[ks]['g_air'].append(float(line[35:35+5]))
                                library[ks]['g_self'].append(float(line[40:40+5]))
                                library[ks]['Epp'].append(float(line[45:45+10]))
                                library[ks]['eta'].append(float(line[55:55+4]))
                                library[ks]['delta'].append(float(line[59:59+8]))
                                pass
                            pass
                        pass
                    pass
                pass
            if verbose:
                for i in set(library[ks]['I']):
                    select = np.array(library[ks]['I']) == i
                    plt.semilogy(np.array(library[ks]['MU'])[select],
                                 np.array(library[ks]['S'])[select], '.')
                    pass
                plt.title(ks)
                plt.xlabel('Wavelength $\\lambda$[$\\mu m$]')
                plt.ylabel('Line intensity $S_{296K}$ [$cm.molecule^{-1}$]')
                plt.show()
                pass
            # BUILDS INTERPOLATORS SIMILAR TO EXOMOL DB DATA HANDLING
            mmr = 2.3  # Fortney 2015 for hot Jupiters
            solrad = 10.
            Hsmax = 15.
            nlevels = 100.
            pgrid = np.arange(np.log(solrad)-Hsmax, np.log(solrad)+Hsmax/nlevels,
                              Hsmax/(nlevels-1))
            pgrid = np.exp(pgrid)
            pressuregrid = pgrid[::-1]
            allxsections = []
            allwavenumbers = []
            alltemperatures = []
            for Tstep in np.arange(300, 2000, 100):
                log.warning('>---- %s K', str(Tstep))
                sigma, lsig = absorb(library[ks], qtgrid[ks], Tstep, pressuregrid, mmr,
                                     False, False, wgrid, debug=False)
                allxsections.append(sigma[0])
                allwavenumbers.append(lsig)
                alltemperatures.append(Tstep)
                pass
            library[ks]['nu'] = []
            library[ks]['I'] = []
            library[ks]['T'] = []
            library[ks]['SPL'] = []
            library[ks]['SPLNU'] = []
            for indextemp, mytemp in enumerate(alltemperatures):
                bini = []
                matnu = np.array(allwavenumbers)[indextemp]
                sigma2 = np.array(allxsections)[indextemp]
                for nubin, mydw in zip(nugrid, dwnu):
                    select = ((matnu > (nubin-mydw/2.)) & (matnu <= nubin+mydw/2.))
                    bini.append(np.sum(sigma2[select]))
                    pass
                bini = np.array(bini)/dwnu
                library[ks]['nu'].extend(list(nugrid))
                library[ks]['I'].extend(list(bini))
                library[ks]['T'].extend(list(np.ones(nugrid.size)*mytemp))
                pass
            for iline in set(library[ks]['nu']):
                select = np.array(library[ks]['nu']) == iline
                y = np.array(library[ks]['I'])[select]
                x = np.array(library[ks]['T'])[select]
                sortme = np.argsort(x)
                x = x[sortme]
                y = y[sortme]
                myspl = itp(x, y, bounds_error=False, fill_value=0)
                library[ks]['SPL'].append(myspl)
                library[ks]['SPLNU'].append(iline)
                pass
            if verbose:
                fts = 20
                plt.figure(figsize=(16,12))
                haha = [huhu for huhu in set(library[ks]['T'])]
                haha = np.sort(np.array(haha))
                haha = haha[::-1]
                for temp in haha:
                    select = np.array(library[ks]['T']) == temp
                    plt.semilogy(1e4/(np.array(library[ks]['nu'])[select]),
                                 np.array(library[ks]['I'])[select],
                                 label=str(int(temp))+'K')
                    pass
                plt.title(ks)
                plt.xlabel('Wavelength $\\lambda$[$\\mu m$]',
                           fontsize=fts+4)
                plt.ylabel('Cross Section [$cm^{2}.molecule^{-1}$]',
                           fontsize=fts+4)
                plt.tick_params(axis='both', labelsize=fts)
                plt.legend(bbox_to_anchor=(0.95, 0., 0.12, 1),
                           loc=5, ncol=1, mode='expand', numpoints=1,
                           borderaxespad=0., frameon=True)
                plt.show()
                pass
            pass
        out['data'][p]['XSECS'] = library
        out['data'][p]['QTGRID'] = qtgrid
        pass
    if out['data'].keys():
        cs = True
        out['STATUS'].append(True)
        pass
    return cs
# ------------------------ -------------------------------------------
# -- TOTAL PARTITION FUNCTION -- -------------------------------------
def gettpf(tips, knownspecies, verbose=False):
    '''
G. ROUDIER: Wrapper around HITRAN partition functions (Gamache et al. 2011)
    '''
    grid = {}
    tempgrid = list(np.arange(60., 3035., 25.))
    for ks in knownspecies:
        grid[ks] = {'T':tempgrid, 'Q':[], 'SPL':[]}
        with open(os.path.join(tips, ks)) as fp:
            data = fp.readlines()
            fp.close()
            for line in data:
                grid[ks]['Q'].append([float(num) for num in line.split(',')])
                pass
            pass
        for y in grid[ks]['Q']:
            myspl = itp(tempgrid, y)
            grid[ks]['SPL'].append(myspl)
            if verbose:
                plt.plot(tempgrid, myspl(tempgrid))
                plt.plot(tempgrid, y, '+')
                pass
            pass
        if verbose:
            plt.title(ks)
            plt.xlabel('Temperature T[K]')
            plt.ylabel('Total Internal Partition Function Q')
            plt.show()
        pass
    return grid
# ------------------------------ -------------------------------------
# -- ATMOS -- --------------------------------------------------------
def atmosversion():
    '''
    Alya Al-Kibbi:121
    Changes so that retrieval is done with CH4 being from HITEMP list instead of Exomol list
    R ESTRELA:131
    Merged Spectra Capability
    '''
    import dawgie
    return dawgie.VERSION(1,3,2)

def atmos(fin, xsl, spc, out, ext,
          hazedir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HAZE'),
          singlemod=None, mclen=int(1e4), sphshell=False, verbose=False):
    '''
G. ROUDIER: Cerberus retrievial
    '''
    am = False
    orbp = fin['priors'].copy()
    ssc = syscore.ssconstants(mks=True)
    crbhzlib = {'PROFILE':[]}
    hazelib(crbhzlib, hazedir=hazedir, verbose=verbose)
    # MODELS
    modfam = ['TEC', 'PHOTOCHEM']
    modparlbl = {'TEC':['XtoH', 'CtoO', 'NtoO'],
                 'PHOTOCHEM':['HCN', 'CH4', 'C2H2', 'CO2', 'H2CO']}
    if (singlemod is not None) and (singlemod in modfam):
        modfam = [modfam[modfam.index(singlemod)]]
        pass
    # PLANET LOOP
    for p in spc['data'].keys():
        out['data'][p] = {}
        out['data'][p]['MODELPARNAMES'] = modparlbl
        eqtemp = orbp['T*']*np.sqrt(orbp['R*']*ssc['Rsun/AU']/(2.*orbp[p]['sma']))
        tspc = np.array(spc['data'][p]['ES'])
        terr = np.array(spc['data'][p]['ESerr'])
        twav = np.array(spc['data'][p]['WB'])
        tspecerr = abs(tspc**2 - (tspc + terr)**2)
        tspectrum = tspc**2
        if 'STIS-WFC3' in ext:
            filters = np.array(spc['data'][p]['Fltrs'])
        cond_specG750 = filters == 'HST-STIS-CCD-G750L-STARE'
        # MASKING G750 WAV > 0.80
        twav_G750 = twav[cond_specG750]
        tspec_G750 = tspectrum[cond_specG750]
        tspecerr_G750 = tspecerr[cond_specG750]
        mask = (twav_G750 > 0.80) & (twav_G750 < 0.95)
        tspec_G750[mask] = np.nan
        tspecerr_G750[mask] = np.nan
        tspectrum[cond_specG750] = tspec_G750
        tspecerr[cond_specG750] = tspecerr_G750
        # CLEAN UP G750
#         cond_nan = np.isfinite(tspec_G750)
#         coefs_spec_G750 = poly.polyfit(twav_G750[cond_nan], tspec_G750[cond_nan], 1)
#         slp = twav_G750*coefs_spec_G750[1] + coefs_spec_G750[0]
#         mask = abs(slp - tspec_G750) >= 7 * np.nanmedian(tspecerr_G750)
#         tspec_G750[mask] = np.nan
#         tspectrum[cond_specG750] = tspec_G750
        Hs = spc['data'][p]['Hs']
        #  Clean up
        spechs = (np.sqrt(tspectrum) - np.sqrt(np.nanmedian(tspectrum)))/Hs
        cleanup2 = abs(spechs) > 3e0  # excluding everything above +-3 Hs
        tspectrum[cleanup2] = np.nan
        tspecerr[cleanup2] = np.nan
        twav[cleanup2] = np.nan
        cleanup = np.isfinite(tspectrum) & (tspecerr < 1e0)
        #
        cleanup = np.isfinite(tspectrum)
        solidr = orbp[p]['rp']*ssc['Rjup']  # MK

        for model in modfam:
            ctxtupdt(cleanup=cleanup, model=model, p=p, solidr=solidr, orbp=orbp,
                     tspectrum=tspectrum, xsl=xsl, spc=spc, modparlbl=modparlbl,
                     hzlib=crbhzlib)
            out['data'][p][model] = {}
            nodes = []
            with pm.Model():
                # CLOUD TOP PRESSURE
                ctp = pm.Uniform('CTP', -6., 1.)
                nodes.append(ctp)

                # HAZE SCAT. CROSS SECTION SCALE FACTOR
                hza = pm.Uniform('HScale', -6e0, 6e0)
                nodes.append(hza)

                # OFFSET BETWEEN STIS AND WFC3 filters
                if 'STIS-WFC3' in ext:
                    cond_off0 = filters == 'HST-STIS-CCD-G430L-STARE'
                    cond_off1 = filters == 'HST-STIS-CCD-G750L-STARE'
                    cond_off2 = filters == 'HST-WFC3-IR-G102-SCAN'
                    cond_off3 = filters == 'HST-WFC3-IR-G141-SCAN'
                    valid0 = True in cond_off0
                    valid1 = True in cond_off1
                    valid2 = True in cond_off2
                    valid3 = True in cond_off3
                    if 'STIS' in filters[0]:
                        if valid0:  # G430
                            if valid1 and valid2 and valid3:  # G430-G750-G102-G141
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off0]))
                                off1_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off1]))
                                off2_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off2]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                                off2 = pm.Uniform('OFF2', -off2_value, off2_value)
                                nodes.append(off2)
                            if valid1 and valid2 and not valid3:
                                off0_value = np.nanmedian(1e2*tspectrum[cond_off2])- np.nanmedian(1e2*tspectrum[cond_off0])
                                off1_value = np.nanmedian(1e2*tspectrum[cond_off2])- np.nanmedian(1e2*tspectrum[cond_off1])
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                            if valid1 and valid3 and not valid2:
                                off0_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off0])
                                off1_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off1])
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                            if valid2 and valid3 and not valid1:
                                off0_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off0])
                                off1_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off2])
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                            if valid3 and not valid1 and not valid2:
                                off0_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off0])
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                        if not valid0:
                            if valid1 and valid2 and valid3:
                                off0_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off1])
                                off1_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off2])
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                            if valid1 and valid3 and not valid2:
                                off0_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off1])
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                            if valid1 and valid2 and not valid3:
                                off0_value = np.nanmedian(1e2*tspectrum[cond_off2])- np.nanmedian(1e2*tspectrum[cond_off1])
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)

                    if 'WFC3' in filters[0]:
                        if valid2 and valid3:
                            off0_value = np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off2])
                            off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                            nodes.append(off0)
                # KILL HAZE POWER INDEX FOR SPHERICAL SHELL
                if sphshell:
                    hzloc = pm.Uniform('HLoc', -6.0, 1.0)
                    nodes.append(hzloc)

                    hzthick = pm.Uniform('HThick', 1.0, 20.0)
                    nodes.append(hzthick)

                    pass
                else:
                    hzi = pm.Uniform('HIndex', -4e0, 0e0)
                    nodes.append(hzi)
                    pass
                # BOOST TEMPERATURE PRIOR TO [75%, 150%] Teq
                tpr = pm.Uniform('T', 0.75e0*eqtemp, 1.5e0*eqtemp)
                nodes.append(tpr)

                # MODEL SPECIFIC ABSORBERS
                modelpar = pm.Uniform(model, lower=-6e0, upper=6e0,
                                      shape=len(modparlbl[model]))
                nodes.append(modelpar)

                # CERBERUS MCMC

                if sphshell:
                    if 'STIS-WFC3' in ext:
                        if 'STIS' in filters[0]:
                            if valid0:  # G430
                                if valid1 and valid2 and valid3:  # G430-G750-G102-G141
                                    _mcdata = pm.Normal('mcdata', mu=offcerberus(*nodes),
                                                        tau=1e0/tspecerr[cleanup]**2,
                                                        observed=tspectrum[cleanup])
                                if valid1 and valid2 and not valid3:
                                    _mcdata = pm.Normal('mcdata', mu=offcerberus1(*nodes),
                                                        tau=1e0/tspecerr[cleanup]**2,
                                                        observed=tspectrum[cleanup])
                                if valid1 and valid3 and not valid2:
                                    _mcdata = pm.Normal('mcdata', mu=offcerberus2(*nodes),
                                                        tau=1e0/tspecerr[cleanup]**2,
                                                        observed=tspectrum[cleanup])
                                if valid2 and valid3 and not valid1:
                                    _mcdata = pm.Normal('mcdata', mu=offcerberus3(*nodes),
                                                        tau=1e0/tspecerr[cleanup]**2,
                                                        observed=tspectrum[cleanup])
                                if valid3 and not valid1 and not valid2:
                                    _mcdata = pm.Normal('mcdata', mu=offcerberus4(*nodes),
                                                        tau=1e0/tspecerr[cleanup]**2,
                                                        observed=tspectrum[cleanup])
                            if not valid0:
                                if valid1 and valid2 and valid3:
                                    _mcdata = pm.Normal('mcdata', mu=offcerberus5(*nodes),
                                                        tau=1e0/tspecerr[cleanup]**2,
                                                        observed=tspectrum[cleanup])
                                if valid1 and valid3 and not valid2:
                                    _mcdata = pm.Normal('mcdata', mu=offcerberus6(*nodes),
                                                        tau=1e0/tspecerr[cleanup]**2,
                                                        observed=tspectrum[cleanup])
                                if valid1 and valid2 and not valid3:
                                    _mcdata = pm.Normal('mcdata', mu=offcerberus7(*nodes),
                                                        tau=1e0/tspecerr[cleanup]**2,
                                                        observed=tspectrum[cleanup])
                        if 'WFC3' in filters[0]:
                            if valid2 and valid3:
                                _mcdata = pm.Normal('mcdata', mu=offcerberus8(*nodes),
                                                    tau=1e0/tspecerr[cleanup]**2,
                                                    observed=tspectrum[cleanup])
                            if not valid2:
                                _mcdata = pm.Normal('mcdata', mu=spshfmcerberus(*nodes),
                                                    tau=1e0/(np.nanmedian(tspecerr[cleanup])**2),
                                                    observed=tspectrum[cleanup])
                            if not valid3:
                                _mcdata = pm.Normal('mcdata', mu=spshfmcerberus(*nodes),
                                                    tau=1e0/(np.nanmedian(tspecerr[cleanup])**2),
                                                    observed=tspectrum[cleanup])
                                pass
                            pass
                    if 'STIS-WFC3' not in ext:
                        _mcdata = pm.Normal('mcdata', mu=spshfmcerberus(*nodes),
                                            tau=1e0/(np.nanmedian(tspecerr[cleanup])**2),
                                            observed=tspectrum[cleanup])
                        pass
                    pass
                else:
                    _mcdata = pm.Normal('mcdata', mu=fmcerberus(*nodes),
                                        tau=1e0/(np.nanmedian(tspecerr[cleanup])**2),
                                        observed=tspectrum[cleanup])
                    pass
                log.warning('>-- MCMC nodes: %s', str([n.name for n in nodes]))
                trace = pm.sample(mclen, cores=4, tune=int(mclen/4),
                                  compute_convergence_checks=False, step=pm.Metropolis(),
                                  progressbar=verbose)
                mcpost = pm.summary(trace)
                pass
            mctrace = {}
            for key in mcpost['mean'].keys():
                tracekeys = key.split('[')
                if tracekeys.__len__() > 1:
                    indtrace = int(tracekeys[1].split(']')[0])
                    mctrace[key] = trace[tracekeys[0]][:, indtrace]
                    pass
                else: mctrace[key] = trace[tracekeys[0]]
                pass
            out['data'][p][model]['MCTRACE'] = mctrace
        out['data'][p]['WAVELENGTH'] = np.array(spc['data'][p]['WB'])
        out['data'][p]['SPECTRUM'] = np.array(spc['data'][p]['ES'])
        out['data'][p]['ERRORS'] = np.array(spc['data'][p]['ESerr'])
#         out['data'][p]['WAVELENGTH'] = twav
#         out['data'][p]['SPECTRUM'] = tspectrum
#         out['data'][p]['ERRORS'] = tspecerr
        out['data'][p]['VALID'] = cleanup
        am = True
        pass
    return am
# ----------- --------------------------------------------------------
# -- CERBERUS MODEL -- -----------------------------------------------
def crbmodel(mixratio, rayleigh, cloudtp, rp0, orbp, xsecs, qtgrid,
             temp, wgrid, lbroadening=False, lshifting=False,
             cialist=['H2-H', 'H2-H2', 'H2-He', 'He-H'].copy(),
             xmollist=['TIO', 'CH4', 'H2O', 'H2CO', 'HCN', 'CO', 'CO2', 'NH3'].copy(),
             nlevels=100, Hsmax=15., solrad=10.,
             hzlib=None, hzp=None, hzslope=-4., hztop=None, hzwscale=1e0,
             cheq=None, h2rs=True, logx=False, pnet='b', sphshell=False,
             verbose=False, debug=False):
    '''
G. ROUDIER: Cerberus forward model probing up to 'Hsmax' scale heights from solid radius solrad evenly log divided amongst nlevels steps
    '''
    ssc = syscore.ssconstants(mks=True)
    pgrid = np.arange(np.log(solrad)-Hsmax, np.log(solrad)+Hsmax/nlevels,
                      Hsmax/(nlevels-1))
    pgrid = np.exp(pgrid)
    dp = np.diff(pgrid[::-1])
    p = pgrid[::-1]
    z = [0e0]
    dz = []
    addz = []
    if cheq is not None:
        mixratio, fH2, fHe = crbce(p, temp,
                                   C2Or=cheq['CtoO'], X2Hr=cheq['XtoH'],
                                   N2Or=cheq['NtoO'])
        mmw, fH2, fHe = getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
        pass
    else: mmw, fH2, fHe = getmmw(mixratio)
    mmw = mmw*cst.m_p  # [kg]
    Hs = cst.Boltzmann*temp/(mmw*1e-2*(10.**orbp[pnet]['logg']))  # [m]
    for press, dpress in zip(p[:-1], dp):
        rdz = abs(Hs/2.*np.log(1. + dpress/press))
        if addz: dz.append(addz[-1]/2. + rdz)
        else: dz.append(2.*rdz)
        addz.append(2.*rdz)
        z.append(z[-1]+addz[-1])
        pass
    dz.append(addz[-1])
    rho = p*1e5/(cst.Boltzmann*temp)
    tau, wtau = gettau(xsecs, qtgrid, temp, mixratio, z, dz, rho, rp0, p, wgrid,
                       lbroadening, lshifting, cialist, fH2, fHe, xmollist, rayleigh,
                       hzlib, hzp, hzslope, hztop,
                       h2rs=h2rs, sphshell=sphshell, hzwscale=hzwscale,
                       verbose=verbose, debug=debug)
    # SEMI FINITE CLOUD ------------------------------------------------------------------
    reversep = np.array(p[::-1])
    selectcloud = p > 10.**cloudtp
    blocked = False
    if all(selectcloud):
        tau = tau*0
        blocked = True
        pass
    if not all(~selectcloud) and not blocked:
        cloudindex = np.max(np.arange(len(p))[selectcloud])+1
        for index in np.arange(wtau.size):
            myspl = itp(reversep, tau[:,index])
            tau[cloudindex,index] = myspl(10.**cloudtp)
            tau[:cloudindex,index] = 0.
            pass
        ctpdpress = 10.**cloudtp - p[cloudindex]
        ctpdz = abs(Hs/2.*np.log(1. + ctpdpress/p[cloudindex]))
        rp0 += (z[cloudindex]+ctpdz)
        pass
    atmdepth = 2e0*np.array(np.mat((rp0+np.array(z))*np.array(dz))*
                            np.mat(1. - np.exp(-tau))).flatten()
    model = (rp0**2 + atmdepth)/(orbp['R*']*ssc['Rsun'])**2
    noatm = rp0**2/(orbp['R*']*ssc['Rsun'])**2
    noatm = np.nanmin(model)
    rp0hs = np.sqrt(noatm*(orbp['R*']*ssc['Rsun'])**2)
    if verbose:
        fig, ax = plt.subplots(figsize=(10,6))
        axes = [ax, ax.twinx(), ax.twinx()]
        fig.subplots_adjust(left=0.125, right=0.775)
        axes[-1].spines['right'].set_position(('axes', 1.2))
        axes[-1].set_frame_on(True)
        axes[-1].patch.set_visible(False)
        axes[0].plot(wtau, 1e2*model)
        axes[0].plot(wtau, model*0+1e2*noatm, '--')
        axes[0].set_xlabel('Wavelength $\\lambda$[$\\mu m$]')
        axes[0].set_ylabel('Transit Depth [%]')
        axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
        yaxmin, yaxmax = axes[0].get_ylim()
        ax2min = (np.sqrt(1e-2*yaxmin)*orbp['R*']*ssc['Rsun'] - rp0hs)/Hs
        ax2max = (np.sqrt(1e-2*yaxmax)*orbp['R*']*ssc['Rsun'] - rp0hs)/Hs
        axes[-1].set_ylabel('Transit Depth Modulation [Hs]')
        axes[-1].set_ylim(ax2min, ax2max)
        axes[-1].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[1].set_ylabel('Transit Depth Modulation [ppm]')
        axes[1].set_ylim(1e6*(1e-2*yaxmin-noatm), 1e6*(1e-2*yaxmax-noatm))
        axes[1].get_yaxis().get_major_formatter().set_useOffset(False)
        if logx:
            plt.semilogx()
            plt.xlim([np.min(wtau), np.max(wtau)])
            pass
        plt.show()
        pass
    return model[::-1]
# -------------------- -----------------------------------------------
# -- MEAN MOLECULAR WEIGHT -- ----------------------------------------
def getmmw(mixratio, protosolar=True, fH2=None, fHe=None):
    '''
G. ROUDIER: Mean molecular weight estimate assuming proton mass dominated nucleous
    '''
    molsum = 0.
    mmw = 0.
    weights = {'H2':2., 'He':4., 'CH4':16., 'NH3':17., 'H2O':18.,
               'H2CO':30., 'TIO':64,
               'HCN':27., 'N2':28., 'C2H2':26., 'NO2':46., 'N2O':44.,
               'O3':48., 'HNO3':63., 'O2':32.,
               'CO':28., 'CO2':44., 'NO':30., 'OH':17.}
    for elem in mixratio:
        molsum = molsum + 10.**(mixratio[elem]-6.)
        mmw = mmw + 10.**(mixratio[elem]-6.)*weights[elem]
        pass
    mrH2He = 1. - molsum
    # Lodders 2010
    if protosolar: HEoH2 = 2.*2.343*1e9/(2.431*1e10)
    else: HEoH2 = fHe/fH2
    mrH2 = mrH2He/(1.+HEoH2)
    mrHe = HEoH2*mrH2
    mmw = mrH2*weights['H2'] + mrHe*weights['He'] + mmw
    return mmw, mrH2, mrHe
# --------------------------- ----------------------------------------
# -- TAU -- ----------------------------------------------------------
def gettau(xsecs, qtgrid, temp, mixratio,
           z, dz, rho, rp0, p, wgrid, lbroadening, lshifting,
           cialist, fH2, fHe, xmollist, rayleigh, hzlib, hzp, hzslope, hztop,
           h2rs=True, sphshell=False, isothermal=True, hzwscale=1e0,
           verbose=False, debug=False):
    '''
G. ROUDIER: Builds optical depth matrix
    '''
    # SPHERICAL SHELL --------------------------------------------------------------------
    if sphshell:
        # MATRICES INIT ------------------------------------------------------------------
        tau = np.zeros((len(z), wgrid.size))
        # DL ARRAY, Z VERSUS ZPRIME ------------------------------------------------------
        dlarray = []
        zprime = np.array(z)
        dzprime = np.array(dz)
        for iz, thisz in enumerate(z):
            dl = np.sqrt((rp0 + zprime + dzprime)**2 - (rp0 + thisz)**2)
            dl[:iz] = 0e0
            dl[iz:] = dl[iz:] - np.sqrt((rp0 + zprime[iz:])**2 - (rp0 + thisz)**2)
            dlarray.append(dl)
            pass
        dlarray = np.array(dlarray)
        # GAS ARRAY, ZPRIME VERSUS WAVELENGTH  -------------------------------------------
        for elem in mixratio:
            mmr = 10.**(mixratio[elem]-6.)
            # Fake use of xmollist due to changes in xslib v112
            # if elem not in xmollist:
            if not xmollist:
                # HITEMP/HITRAN ROTHMAN ET AL. 2010 --------------------------------------
                sigma, lsig = absorb(xsecs[elem], qtgrid[elem], temp, p, mmr,
                                     lbroadening, lshifting, wgrid, debug=False)
                sigma = np.array(sigma)  # cm^2/mol
                if True in sigma < 0: sigma[sigma < 0] = 0e0
                if True in ~np.isfinite(sigma): sigma[~np.isfinite(sigma)] = 0e0
                sigma = sigma*1e-4  # m^2/mol
                pass
            else:
                # EXOMOL HILL ET AL. 2013 ------------------------------------------------
                sigma, lsig = getxmolxs(temp, xsecs[elem])
                sigma = np.array(sigma)   # cm^2/mol
                if True in sigma < 0: sigma[sigma < 0] = 0e0
                if True in ~np.isfinite(sigma): sigma[~np.isfinite(sigma)] = 0e0
                sigma = np.array(sigma)*1e-4  # m^2/mol
                pass
            if isothermal: tau = tau + mmr*sigma*np.array([rho]).T
            pass
        # CIA ARRAY, ZPRIME VERSUS WAVELENGTH  -------------------------------------------
        for cia in cialist:
            if cia == 'H2-H2':
                f1 = fH2
                f2 = fH2
                pass
            if cia == 'H2-He':
                f1 = fH2
                f2 = fHe
                pass
            if cia == 'H2-H':
                f1 = fH2
                f2 = fH2*2.
                pass
            if cia == 'He-H':
                f1 = fHe
                f2 = fH2*2.
                pass
            # HITRAN RICHARD ET AL. 2012
            sigma, lsig = getciaxs(temp, xsecs[cia])  # cm^5/mol^2
            sigma = np.array(sigma)*1e-10  # m^5/mol^2
            if True in sigma < 0: sigma[sigma < 0] = 0e0
            if True in ~np.isfinite(sigma): sigma[~np.isfinite(sigma)] = 0e0
            tau = tau + f1*f2*sigma*np.array([rho**2]).T
            pass
        # RAYLEIGH ARRAY, ZPRIME VERSUS WAVELENGTH  --------------------------------------
        # NAUS & UBACHS 2000
        slambda0 = 750.*1e-3  # microns
        sray0 = 2.52*1e-28*1e-4  # m^2/mol
        sigma = sray0*(wgrid[::-1]/slambda0)**(-4)
        tau = tau + fH2*sigma*np.array([rho]).T
        # HAZE ARRAY, ZPRIME VERSUS WAVELENGTH  ------------------------------------------
        if hzlib is None:
            slambda0 = 750.*1e-3  # microns
            sray0 = 2.52*1e-28*1e-4  # m^2/mol
            sigma = sray0*(wgrid[::-1]/slambda0)**(hzslope)
            hazedensity = np.ones(len(z))
            tau = tau + (10.**rayleigh)*sigma*np.array([hazedensity]).T
            pass
        if hzlib is not None:
            # WEST ET AL. 2004
            sigma = 0.0083*(wgrid[::-1])**(hzslope)*(1e0 +
                                                     0.014*(wgrid[::-1])**(hzslope/2e0) +
                                                     0.00027*(wgrid[::-1])**(hzslope))
            if hzp in ['MAX', 'MEDIAN', 'AVERAGE']:
                frh = hzlib['PROFILE'][0][hzp][0]
                rh = frh(p)
                rh[rh < 0] = 0.
                refhzp = float(p[rh == np.max(rh)])
                if hztop is None: hzshift = 0e0
                else: hzshift = hztop - np.log10(refhzp)
                splp = np.log10(p[::-1])
                splrh = rh[::-1]
                thisfrh = itp(splp, splrh,
                              kind='linear', bounds_error=False, fill_value=0e0)
                hzwdist = hztop - np.log10(p)
                if hzwscale > 0:
                    preval = hztop - hzwdist/hzwscale - hzshift
                    rh = thisfrh(preval)
                    rh[rh < 0] = 0e0
                    pass
                else: rh = thisfrh(np.log10(p))*0
                if debug:
                    jptprofile = 'J'+hzp
                    jdata = np.array(hzlib['PROFILE'][0][jptprofile])
                    jpres = np.array(hzlib['PROFILE'][0]['PRESSURE'])
                    myfig = plt.figure(figsize=(12, 6))
                    plt.plot(1e6*jdata, jpres, color='blue',
                             label='Lavvas et al. 2017')
                    plt.axhline(refhzp, linestyle='--', color='blue')
                    plt.plot(1e6*rh, p, 'r', label='Parametrized density profile')
                    plt.plot(1e6*thisfrh(np.log10(p) - hzshift), p, 'g^')
                    if hztop is not None:
                        plt.axhline(10**hztop, linestyle='--', color='red')
                        pass
                    plt.semilogy()
                    plt.semilogx()
                    plt.gca().invert_yaxis()
                    plt.xlim([1e-4, np.max(1e6*rh)])
                    plt.tick_params(axis='both', labelsize=20)
                    plt.xlabel('Aerosol Density [$n.{cm}^{-3}$]', fontsize=24)
                    plt.ylabel('Pressure [bar]', fontsize=24)
                    plt.title('Aerosol density profile', fontsize=24)
                    plt.legend(loc='center left', frameon=False, fontsize=24,
                               bbox_to_anchor=(1, 1))
                    myfig.tight_layout()
                    plt.show()
                    pass
                pass
            else:
                rh = np.array([np.nanmean(hzlib['PROFILE'][0]['CONSTANT'])]*len(z))
                negrh = rh < 0e0
                if True in negrh: rh[negrh] = 0e0
                pass
            tau = tau + (10.**rayleigh)*sigma*np.array([rh]).T
            pass
        tau = 2e0*np.array(np.mat(dlarray)*np.mat(tau))
        pass
    else:
        # PLANE PARALLEL APPROXIMATION ---------------------------------------------------
        firstelem = True
        vectauelem = []
        for myz in list(z):
            tauelem = 0e0
            index = list(z).index(myz)
            for myzp, dzp, myrho in zip(z[index:], dz[index:], rho[index:]):
                dl = (np.sqrt((rp0+myzp+dzp)**2. - (rp0+myz)**2.) -
                      np.sqrt((rp0+myzp)**2. - (rp0+myz)**2.))
                tauelem += myrho*dl
                pass
            vectauelem.append(tauelem)
            pass
        vectauelem = np.array([vectauelem]).T
        for elem in mixratio:
            mmr = 10.**(mixratio[elem]-6.)
            if elem not in xmollist:
                # Rothman et .al 2010
                sigma, lsig = absorb(xsecs[elem], qtgrid[elem], temp,
                                     p, mmr, lbroadening, lshifting, wgrid,
                                     debug=False)  # cm^2/mol
                sigma = np.array(sigma)
                if np.sum(sigma < 0) > 0: sigma[sigma < 0] = 0
                if np.sum(~np.isfinite(sigma)) > 0: sigma[~np.isfinite(sigma)] = 0
                sigma = np.array(sigma)*1e-4  # m^2/mol
                if sigma.shape[0] < 2: sigma = sigma*np.array([np.ones(len(z))]).T
                pass
            else:
                # Hill et al. 2013
                sigma, lsig = getxmolxs(temp, xsecs[elem])  # cm^2/mol
                sigma = np.array(sigma)
                if np.sum(sigma < 0) > 0: sigma[sigma < 0] = 0
                if np.sum(~np.isfinite(sigma)) > 0: sigma[~np.isfinite(sigma)] = 0
                sigma = np.array(sigma)*1e-4  # m^2/mol
                sigma = sigma*np.array([np.ones(len(z))]).T
                pass
            # Tinetti et .al 2011
            tauz = vectauelem*sigma
            if firstelem:
                tau = np.array(tauz)*mmr
                firstelem = False
                pass
            else: tau = tau+np.array(tauz)*mmr
            pass
        veccia = []
        for myz in list(z):
            tauelem = 0.
            index = list(z).index(myz)
            for myzp, dzp, myrho in zip(z[index:], dz[index:], rho[index:]):
                dl = (np.sqrt((rp0+myzp+dzp)**2. - (rp0+myz)**2.) -
                      np.sqrt((rp0+myzp)**2. - (rp0+myz)**2.))
                tauelem += (myrho**2)*dl
                pass
            veccia.append(tauelem)
            pass
        veccia = np.array([veccia]).T
        for cia in cialist:
            tauz = []
            if cia == 'H2-H2':
                f1 = fH2
                f2 = fH2
                pass
            if cia == 'H2-He':
                f1 = fH2
                f2 = fHe
                pass
            if cia == 'H2-H':
                f1 = fH2
                f2 = fH2*2.
                pass
            if cia == 'He-H':
                f1 = fHe
                f2 = fH2*2.
                pass
            # Richard et al. 2012
            sigma, lsig = getciaxs(temp, xsecs[cia])  # cm^5/mol^2
            sigma = np.array(sigma)*1e-10  # m^5/mol^2
            if np.sum(sigma < 0) > 0: sigma[sigma < 0] = 0
            if np.sum(~np.isfinite(sigma)) > 0: sigma[~np.isfinite(sigma)] = 0
            tauz = veccia*sigma
            if firstelem:
                tau = np.array(tauz)*f1*f2
                firstelem = False
                pass
            else: tau = tau+np.array(tauz)*f1*f2
            pass
        if h2rs:
            # Naus & Ubachs 2000
            slambda0 = 750.*1e-3  # microns
            sray0 = 2.52*1e-28*1e-4  # m^2/mol
            sray = sray0*(wgrid[::-1]/slambda0)**(-4)
            tauz = vectauelem*sray
            if firstelem:
                tau = np.array(tauz)*fH2
                firstelem = False
                pass
            else: tau = tau+np.array(tauz)*fH2
            pass
        if hzlib is None:
            slambda0 = 750.*1e-3  # microns
            sray0 = 2.52*1e-28*1e-4  # m^2/mol
            sray = (10**rayleigh)*sray0*(wgrid[::-1]/slambda0)**(hzslope)
            tauz = vectauelem*sray
            if firstelem:
                tau = np.array(tauz)
                firstelem = False
                pass
            else: tau = tau+np.array(tauz)
            pass
        if hzlib is not None:
            # West et al. 2004
            tauaero = (10**rayleigh)*(0.0083*(wgrid[::-1])**(hzslope)*
                                      (1+0.014*(wgrid[::-1])**(hzslope/2e0)+
                                       0.00027*(wgrid[::-1])**(hzslope)))
            vectauhaze = []
            if hzp in ['MAX', 'MEDIAN', 'AVERAGE']:
                frh = hzlib['PROFILE'][0][hzp][0]
                rh = frh(p)
                rh[rh < 0] = 0e0
                refhzp = float(p[rh == np.max(rh)])
                if hztop is None: hzshift = 0e0
                else: hzshift = hztop - np.log10(refhzp)
                splp = np.log10(p[::-1]) + hzshift
                splrh = rh[::-1]
                thisfrh = itp(splp, splrh,
                              kind='linear', bounds_error=False, fill_value=0e0)
                rh = thisfrh(np.log10(p))
                rh[rh < 0] = 0e0
                if verbose:
                    jptprofile = 'J'+hzp
                    jdata = np.array(hzlib['PROFILE'][0][jptprofile])
                    jpres = np.array(hzlib['PROFILE'][0]['PRESSURE'])
                    plt.figure(figsize=(12, 6))
                    plt.plot(1e6*jdata, jpres, color='blue',
                             label='Zhang, West et al. 2014')
                    plt.axhline(refhzp, linestyle='--', color='blue')
                    plt.plot(1e6*rh, p, color='red', label='Parametrized density profile')
                    plt.plot(1e6*rh, p, 'r*')
                    if hztop is not None:
                        plt.axhline(10**hztop, linestyle='--', color='red')
                        pass
                    plt.semilogy()
                    plt.semilogx()
                    plt.gca().invert_yaxis()
                    plt.xlim([1e-10, 1e4])
                    plt.tick_params(axis='both', labelsize=20)
                    plt.xlabel('Aerosol Density [$n.{cm}^{-3}$]', fontsize=24)
                    plt.ylabel('Pressure [bar]', fontsize=24)
                    plt.title('Aerosol density profile', fontsize=24)
                    plt.legend(loc=3, frameon=False, fontsize=24)
                    plt.show()
                    pass
                pass
            else:
                rh = np.array(hzlib['PROFILE'][0]['CONSTANT'])
                negrh = rh < 0e0
                if True in negrh: rh[negrh] = 0e0
                pass
            for myz in list(z):
                tauelem = 0.
                index = list(z).index(myz)
                for myzp, dzp, rhohaze, in zip(z[index:], dz[index:], rh[index:]):
                    dl = (np.sqrt((rp0+myzp+dzp)**2. - (rp0+myz)**2.) -
                          np.sqrt((rp0+myzp)**2. - (rp0+myz)**2.))
                    tauelem += rhohaze*dl
                    pass
                vectauhaze.append(tauelem)
                pass
            tauz = (np.array([vectauhaze]).T*np.array([p]).T)*tauaero
            tau = tau + np.array(tauz)
            pass
        tau *= 2
        pass
    if debug:
        plt.figure(figsize=(12, 6))
        plt.imshow(np.log10(tau), aspect='auto', origin='lower',
                   extent=[max(wgrid), min(wgrid), np.log10(max(p)), np.log10(min(p))])
        plt.ylabel('log10(Pressure)', fontsize=24)
        plt.xlabel('Wavelength [$\\mu m$]', fontsize=24)
        plt.gca().invert_xaxis()
        plt.title('log10(Optical Depth)', fontsize=24)
        plt.tick_params(axis='both', labelsize=20)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
        plt.show()
        pass
    return tau, 1e4/lsig
# --------- ----------------------------------------------------------
# -- ATTENUATION COEFFICIENT -- --------------------------------------
def absorb(xsecs, qtgrid, T, p, mmr, lbroadening, lshifting, wgrid,
           iso=0, Tref=296., debug=False):
    '''
G. ROUDIER: HITRAN HITEMP database parser
    '''
    select = (np.array(xsecs['I']) == iso+1)
    S = np.array(xsecs['S'])[select]
    E = np.array(xsecs['Epp'])[select]
    gself = np.array(xsecs['g_self'])[select]
    nu = np.array(xsecs['nu'])[select]
    delta = np.array(xsecs['delta'])[select]
    eta = np.array(xsecs['eta'])[select]
    gair = np.array(xsecs['g_air'])[select]
    Qref = float(qtgrid['SPL'][iso](Tref))
    try: Q = float(qtgrid['SPL'][iso](T))
    except ValueError: Q = np.nan
    c2 = 1e2*cst.h*cst.c/cst.Boltzmann
    tips = (Qref*np.exp(-c2*E/T)*(1.-np.exp(-c2*nu/T)))/(Q*np.exp(-c2*E/Tref)*
                                                         (1.-np.exp(-c2*nu/Tref)))
    if np.all(~np.isfinite(tips)): tips = 0
    sigma = S*tips
    ps = mmr*p
    gamma = np.array(np.mat(p-ps).T*np.mat(gair*(Tref/T)**eta)+np.mat(ps).T*np.mat(gself))
    if lbroadening:
        if lshifting: matnu = np.array(np.mat(np.ones(p.size)).T*np.mat(nu) +
                                       np.mat(p).T*np.mat(delta))
        else: matnu = np.array(nu)*np.array([np.ones(len(p))]).T
        pass
    else: matnu = np.array(nu)
    absgrid = []
    nugrid = (1e4/wgrid)[::-1]
    dwnu = np.concatenate((np.array([np.diff(nugrid)[0]]), np.diff(nugrid)))
    if lbroadening:
        for mymatnu, mygamma in zip(matnu, gamma):
            binsigma = np.mat(sigma)*np.mat(intflor(nugrid, dwnu/2.,
                                                    np.array([mymatnu]).T,
                                                    np.array([mygamma]).T))
            binsigma = np.array(binsigma).flatten()
            absgrid.append(binsigma/dwnu)
            pass
        pass
    else:
        binsigma = []
        for nubin, dw in zip(nugrid, dwnu):
            select = (matnu > (nubin-dw/2.)) & (matnu <= nubin+dw/2.)
            binsigma.append(np.sum(sigma[select]))
            pass
        binsigma = np.array(binsigma)/dwnu
        absgrid.append(binsigma)
        pass
    if debug:
        plt.semilogy(1e4/matnu.T, sigma, '.')
        plt.semilogy(wgrid[::-1], binsigma, 'o')
        plt.xlabel('Wavelength $\\lambda$[$\\mu m$]')
        plt.ylabel('Absorption Coeff [$cm^{2}.molecule^{-1}$]')
        plt.show()
        pass
    return absgrid, nugrid
# ----------------------------- --------------------------------------
# -- PRESSURE BROADENING -- ------------------------------------------
def intflor(wave, dwave, nu, gamma):
    '''
G. ROUDIER: Pressure Broadening
    '''
    f = 1e0/np.pi*(np.arctan((wave+dwave - nu)/gamma) -
                   np.arctan((wave-dwave - nu)/gamma))
    return f
# ------------------------- ------------------------------------------
# -- CIA -- ----------------------------------------------------------
def getciaxs(temp, xsecs):
    '''
G. ROUDIER: Wrapper around CIA Cerberus library
    '''
    sigma = np.array([thisspl(temp) for thisspl in xsecs['SPL']])
    nu = np.array(xsecs['SPLNU'])
    select = np.argsort(nu)
    nu = nu[select]
    sigma = sigma[select]
    return sigma, nu
# --------- ----------------------------------------------------------
# -- EXOMOL -- -------------------------------------------------------
def getxmolxs(temp, xsecs):
    '''
G. ROUDIER: Wrapper around EXOMOL Cerberus library
    '''
    sigma = np.array([thisspl(temp) for thisspl in xsecs['SPL']])
    nu = np.array(xsecs['SPLNU'])
    select = np.argsort(nu)
    nu = nu[select]
    sigma = sigma[select]
    return sigma, nu
# ------------ -------------------------------------------------------
# -- CHEMICAL EQUILIBRIUM -- -----------------------------------------
def crbce(p, temp, C2Or=0., X2Hr=0., N2Or=0.):
    '''
G. ROUDIER: BURROWS AND SHARP 1998 + ANDERS & GREVESSE 1989
    '''
    solar = {
        'nH':9.10e-1, 'nHe':8.87e-2, 'nO':7.76e-4, 'nC':3.29e-4,
        'nNE':1.12e-4, 'nN':1.02e-4, 'nMg':3.49e-5, 'nSi':3.26e-5,
        'nFe':2.94e-5, 'nS':1.68e-5, 'nAr':3.29e-6, 'nAl':2.77e-6,
        'nCa':1.99e-6, 'nNa':1.87e-6, 'nNi':1.61e-6, 'nCr':4.40e-7,
        'nP':3.39e-7, 'nMn':3.11e-7, 'nCl':1.71e-7, 'nK':1.23e-7,
        'nTi':7.83e-8, 'nCo':7.34e-8, 'nF':2.75e-8, 'nV':9.56e-9,
        'nLi':1.86e-9, 'nRb':2.31e-10, 'nCs':1.21e-11
    }
    a1 = 1.106131e6
    b1 = -5.6895e4
    c1 = 62.565
    d1 = -5.81396e-4
    e1 = 2.346515e-8
    RcalpmolpK = 1.9872036  # cal/mol/K
    solCtO = solar['nC']/solar['nO']
    solNtO = solar['nN']/solar['nO']
    metal = solar.copy()
    metal['nH'] = 0.
    metal['nHe'] = 0.
    solvec = np.array([metal[temp] for temp in metal])
    if X2Hr >= np.log10(1./np.sum(solvec)):
        nH = 1e-16
        nH2 = 1e-16
        nHe = 1e-16
        X2Hr = np.log10(1./np.sum(solvec))  # 2.84 MAX
        pass
    else:
        if X2Hr < -10.:
            nH = 1./(1. + solar['nHe']/solar['nH'])
            nH2 = nH/2.
            nHe = nH*solar['nHe']/solar['nH']
            X2Hr = -10.
            pass
        else:
            nH = 1. - (10.**X2Hr)*np.sum(solvec)
            nH2 = nH/2.
            nHe = nH*solar['nHe']/solar['nH']
            pass
        pass
    if C2Or < -10.: C2Or = -10.
    if C2Or > 10.: C2Or = 10.
    if N2Or < -10.: N2Or = -10.
    if N2Or > 10.: N2Or = 10.
    pH2 = nH2*p
    K1 = np.exp((a1/temp + b1 + c1*temp + d1*temp**2 + e1*temp**3)/(RcalpmolpK*temp))
    AH2 = (pH2**2.)/(2.*K1)
    ACpAO = (10.**X2Hr)/nH*solar['nO']*(1. + (10.**C2Or)*solCtO)
    ACtAO = (10.**C2Or)*solCtO*(solar['nO']**2)*(((10.**X2Hr)/nH)**2)
    BCO = ACpAO + AH2 - np.sqrt((ACpAO+AH2)**2 - 4.*ACtAO)
    nCO = np.mean(BCO*pH2/p)
    if nCO <= 0: nCO = 1e-16
    nCH4 = np.mean((2.*(10.**X2Hr)/nH*solar['nC'] - BCO)*pH2/p)
    nH2O = np.mean((2.*(10.**X2Hr)/nH*solar['nO'] - BCO)*pH2/p)
    if nCH4 <= 0: nCH4 = 1e-16
    if nH2O <= 0: nH2O = 1e-16
    a2 = 8.16413e5
    b2 = -2.9109e4
    c2 = 58.5878
    d2 = -7.8284e-4
    e2 = 4.729048e-8
    K2 = np.exp((a2/temp + b2 + c2*temp + d2*temp**2 + e2*temp**3)/(RcalpmolpK*temp))
    AN = (10.**X2Hr)*(10.**N2Or)*solNtO*solar['nO']/nH  # solar['nN']/nH
    AH2 = (pH2**2.)/(8.*K2)
    BN2 = AN + AH2 - np.sqrt((AN + AH2)**2. - (AN)**2.)
    BNH3 = 2.*(AN - BN2)
    nN2 = np.mean(BN2*pH2/p)
    if nN2 <= 0: nN2 = 1e-16
    nNH3 = np.mean(BNH3*pH2/p)
    if nNH3 <= 0: nNH3 = 1e-16
    mixratio = {'H2O':np.log10(nH2O)+6., 'CH4':np.log10(nCH4)+6., 'NH3':np.log10(nNH3)+6.,
                'N2':np.log10(nN2)+6., 'CO':np.log10(nCO)+6.}
    return mixratio, nH2, nHe
# -------------------------- -----------------------------------------
# -- PYMC3 DETERMINISTIC FUNCTIONS -- --------------------------------
@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dvector],
           otypes=[tt.dvector])
def fmcerberus(*crbinputs):
    '''
G. ROUDIER: Wrapper around Cerberus forward model
    '''
    # ctp, hza, hzi, tpr, mdp = crbinputs
    ctp, hza, hzi, tpr, mdp = crbinputs
    fmc = np.zeros(ctxt.tspectrum.size)
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzslope=float(hzi), cheq=tceqdict, pnet=ctxt.p,
                       verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzslope=float(hzi), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def spshfmcerberus(*crbinputs):
    '''
G. ROUDIER: Wrapper around Cerberus forward model, spherical shell symmetry
    '''
    ctp, hza, hzloc, hzthick, tpr, mdp = crbinputs
    # ctp, hza, hzloc, hzthick, tpr, mdp = crbinputs
    fmc = np.zeros(ctxt.tspectrum.size)
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
# @tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar],
#             otypes=[tt.dvector])
def offcerberus(*crbinputs):
    '''
R.ESTRELA: ADD offsets between STIS filters and STIS and WFC3 filters
    '''
    ctp, hza, off0, off1, off2, hzloc, hzthick, tpr, mdp = crbinputs
#     off0, off1, off2 = crbinputs
#     ctp = -2.5744083
#     hza = -1.425234
#     hzloc = -0.406851
#     hzthick = 5.58950953
#     tpr = 1551.41137
#     mdp = [-1.24882918, -4.08582557, -2.4664526]
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    #  cond_wav = (wbb < 0.56) | (wbb > 1.02)
    fmc = np.zeros(ctxt.tspectrum.size)
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    cond_G430 = flt[ctxt.cleanup] == 'HST-STIS-CCD-G430L-STARE'
    cond_G141 = flt[ctxt.cleanup] == 'HST-WFC3-IR-G141-SCAN'
    tspectrum_clean = ctxt.tspectrum[ctxt.cleanup]
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup][cond_G141])
    fmc = fmc + np.nanmean(tspectrum_clean[cond_G141])
#     fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
#     fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    cond_G750 = flt[ctxt.cleanup] == 'HST-STIS-CCD-G750L-STARE'
    cond_G102 = flt[ctxt.cleanup] == 'HST-WFC3-IR-G102-SCAN'
    fmc[cond_G430] = fmc[cond_G430] - 1e-2*float(off0)
    fmc[cond_G750] = fmc[cond_G750] - 1e-2*float(off1)
    fmc[cond_G102] = fmc[cond_G102] - 1e-2*float(off2)
#     fmc[cond_G141] = fmc[cond_G141] + 1e-2*float(off2)
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def offcerberus1(*crbinputs):
    '''
R.ESTRELA: ADD offsets between STIS filters and STIS and WFC3 filters
    '''
    ctp, hza, off0, off1, hzloc, hzthick, tpr, mdp = crbinputs
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    fmc = np.zeros(ctxt.tspectrum.size)
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    cond_G430 = 'HST-STIS-CCD-G430L-STARE' in flt
    cond_G750 = 'HST-STIS-CCD-G750L-STARE' in flt
    fmc[cond_G430] = fmc[cond_G430] + 1e-2*float(off0)
    fmc[cond_G750] = fmc[cond_G750] + 1e-2*float(off1)
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def offcerberus2(*crbinputs):
    '''
R.ESTRELA: ADD offsets between STIS filters and STIS and WFC3 filters
    '''
    ctp, hza, off0, off1, hzloc, hzthick, tpr, mdp = crbinputs
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    fmc = np.zeros(ctxt.tspectrum.size)
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    cond_G430 = 'HST-STIS-CCD-G430-STARE' in flt
    cond_G750 = 'HST-STIS-CCD-G750-STARE' in flt
    fmc[cond_G430] = fmc[cond_G430] + 1e-2*float(off0)
    fmc[cond_G750] = fmc[cond_G750] + 1e-2*float(off1)
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def offcerberus3(*crbinputs):
    '''
R.ESTRELA: ADD offsets between STIS filters and STIS and WFC3 filters
    '''
    ctp, hza, off0, off1, hzloc, hzthick, tpr, mdp = crbinputs
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    fmc = np.zeros(ctxt.tspectrum.size)
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    cond_G430 = 'HST-STIS-CCD-G430-STARE' in flt
    cond_G102 = 'HST-WFC3-IR-G102-SCAN' in flt
    fmc[cond_G430] = fmc[cond_G430] + 1e-2*float(off0)
    fmc[cond_G102] = fmc[cond_G102] + 1e-2*float(off1)
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def offcerberus4(*crbinputs):
    '''
R.ESTRELA: ADD offsets between STIS filters and STIS and WFC3 filters
    '''
    ctp, hza, off0, hzloc, hzthick, tpr, mdp = crbinputs
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    fmc = np.zeros(ctxt.tspectrum.size)
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    cond_G430 = 'HST-STIS-CCD-G430-STARE' in flt
    fmc[cond_G430] = fmc[cond_G430] + 1e-2*float(off0)
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def offcerberus5(*crbinputs):
    '''
R.ESTRELA: ADD offsets between STIS filters and STIS and WFC3 filters
    '''
    ctp, hza, off0, off1, hzloc, hzthick, tpr, mdp = crbinputs
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    fmc = np.zeros(ctxt.tspectrum.size)
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    cond_G102 = 'HST-WFC3-IR-G102-SCAN' in flt
    cond_G750 = 'HST-STIS-CCD-G750-STARE' in flt
    fmc[cond_G750] = fmc[cond_G750] + 1e-2*float(off0)
    fmc[cond_G102] = fmc[cond_G102] + 1e-2*float(off1)
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def offcerberus6(*crbinputs):
    '''
R.ESTRELA: ADD offsets between STIS filters and STIS and WFC3 filters
    '''
    ctp, hza, off0, hzloc, hzthick, tpr, mdp = crbinputs
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    fmc = np.zeros(ctxt.tspectrum.size)
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    cond_G750 = 'HST-STIS-CCD-G750-STARE' in flt
    fmc[cond_G750] = fmc[cond_G750] + 1e-2*float(off0)
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def offcerberus7(*crbinputs):
    '''
R.ESTRELA: ADD offsets between STIS filters and WFC3 filters
    '''
    ctp, hza, off0, hzloc, hzthick, tpr, mdp = crbinputs
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    fmc = np.zeros(ctxt.tspectrum.size)
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    cond_G750 = 'HST-STIS-CCD-G750-STARE' in flt
    fmc[cond_G750] = fmc[cond_G750] + 1e-2*float(off0)
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def offcerberus8(*crbinputs):
    '''
R.ESTRELA: ADD offsets between WFC3 filters
    '''
    ctp, hza, off0, hzloc, hzthick, tpr, mdp = crbinputs
    wbb = np.array(ctxt.spc['data'][ctxt.p]['WB'])
    fmc = np.zeros(ctxt.tspectrum.size)
    flt = np.array(ctxt.spc['data'][ctxt.p]['Fltrs'])
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])
        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), wbb,
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    else:
        mixratio = {}
        for index, key in enumerate(ctxt.modparlbl[ctxt.model]):
            mixratio[key] = float(mdp[index])
            pass
        fmc = crbmodel(mixratio, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       sphshell=True, verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    cond_G102 = 'HST-WFC3-IR-G102-SCAN' in flt
    fmc[cond_G102] = fmc[cond_G102] + 1e-2*float(off0)
    return fmc

# ----------------------------------- --------------------------------
# -- HAZE DENSITY PROFILE LIBRARY -- ---------------------------------
def hazelib(sv,
            hazedir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HAZE'),
            datafile='Jup-ISS-aerosol.dat', verbose=False,
            fromjupiter=True, narrow=True):
    vdensity = {'PRESSURE':[], 'CONSTANT':[], 'JMAX':[], 'MAX':[],
                'JMEDIAN':[], 'MEDIAN':[], 'JAVERAGE':[], 'AVERAGE':[]}
    with open(os.path.join(hazedir, datafile), 'r') as fp: data = fp.readlines()
    # LATITUDE GRID
    latitude = data[0]
    latitude = np.array(latitude.split(' '))
    latitude = latitude[latitude != '']
    latitude = latitude.astype(float)  # [DEGREES]
    # DENSITY MATRIX
    pressure = []  # [mbar]
    density = []
    for line in data[1:]:
        line = np.array(line.split(' '))
        line = line[line != '']
        line = line.astype(float)
        pressure.append(line[0])
        density.append(line[1:])
        pass
    # JUPITER PROFILES
    if fromjupiter:
        pressure = 1e-3*np.array(pressure)  # [bar]
        density = 1e-6*np.array(density)  # [n/m^3]
        jmax = np.nanmax(density, 1)
        jmed = np.median(density, 1)
        jav = np.mean(density, 1)
        isobar = pressure*0. + np.mean(jav)
        sortme = np.argsort(pressure)
        jmaxspl = itp(pressure[sortme], jmax[sortme],
                      kind='linear', bounds_error=False, fill_value=0e0)
        jmedspl = itp(pressure[sortme], jmed[sortme],
                      kind='linear', bounds_error=False, fill_value=0e0)
        javspl = itp(pressure[sortme], jav[sortme],
                     kind='linear', bounds_error=False, fill_value=0e0)
        pass
    else:
        if narrow:
            density = [0.322749941, 0.618855379, 0.659653289, 0.660104488, 0.687461411,
                       0.701591071, 0.823367371, 0.998717644, 1.254642603, 1.483661838,
                       1.726335787, 1.928591783, 2.157824745, 2.387176443, 2.58959867,
                       2.778603657, 2.981049632, 3.237164569, 3.573854191, 3.9373308,
                       4.273759202, 4.448824507, 4.341700309, 4.019449062, 3.683756827,
                       3.227214438, 2.891522204, 2.461790549, 1.978247447, 1.481168369,
                       1.010947518, 0.500379957, 0.030087865, -0.252101639]
            pressure = [1.874253147, 1.632296367, 1.349380195, 1.080826407, 0.797986227,
                        0.388012349, -0.09324151, -0.461724056, -0.788259321,
                        -1.100508193, -1.540042745, -1.922811684, -2.362270245,
                        -2.872400855, -3.354110663, -3.849878889, -4.345723106,
                        -4.78533365, -5.182996913, -5.524274519, -5.766459273,
                        -5.9653289, -6.205005937, -6.40106388, -6.597045832,
                        -6.863015911, -7.058997863, -7.282716694, -7.47786274,
                        -7.616395156, -7.740945144, -7.851132748, -7.933279506,
                        -7.974086915]
            pass
        else:
            density = [-1.222929936, -0.76433121, -0.229299363, 0.025477707, 0.25477707,
                       0.789808917, 1.503184713, 2.089171975, 2.369426752, 2.522292994,
                       2.675159236, 2.904458599, 3.133757962, 3.337579618, 3.541401274,
                       3.694267516, 3.796178344, 3.821656051, 3.898089172, 3.923566879,
                       4.0,
                       4.025477707, 4.050955414, 4.127388535, 4.152866242, 4.178343949,
                       4.127388535, 3.974522293, 3.796178344, 3.541401274, 3.261146497,
                       2.904458599, 2.624203822, 2.267515924, 1.961783439, 1.630573248,
                       1.299363057, 0.968152866, 0.636942675, 0.280254777, -0.076433121,
                       -0.433121019, -0.76433121, -1.070063694, -1.299363057]
            pressure = [2.812911584, 2.809025154, 2.63499946, 2.327755587, 2.156320846,
                        2.083990068, 1.976249595, 1.835690381, 1.528230595, 1.086257152,
                        0.678182014, 0.337255749, 0.030227788, -0.310482565, -0.718989528,
                        -1.059268056, -1.534707978, -1.941703552, -2.518622477,
                        -2.85782144, -3.33304545, -3.807837634, -4.214833207,
                        -4.690057217, -5.097052791, -5.60574328, -6.012091115,
                        -6.4175753, -6.788945266, -7.057972579, -7.292885674,
                        -7.493252726, -7.626470906, -7.725143042, -7.790348699,
                        -7.855338443, -7.954226492, -8.019216237, -8.118104286,
                        -8.182878117, -8.281550254, -8.38022239, -8.411313829,
                        -8.476519486, -8.508474576]
            pass
        density = 1e-6*(10**(np.array(density)))
        density = np.array([density, density])
        density = density.T
        jmax = np.nanmax(density, 1)
        jmed = np.median(density, 1)
        jav = np.mean(density, 1)
        pressure = 10**(np.array(pressure))
        isobar = pressure*0. + np.mean(jav)
        sortme = np.argsort(pressure)
        jmaxspl = itp(pressure[sortme], jmax[sortme],
                      kind='linear', bounds_error=False, fill_value=0e0)
        jmedspl = itp(pressure[sortme], jmed[sortme],
                      kind='linear', bounds_error=False, fill_value=0e0)
        javspl = itp(pressure[sortme], jav[sortme],
                     kind='linear', bounds_error=False, fill_value=0e0)
        pass
    if verbose:
        myfig = plt.figure(figsize=(12, 6))
        for vmr in density.T:
            plt.plot(1e6*vmr, pressure, 'k+')
            plt.plot(1e6*vmr, pressure)
            pass
        plt.xlabel('Aerosol Density [$n.{cm}^{-3}$]', fontsize=24)
        plt.ylabel('Pressure [bar]', fontsize=24)
        plt.semilogy()
        plt.semilogx()
        plt.gca().invert_yaxis()
        plt.tick_params(axis='both', labelsize=20)
        plt.xlim([1e-10, 1e4])
        plt.title('Latitudinal Variations', fontsize=24)
        myfig.tight_layout()

        myfig = plt.figure(figsize=(12, 6))
        plt.plot(1e6*jmax, pressure, label='Max', color='blue')
        plt.plot(1e6*jmed, pressure, label='Median', color='red')
        plt.plot(1e6*jav, pressure, label='Average', color='green')
        plt.plot(1e6*isobar, pressure, label='Constant', color='black')
        plt.legend(loc=3, frameon=False, fontsize=24)
        plt.plot(1e6*jmaxspl(pressure), pressure, 'bo')
        plt.plot(1e6*jmedspl(pressure), pressure, 'ro')
        plt.plot(1e6*javspl(pressure), pressure, 'go')
        plt.xlabel('Aerosol Density [$n.{cm}^{-3}$]', fontsize=24)
        plt.ylabel('Pressure [bar]', fontsize=24)
        plt.semilogy()
        plt.semilogx()
        plt.gca().invert_yaxis()
        plt.xlim([1e-4, 1e5])
        plt.tick_params(axis='both', labelsize=20)
        plt.title('Profile Library', fontsize=24)
        myfig.tight_layout()

        plt.show()
        pass
    vdensity['PRESSURE'] = list(pressure)
    vdensity['CONSTANT'] = list(isobar)
    vdensity['JMAX'] = list(jmax)
    vdensity['MAX'].append(jmaxspl)
    vdensity['JMEDIAN'] = list(jmed)
    vdensity['MEDIAN'].append(jmedspl)
    vdensity['JAVERAGE'] = list(jav)
    vdensity['AVERAGE'].append(javspl)
    sv['PROFILE'].append(vdensity)
    return
# ---------------------------------- ---------------------------------
