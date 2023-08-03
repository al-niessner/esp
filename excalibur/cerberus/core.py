'''cerberus core ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import excalibur
# pylint: disable=import-self
import excalibur.cerberus.core
import excalibur.system.core as syscore
from excalibur.target.targetlists import get_target_lists
# from excalibur.cerberus.core import savesv
from excalibur.util.cerberus import crbce,getmmw
from excalibur.cerberus.plotting import rebinData, plot_bestfit, \
    plot_corner, plot_vsPrior, plot_walkerEvolution, \
    plot_fitsVStruths, plot_massVSmetals

import logging
log = logging.getLogger(__name__)
pymc3log = logging.getLogger('pymc3')
pymc3log.setLevel(logging.ERROR)

import os
import pymc3 as pm
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from collections import defaultdict

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
    return dawgie.VERSION(1,1,3)

def myxsecs(spc, out,
            hitemp=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HITEMP'),
            tips=os.path.join(excalibur.context['data_dir'], 'CERBERUS/TIPS'),
            ciadir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HITRAN/CIA'),
            exomoldir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/EXOMOL'),
            knownspecies=['NO','OH', 'C2H2', 'N2', 'N2O', 'O3', 'O2'].copy(),
            cialist=['H2-H', 'H2-H2', 'H2-He', 'He-H'].copy(),
            xmspecies=['TIO', 'H2O', 'H2CO', 'HCN', 'CO', 'CO2', 'NH3', 'CH4'].copy(),
            verbose=False):
    '''
G. ROUDIER: Builds Cerberus cross section library
    '''
    cs = False
    for p in spc['data'].keys():
        out['data'][p] = {}
        wgrid = np.array(spc['data'][p]['WB'])
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
                with open(os.path.join(thisxmdir, mf), 'r', encoding="utf-8") as fp:
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
                haha = list(set(library[myexomol]['T']))
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
            with open(myfile, 'r', encoding="utf-8") as fp:
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
                    with open(os.path.join(hitemp, ks, fdata), 'r',
                              encoding="utf-8") as fp:
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
                haha = list(set(library[ks]['T']))
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
        with open(os.path.join(tips, ks), 'r', encoding="utf-8") as fp:
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
    return dawgie.VERSION(1,3,2)

def atmos(fin, xsl, spc, out, ext,
          hazedir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HAZE'),
          singlemod=None, mclen=int(1e4), sphshell=False, verbose=False):
    '''
G. ROUDIER: Cerberus retrievial
    '''
    noClouds = True

    am = False
    orbp = fin['priors'].copy()
    ssc = syscore.ssconstants(mks=True)
    crbhzlib = {'PROFILE':[]}
    hazelib(crbhzlib, hazedir=hazedir, verbose=False)
    # MODELS
    modfam = ['TEC', 'PHOTOCHEM']
    if ext=='Ariel-sim': modfam = ['TEC']
    modparlbl = {'TEC':['XtoH', 'CtoO', 'NtoO'],
                 'PHOTOCHEM':['HCN', 'CH4', 'C2H2', 'CO2', 'H2CO']}
    # if ext=='Ariel-sim': modparlbl = {'TEC':['XtoH', 'CtoO']}
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
            pass
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
            out['data'][p][model]['prior_ranges'] = {}
            # keep track of the bounds put on each parameter
            # this will be helpful for later plotting and analysis
            prior_ranges = {}
            nodes = []
            with pm.Model():
                if not noClouds:
                    # CLOUD TOP PRESSURE
                    prior_ranges['CTP'] = (-6,1)
                    ctp = pm.Uniform('CTP', -6., 1.)
                    nodes.append(ctp)

                    # HAZE SCAT. CROSS SECTION SCALE FACTOR
                    prior_ranges['Hscale'] = (-6,6)
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
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off2])- np.nanmedian(1e2*tspectrum[cond_off0]))
                                off1_value = abs(np.nanmedian(1e2*tspectrum[cond_off2])- np.nanmedian(1e2*tspectrum[cond_off1]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                            if valid1 and valid3 and not valid2:
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off0]))
                                off1_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off1]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                            if valid2 and valid3 and not valid1:
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off0]))
                                off1_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off2]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                            if valid3 and not valid1 and not valid2:
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off0]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                        if not valid0:
                            if valid1 and valid2 and valid3:
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off1]))
                                off1_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off2]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                                off1 = pm.Uniform('OFF1', -off1_value, off1_value)
                                nodes.append(off1)
                            if valid1 and valid3 and not valid2:
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off1]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                            if valid1 and valid2 and not valid3:
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off2])- np.nanmedian(1e2*tspectrum[cond_off1]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)
                            if valid1 and valid2 and not valid3:
                                off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off2])- np.nanmedian(1e2*tspectrum[cond_off1]))
                                off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                                nodes.append(off0)

                    if 'WFC3' in filters[0]:
                        if valid2 and valid3:
                            off0_value = abs(np.nanmedian(1e2*tspectrum[cond_off3])- np.nanmedian(1e2*tspectrum[cond_off2]))
                            off0 = pm.Uniform('OFF0', -off0_value, off0_value)
                            nodes.append(off0)
                # KILL HAZE POWER INDEX FOR SPHERICAL SHELL
                if not noClouds:
                    if sphshell:
                        prior_ranges['HLoc'] = (-6,1)
                        hzloc = pm.Uniform('HLoc', -6.0, 1.0)
                        nodes.append(hzloc)

                        prior_ranges['HThick'] = (1,20)
                        hzthick = pm.Uniform('HThick', 1.0, 20.0)
                        nodes.append(hzthick)
                        pass
                    else:
                        prior_ranges['HIndex'] = (-4,0)
                        hzi = pm.Uniform('HIndex', -4e0, 0e0)
                        nodes.append(hzi)
                        pass
                # BOOST TEMPERATURE PRIOR TO [75%, 150%] Teq
                prior_ranges['T'] = (0.75e0*eqtemp, 1.5e0*eqtemp)
                tpr = pm.Uniform('T', 0.75e0*eqtemp, 1.5e0*eqtemp)
                nodes.append(tpr)

                # MODEL SPECIFIC ABSORBERS
                for param in modparlbl[model]:
                    if param=='XtoH':
                        prior_ranges['[X/H]'] = (-6,6)
                    elif param=='CtoO':
                        prior_ranges['[C/O]'] = (-6,6)
                    elif param=='NtoO':
                        prior_ranges['[N/O]'] = (-6,6)
                    else:
                        prior_ranges[param] = (-6,6)
                modelpar = pm.Uniform(model, lower=-6e0, upper=6e0,
                                      shape=len(modparlbl[model]))
                nodes.append(modelpar)

                # CERBERUS MCMC
                if noClouds:
                    print('RUNNING MCMC - NO CLOUDS!')
                    _mcdata = pm.Normal('mcdata', mu=clearfmcerberus(*nodes),
                                        tau=1e0/(np.nanmedian(tspecerr[cleanup])**2),
                                        observed=tspectrum[cleanup])
                    pass

                elif sphshell:
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
                        print('STANDARD MCMC (WITH CLOUDS)')
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
                # GMR: Should be able to find it... Joker
                # pylint: disable=no-member
                mcpost = pm.stats.summary(trace)
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
        out['data'][p][model]['prior_ranges'] = prior_ranges
        out['data'][p]['WAVELENGTH'] = np.array(spc['data'][p]['WB'])
        out['data'][p]['SPECTRUM'] = np.array(spc['data'][p]['ES'])
        out['data'][p]['ERRORS'] = np.array(spc['data'][p]['ESerr'])
        if ext=='Ariel-sim':
            print('key check',spc['data'][p].keys())
            # wavelength should be identical to just above, but just in case load it here too
            if 'true_spectrum' in spc['data'][p].keys():

                out['data'][p]['TRUTH_SPECTRUM'] = np.array(spc['data'][p]['true_spectrum']['fluxDepth'])
                out['data'][p]['TRUTH_WAVELENGTH'] = np.array(spc['data'][p]['true_spectrum']['wavelength_um'])
                out['data'][p]['TRUTH_MODELPARAMS'] = spc['data'][p]['model_params']
        out['data'][p]['VALID'] = cleanup
        #  big problem!!! this line was missing!  added 7/9/23
        out['STATUS'].append(True)
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
    G. ROUDIER: Cerberus forward model probing up to 'Hsmax' scale heights from solid
    radius solrad evenly log divided amongst nlevels steps
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
        mixratio['CO2'] = mixratio['NH3']
        mmw, fH2, fHe = getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
        pass
    else: mmw, fH2, fHe = getmmw(mixratio)
    mmw = mmw*cst.m_p  # [kg]
    Hs = cst.Boltzmann*temp/(mmw*1e-2*(10.**float(orbp[pnet]['logg'])))  # [m]
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
            # THIS HAS TO BE FIXED
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
# -------------------------- -----------------------------------------
# -- PYMC3 DETERMINISTIC FUNCTIONS -- --------------------------------
@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dvector],
           otypes=[tt.dvector])
def fmcerberus(*crbinputs):
    '''
    G. ROUDIER: Wrapper around Cerberus forward model
    '''
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

@tco.as_op(itypes=[tt.dscalar, tt.dvector],
           otypes=[tt.dvector])
def clearfmcerberus(*crbinputs):
    '''
    Wrapper around Cerberus forward model - NO CLOUDS!
    '''
    # print(' running a NO-CLOUDS forward model')
    ctp = 3.    # cloud deck is very deep - 1000 bars
    hza = -10.  # small number means essentially no haze
    hzloc = 0.
    hzthick = 0.

    tpr, mdp = crbinputs
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

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dscalar, tt.dscalar, tt.dscalar, tt.dvector],
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
    return fmc

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dscalar, tt.dscalar, tt.dvector],
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

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dscalar, tt.dscalar, tt.dvector],
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

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dscalar, tt.dscalar, tt.dvector],
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

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dscalar, tt.dscalar, tt.dvector],
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

@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dscalar, tt.dscalar, tt.dvector],
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
    '''Haze density profiles'''
    vdensity = {'PRESSURE':[], 'CONSTANT':[], 'JMAX':[], 'MAX':[],
                'JMEDIAN':[], 'MEDIAN':[], 'JAVERAGE':[], 'AVERAGE':[]}
    with open(os.path.join(hazedir, datafile), 'r', encoding="utf-8") as fp:
        data = fp.readlines()
        pass
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
# -- ROUDIER ET AL. 2021 RELEASE -- ----------------------------------
def rlsversion():
    '''
    GMR:110 Initial release to IPAC
    GMR:111 Removed empty keys
    '''
    return dawgie.VERSION(1,1,1)

def release(trgt, fin, out, verbose=False):
    '''
    GMR: Format Cerberus SV products to be released to IPAC
    trgt [INPUT]: target name
    fin [INPUT]: system.finalize.parameters
    out [INPUT/OUTPUT]
    ext [INPUT]: 'HST-WFC3-IR-G141-SCAN'
    verbose [OPTIONAL]: verbosity
    '''
    rlsed = False
    plist = fin['priors']['planets']
    thispath = os.path.join(excalibur.context['data_dir'], 'CERBERUS')
    for p in plist:
        intxtf = os.path.join(thispath, 'P.CERBERUS.atmos', trgt+p+'.txt')
        incorrpng = os.path.join(thispath, 'P.CERBERUS.atmos', trgt+p+'_atmos_corr.png')
        intxtpng = os.path.join(thispath, 'P.CERBERUS.atmos', trgt+p+'_atmos.png')
        out['data'][p] = {}
        try:
            atm = np.loadtxt(intxtf)
            out['data'][p]['atmos'] = atm
            out['STATUS'].append(True)
            pass
        except FileNotFoundError: pass
        try:
            corrplot = img.imread(incorrpng)
            out['data'][p]['corrplot'] = corrplot
            out['STATUS'].append(True)
            pass
        except FileNotFoundError: pass
        try:
            modelplot = img.imread(intxtpng)
            out['data'][p]['modelplot'] = modelplot
            out['STATUS'].append(True)
            pass
        except FileNotFoundError: pass
        if not out['data'][p]:
            if verbose: log.warning('--< No data found for %s', p)
            out['data'].pop(p)
            pass
        pass
    rlsed = out['STATUS'][-1]
    if verbose: log.warning('--< %s', out['STATUS'])
    return rlsed
# --------------------------------- ----------------------------------
# -- RESULTS ----------------------------------------------------------
def resultsversion():
    '''
    V1.0.0: plot the best-fit spectrum on top of the data
    '''
    return dawgie.VERSION(1,0,0)
# ------------------------------ -------------------------------------
def results(trgt, filt, fin, xsl, atm, out, verbose=False):
    '''
    Plot out the results from atmos()
    trgt [INPUT]: target name
    filt [INPUT]: filter
    fin [INPUT]: system.finalize.parameters
    xls [INPUT]: cerberus.xslib.data
    atm [INPUT]: cerberus.atmos.data
    out [INPUT/OUTPUT]
    verbose [OPTIONAL]: verbosity
    '''
    ssc = syscore.ssconstants(mks=True)

    if verbose: print('cerberus/results for target:',trgt)

    completed_at_least_one_planet = False

    for p in fin['priors']['planets']:
        # print('post-analysis for planet:',p)

        # TEC params - X/H, C/O, N/O
        # disEq params - HCN, CH4, C2H2, CO2, H2CO

        # limit results to just the TEC model
        if 'TEC' not in atm[p]['MODELPARNAMES'].keys():
            log.warning('>-- %s', 'TROUBLE: theres no TEC fit!?')
            return False

        alltraces = []
        allkeys = []
        for key in atm[p]['TEC']['MCTRACE']:
            # print('fillin tru the keys again?',key)

            alltraces.append(atm[p]['TEC']['MCTRACE'][key])

            if key=='TEC[0]': allkeys.append('[X/H]')
            elif key=='TEC[1]': allkeys.append('[C/O]')
            elif key=='TEC[2]': allkeys.append('[N/O]')
            else: allkeys.append(key)
        # print('allkeys',allkeys)

        # make note of the bounds placed on each parameter
        if 'prior_ranges' in atm[p]['TEC'].keys():
            prior_ranges = atm[p]['TEC']['prior_ranges']
        else:
            prior_ranges = {}

        if 'CTP' in allkeys:
            noClouds = False
        else:
            noClouds = True

        # ndim = len(alltraces)
        # nsamples = len(alltraces[0])
        # print('ndim,nsamples',ndim,nsamples)

        # save the relevant info
        transitdata = {}
        transitdata['wavelength'] = atm[p]['WAVELENGTH']
        transitdata['depth'] = atm[p]['SPECTRUM']**2
        transitdata['error'] = 2 * atm[p]['SPECTRUM'] * atm[p]['ERRORS']

        truth_spectrum = None
        truth_params = None
        if 'sim' in filt:
            if 'TRUTH_SPECTRUM' in atm[p].keys():
                # print('NOT ERROR: true spectrum found in atmos output')
                truth_spectrum = {'depth':atm[p]['TRUTH_SPECTRUM'],
                                  'wavelength':atm[p]['TRUTH_WAVELENGTH']}
                truth_params = atm[p]['TRUTH_MODELPARAMS']
            else:
                print('ERROR: true spectrum is missing from the atmos output')
        elif 'TRUTH_SPECTRUM' in atm[p].keys():
            print('ERROR: true spectrum is present for non-simulated data')

        # print('results',atm[p]['TEC'].keys())
        # print('results',atm[p]['TEC']['MCTRACE'].keys())

        # print('T',np.median(atm[p]['TEC']['MCTRACE']['T']))
        # if not noClouds:
            # print('CTP',np.median(atm[p]['TEC']['MCTRACE']['CTP']))
            # print('Hscale',np.median(atm[p]['TEC']['MCTRACE']['HScale']))
            # print('HLoc  ',np.median(atm[p]['TEC']['MCTRACE']['HLoc']))
            # print('HThick',np.median(atm[p]['TEC']['MCTRACE']['HThick']))
        # print('TEC0',np.median(atm[p]['TEC']['MCTRACE']['TEC[0]']))
        # print('TEC1',np.median(atm[p]['TEC']['MCTRACE']['TEC[1]']))
        # print('TEC2',np.median(atm[p]['TEC']['MCTRACE']['TEC[2]']))

        tprtrace = atm[p]['TEC']['MCTRACE']['T']
        mdplist = [key for key in atm[p]['TEC']['MCTRACE'] if 'TEC' in key]
        mdptrace = []
        # print('mdplist',mdplist)
        for key in mdplist: mdptrace.append(atm[p]['TEC']['MCTRACE'][key])
        if not noClouds:
            ctptrace = atm[p]['TEC']['MCTRACE']['CTP']
            hzatrace = atm[p]['TEC']['MCTRACE']['HScale']
            hloctrace = atm[p]['TEC']['MCTRACE']['HLoc']
            hthicktrace = atm[p]['TEC']['MCTRACE']['HThick']
            ctp = np.median(ctptrace)
            hza = np.median(hzatrace)
            hloc = np.median(hloctrace)
            hthc = np.median(hthicktrace)
            # print('fit results; CTP:',ctp)
            # print('fit results; Hscale:',hza)
            # print('fit results; HLoc:',hloc)
            # print('fit results; HThick:',hthc)
        else:
            ctp = 3.
            hza = 0.
            hloc = 0.
            hthc = 0.
        tpr = np.median(tprtrace)
        mdp = np.median(np.array(mdptrace), axis=1)
        # print('fit results; T:',tpr)
        # print('fit results; mdplist:',mdp)

        solidr = fin['priors'][p]['rp'] * ssc['Rjup']

        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
        tceqdict['NtoO'] = float(mdp[2])

        # print('CONFIRMING xsl keys',xsl[p].keys())  # (XSECS, QTGRID)

        crbhzlib = {'PROFILE':[]}
        hazedir = os.path.join(excalibur.context['data_dir'], 'CERBERUS/HAZE')
        hazelib(crbhzlib, hazedir=hazedir, verbose=False)

        fmc = np.zeros(transitdata['depth'].size)
        fmc = crbmodel(None, float(hza), float(ctp),
                       solidr, fin['priors'],
                       xsl[p]['XSECS'], xsl[p]['QTGRID'],
                       float(tpr),
                       transitdata['wavelength'],
                       hzlib=crbhzlib, hzp='AVERAGE', hztop=float(hloc),
                       hzwscale=float(hthc), cheq=tceqdict, pnet=p,
                       sphshell=True, verbose=False, debug=False)
        # print('median fmc',np.nanmedian(fmc))
        # print('mean model',np.nanmean(fmc))
        # print('mean data',np.nanmean(transitdata['depth']))
        patmos_model = fmc - np.nanmean(fmc) + np.nanmean(transitdata['depth'])
        # print('median pmodel',np.nanmedian(patmos_model))

        # make an array of 10 random walker results
        nrandomwalkers = 10
        nrandomwalkers = 100
        fmcarray = []
        for _ in range(nrandomwalkers):
            iwalker = int(len(tprtrace) * np.random.rand())
            # iwalker = max(0, len(tprtrace) - 1 - int(1000* np.random.rand()))
            if not noClouds:
                ctp = ctptrace[iwalker]
                hza = hzatrace[iwalker]
                hloc = hloctrace[iwalker]
                hthc = hthicktrace[iwalker]
            tpr = tprtrace[iwalker]
            mdp = np.array(mdptrace)[:,iwalker]
            # print('shape mdp',mdp.shape)
            # if not noClouds:
            #    print('fit results; CTP:',ctp)
            #    print('fit results; Hscale:',hza)
            #    print('fit results; HLoc:',hloc)
            #    print('fit results; HThick:',hthc)
            # print('fit results; T:',tpr)
            # print('fit results; mdplist:',mdp)

            tceqdict = {}
            tceqdict['XtoH'] = float(mdp[0])
            tceqdict['CtoO'] = float(mdp[1])
            tceqdict['NtoO'] = float(mdp[2])

            fmcrand = np.zeros(transitdata['depth'].size)
            fmcrand = crbmodel(None, float(hza), float(ctp),
                               solidr, fin['priors'],
                               xsl[p]['XSECS'], xsl[p]['QTGRID'],
                               float(tpr),
                               transitdata['wavelength'],
                               hzlib=crbhzlib, hzp='AVERAGE', hztop=float(hloc),
                               hzwscale=float(hthc), cheq=tceqdict, pnet=p,
                               sphshell=True, verbose=False, debug=False)
            # print('len',len(fmcrand))
            # print('median fmc',np.nanmedian(fmcrand))
            # print('mean model',np.nanmean(fmcrand))
            # print('stdev model',np.nanstd(fmcrand))
            fmcarray.append(fmcrand)

        # _______________MAKE SOME PLOTS________________
        out['data'][p] = {}
        saveDir = os.path.join(excalibur.context['data_dir'], 'bryden/')
        # print('saveDir',saveDir)

        # _______________BEST-FIT SPECTRUM PLOT________________
        transitdata = rebinData(transitdata)
        out['data'][p]['plot_spectrum'] = plot_bestfit(transitdata,
                                                       patmos_model, fmcarray,
                                                       truth_spectrum,
                                                       filt, trgt, p, saveDir)

        # _______________CORNER PLOT________________
        out['data'][p]['plot_corner'] = plot_corner(allkeys, alltraces,
                                                    prior_ranges,
                                                    # truth_params,
                                                    filt, trgt, p, saveDir)

        # _______________VS-PRIOR PLOT________________
        out['data'][p]['plot_vsprior'] = plot_vsPrior(allkeys, alltraces,
                                                      truth_params, prior_ranges,
                                                      filt, trgt, p, saveDir)

        # _______________WALKER-EVOLUTION PLOT________________
        out['data'][p]['plot_walkerevol'] = plot_walkerEvolution(allkeys, alltraces,
                                                                 truth_params, prior_ranges,
                                                                 filt, trgt, p, saveDir)

        out['target'].append(trgt)
        out['planets'].append(p)
        completed_at_least_one_planet = True
        # print('out-data keys at end of this planet',out['data'][p].keys())

    if completed_at_least_one_planet: out['STATUS'].append(True)
    return out['STATUS'][-1]
# --------------------------------------------------------------------
def analysis(aspects, out, verbose=False):
    '''
    Plot out the population analysis (retrieval vs truth, mass-metallicity, etc)
    aspects: cross-target information
    out [INPUT/OUTPUT]
    verbose [OPTIONAL]: verbosity
    '''
    if verbose: print('cerberus/analysis...')

    svname = 'cerberus.atmos'
    filts = ['Ariel-sim']
    targetlists = get_target_lists()

    for filt in filts:
        param_names = []
        truth_values = defaultdict(list)
        fit_values = defaultdict(list)
        fit_errors = defaultdict(list)

        # for trgt in filter(lambda tgt: 'STATUS' in aspects[tgt][svname+'.'+filt], targetlists['active']):
        # JenkinsPEP8 needs this param outside loop
        # svname_with_filter = svname+'.'+filt
        # for trgt in filter(lambda tgt: 'STATUS' in aspects[tgt][svname_with_filter], targetlists['active']):
        # nope! still not jenkins compatible. arg!
        for trgt in targetlists['active']:
            if 'STATUS' in aspects[trgt][svname+'.'+filt]:
                # print('target with valid data format for this filter:',trgt)

                system_data = aspects[trgt][svname+'.'+filt]

                # verify SV succeeded for target
                if system_data['STATUS'][-1]:
                    for planetLetter in system_data['data'].keys():
                        if 'TEC' not in system_data['data'][planetLetter]['MODELPARNAMES'].keys():
                            print(' BIG PROBLEM theres no TEC model!')
                        elif 'TRUTH_MODELPARAMS' not in system_data['data'][planetLetter].keys():
                            print(' TEMP PROBLEM theres no truth info')
                        else:

                            alltraces = []
                            allkeys = []
                            for key in system_data['data'][planetLetter]['TEC']['MCTRACE']:
                                alltraces.append(system_data['data'][planetLetter]['TEC']['MCTRACE'][key])

                                if key=='TEC[0]': allkeys.append('[X/H]')
                                elif key=='TEC[1]': allkeys.append('[C/O]')
                                elif key=='TEC[2]': allkeys.append('[N/O]')
                                else: allkeys.append(key)

                            for key,trace in zip(allkeys,alltraces):
                                if key not in param_names: param_names.append(key)
                                fit_values[key].append(np.median(trace))
                                lo = np.percentile(np.array(trace), 16)
                                hi = np.percentile(np.array(trace), 84)
                                fit_errors[key].append((hi-lo)/2)

                            if isinstance(system_data['data'][planetLetter]['TRUTH_MODELPARAMS'], dict):
                                truth_params = system_data['data'][planetLetter]['TRUTH_MODELPARAMS'].keys()
                                # print('truth keys:',system_data['data'][planetLetter]['TRUTH_MODELPARAMS'].keys())
                            else:
                                truth_params = []
                                print('CORRUPTED TRUTH (bad dict trouble)')

                            for trueparam,fitparam in zip(['Teq','metallicity','C/O','Mp'],
                                                          ['T','[X/H]','[C/O]','Mp']):
                                if trueparam in truth_params:
                                    true_value = system_data['data'][planetLetter]['TRUTH_MODELPARAMS'][trueparam]
                                    # careful here: metallicity and C/O have to be converted to log-solar
                                    if trueparam=='metallicity':
                                        true_value = np.log10(true_value)
                                    elif trueparam=='C/O':
                                        true_value = np.log10(true_value/0.55)  # solar is C/O=0.55 ?
                                    truth_values[fitparam].append(true_value)
                                else:
                                    truth_values[fitparam].append(666)
                            # print('fits',dict(fit_values))
                            # print('truths',dict(truth_values))
                            # print()

    # plot analysis of the results.  save as png and as state vector for states/view
    saveDir = os.path.join(excalibur.context['data_dir'], 'bryden/')
    # jenkins doesn't like to have a triple-packed return here because it's fussy
    plotarray = plot_fitsVStruths(
        truth_values, fit_values, fit_errors, saveDir)
    fitTplot, fitMetalplot, fitCOplot = plotarray[0],plotarray[1],plotarray[2]
    massMetalsplot = plot_massVSmetals(
        truth_values, fit_values, fit_errors, saveDir)

    # save the analysis as .csv file? (in /proj/data/spreadsheets/)
    # savesv(aspects, targetlists)

    # Add to SV
    out['data']['params'] = param_names
    out['data']['truths'] = dict(truth_values)
    out['data']['values'] = dict(fit_values)
    out['data']['errors'] = dict(fit_errors)
    out['data']['plot_fitT'] = fitTplot
    out['data']['plot_fitMetal'] = fitMetalplot
    out['data']['plot_fitCO'] = fitCOplot
    out['data']['plot_massVmetals'] = massMetalsplot
    out['STATUS'].append(True)
    return out['STATUS'][-1]
