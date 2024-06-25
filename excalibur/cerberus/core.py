'''cerberus core ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import excalibur
# pylint: disable=import-self
# import excalibur.cerberus.core  # is this still needed?
import excalibur.system.core as syscore
from excalibur.target.targetlists import get_target_lists
# from excalibur.cerberus.core import savesv
from excalibur.cerberus.forwardModel import \
    ctxtupdt, absorb, \
    crbmodel, fmcerberus, spshfmcerberus, clearfmcerberus, offcerberus, \
    offcerberus1, offcerberus2, offcerberus3, offcerberus4, \
    offcerberus5, offcerberus6, offcerberus7, offcerberus8
from excalibur.cerberus.plotting import rebinData, plot_bestfit, \
    plot_corner, plot_vsPrior, plot_walkerEvolution, \
    plot_fitsVStruths, plot_fitUncertainties, plot_massVSmetals
from excalibur.cerberus.bounds import setPriorBound, getProfileLimits_HSTG141, applyProfiling

import logging
log = logging.getLogger(__name__)
pymc3log = logging.getLogger('pymc3')
pymc3log.setLevel(logging.ERROR)

import os
# import sys
import pymc3 as pm
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from collections import defaultdict

from scipy.interpolate import interp1d as itp

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
    # logarithmicOpacitySumming = True
    logarithmicOpacitySumming = False
    cs = False
    planetLetters = []
    for p in spc['data'].keys():
        if len(p)==1:  # filter out non-planetletter keywords, e.g. 'models','target'
            if 'WB' in spc['data'][p].keys():  # make sure it has a spectrum (Kepler-37e bug)
                planetLetters.append(p)
            else:
                log.warning('--< CERBERUS.XSLIB: wavelength grid is missing for %s %s >--', spc['data']['target'],p)
    for p in planetLetters:
        out['data'][p] = {}

        # model has to be specified, if there is a list of models
        # if 'models' in spc['data'].keys():
        #    arielModel = spc['data']['models'][0]  # arbitrary model choice; all have same WB grid
        #    wgrid = np.array(spc['data'][p][arielModel]['WB'])
        # else:

        wgrid = np.array(spc['data'][p]['WB'])
        qtgrid = gettpf(tips, knownspecies)
        library = {}

        # EDIT HERE!
        # print('cerb core  spc keys',spc['data'][p]['WB'])
        # exit()

        nugrid = (1e4/np.copy(wgrid))[::-1]
        dwnu = np.concatenate((np.array([np.diff(nugrid)[0]]), np.diff(nugrid)))
        for myexomol in xmspecies:
            # log.warning('>-- %s', str(myexomol))
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
                    if logarithmicOpacitySumming:
                        # linearsum = np.sum(sigma2[select])
                        logmean = np.mean(np.log(sigma2[select]))
                        Nbin = np.sum(select)
                        # print('Nbin',Nbin)
                        bini.append(Nbin * np.exp(logmean))
                        # print('old,new',linearsum,Nbin * np.exp(logmean),
                        #      'Nbin,min,max',Nbin,np.min(sigma2[select]),np.max(sigma2[select]))
                    else:
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
                plt.savefig(excalibur.context['data_dir']+'/bryden/'+myexomol+'_xslib.png', dpi=200)
                plt.show()
                pass
            pass
        for mycia in cialist:
            # log.warning('>-- %s', str(mycia))
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
                        if logarithmicOpacitySumming:
                            logmean = np.mean(np.log(sigma2[select]))
                            Nbin = np.sum(select)
                            bini.append(Nbin * np.exp(logmean))
                        else:
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
            # log.warning('>-- %s', str(ks))
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
            # increase the number of scale heights from 15 to 20, to match the Ariel forward model
            # (this is the range used for xslib; also has to be set for atmos)
            Hsmax = 20.
            nlevels = 100.
            pgrid = np.arange(np.log(solrad)-Hsmax, np.log(solrad)+Hsmax/nlevels,
                              Hsmax/(nlevels-1))
            pgrid = np.exp(pgrid)
            pressuregrid = pgrid[::-1]
            allxsections = []
            allwavenumbers = []
            alltemperatures = []
            for Tstep in np.arange(300, 2000, 100):
                # log.warning('>---- %s K', str(Tstep))
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
                    if logarithmicOpacitySumming:
                        logmean = np.mean(np.log(sigma2[select]))
                        Nbin = np.sum(select)
                        bini.append(Nbin * np.exp(logmean))
                    else:
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
    G. ROUDIER: Cerberus retrieval
    '''
    fitCloudParameters = True
    # fitNtoO = True
    fitNtoO = False
    fitCtoO = True
    fitT = True
    am = False
    orbp = fin['priors'].copy()

    ssc = syscore.ssconstants(mks=True)
    crbhzlib = {'PROFILE':[]}
    hazelib(crbhzlib, hazedir=hazedir, verbose=False)
    # SELECT WHICH MODELS TO RUN FOR THIS FILTER
    if ext=='Ariel-sim':
        modfam = ['TEC']  # Ariel sims are currently only TEC equilibrium models
        # modparlbl = {'TEC':['XtoH', 'CtoO']}
        modparlbl = {'TEC':['XtoH', 'CtoO', 'NtoO']}
        # ** select which Ariel model to fit.  there are 8 options **
        # atmosModels = ['cerberus', 'cerberusNoclouds',
        #               'cerberuslowmmw', 'cerberuslowmmwNoclouds',
        #               'taurex', 'taurexNoclouds',
        #               'taurexlowmmw', 'taurexlowmmwNoclouds']
        arielModel = 'cerberusNoclouds'
        # arielModel = 'cerberus'
        # arielModel = 'taurex'

        # if the ariel sim doesn't have clouds, then don't fit the clouds
        fitCloudParameters = 'Noclouds' not in arielModel
        #  OR
        # don't fit the 4 cloud parameters, even if the model has clouds
        # fitCloudParameters = False
        #  OR
        # do fit the 4 cloud parameters, even if the model doesn't have cloud
        fitCloudParameters = True

        # let's fix N/O; one less param to worry about
        # fitNtoO = False
        if not fitNtoO:
            modparlbl = {'TEC':['XtoH', 'CtoO']}
        # let's fix C/O, to see if this parameter is the problem
        # fitCtoO = False
        if not fitCtoO:
            modparlbl = {'TEC':['XtoH']}
        # maybe fix the planet temperature?
        # fitT = False  # not tested yet

        # print('name of the forward model:',arielModel)
        # print('fitCloudParameters for retrieved model:',fitCloudParameters)
        # print('available models',spc['data']['models'])
        if arielModel not in spc['data']['models']:
            log.warning('--< BIG PROB: ariel model doesnt exist!!! >--')
    else:
        modfam = ['TEC', 'PHOTOCHEM']
        modparlbl = {'TEC':['XtoH', 'CtoO', 'NtoO'],
                     'PHOTOCHEM':['HCN', 'CH4', 'C2H2', 'CO2', 'H2CO']}

    if (singlemod is not None) and (singlemod in modfam):
        modfam = [modfam[modfam.index(singlemod)]]

    # PLANET LOOP
    for p in spc['data'].keys():
        # make sure that it really is a planet letter, not another dict key
        #  (ariel has other keys, e.g. 'target', 'planets', 'models')
        # if len(p) > 1:
        #    log.warning('--< OK: skipping a planet letter that is actually a system keyword: %s >--',p)
        #    pass
        # elif len(p)==1:
        # make sure it has a spectrum (Kepler-37e bug)
        if len(p)==1 and 'WB' not in spc['data'][p].keys():
            log.warning('--< CERBERUS.ATMOS: wavelength grid is missing for %s %s >--',spc['data']['target'],p)
        elif len(p)==1 and 'WB' in spc['data'][p].keys():
            if ext=='Ariel-sim':
                if arielModel in spc['data'][p].keys():
                    inputData = spc['data'][p][arielModel]
                    # make sure that the wavelength is saved in usual location
                    # (the cerberus forward models expect it to come after [p])
                    # spc['data'][p]['WB'] = spc['data'][p][arielModel]['WB']
                    inputData['WB'] = spc['data'][p]['WB']
                else:
                    log.warning('--< THIS arielModel DOESNT EXIST!!! (rerun ariel task?) >--')
            else:
                inputData = spc['data'][p]

            out['data'][p] = {}
            out['data'][p]['MODELPARNAMES'] = modparlbl

            # save the planet params (mass), so that analysis can make a mass-metallicity plot
            out['data'][p]['planet_params'] = orbp[p]
            # save the tier and #-of-visits (for Ariel-sim targets), for plot labelling
            if ext=='Ariel-sim':
                out['data'][p]['tier'] = spc['data'][p][arielModel]['tier']
                out['data'][p]['visits'] = spc['data'][p][arielModel]['visits']

            # eqtemp1 = orbp['T*']*np.sqrt(orbp['R*']*ssc['Rsun/AU']/(2.*orbp[p]['sma']))
            # use of L* might be better than T*,R*; it's more of a direct observable
            # eqtemp2 = 278.6 * orbp['L*']**0.25 / np.sqrt(orbp[p]['sma'])

            # These four estimates of T_equilibrium are usually about the same
            #  but sometimes they are very different
            #  system/consistency.py checks for and flags these inconsistencies
            # print('eqtemp check',eqtemp1)
            # print('eqtemp check',eqtemp2)
            # print('eqtemp check',orbp[p]['teq'])
            # print('eqtemp check',inputData['model_params']['Teq'])

            # CAREFUL:
            #  equilibrium temperatures from the archive sometimes don't match this!
            #  e.g. GJ 3053=LHS 1140 b,c (the Archive has 379K,709K vs 216,403K here)
            #  that one is wrong it seems. Lillo-Box 2020 has strange extra factor

            # print('model_params',inputData['model_params'])

            # bottom line:
            #  use the same Teq as in ariel-sim, otherwise truth/retrieved won't match
            if ext=='Ariel-sim':
                eqtemp = inputData['model_params']['Teq']
            else:
                # (real data doesn't have any 'model_params' defined)
                # eqtemp = orbp['T*']*np.sqrt(orbp['R*']*ssc['Rsun/AU']/(2.*orbp[p]['sma']))
                eqtemp = float(orbp[p]['teq'])

            tspc = np.array(inputData['ES'])
            terr = np.array(inputData['ESerr'])
            twav = np.array(inputData['WB'])
            # twav = np.array(spc['data'][p]['WB'])

            tspecerr = abs(tspc**2 - (tspc + terr)**2)
            tspectrum = tspc**2
            if 'STIS-WFC3' in ext:
                filters = np.array(inputData['Fltrs'])
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
            Hs = inputData['Hs']
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

                # new method for setting priors (no change, but easier to view in bounds.py)
                priorRangeTable = setPriorBound()

                out['data'][p][model]['prior_ranges'] = {}
                # keep track of the bounds put on each parameter
                # this will be helpful for later plotting and analysis
                prior_ranges = {}
                nodes = []
                # with pm.Model() as pmmodel:
                with pm.Model():
                    fixedParams = {}

                    if fitCloudParameters:
                        # CLOUD TOP PRESSURE
                        # prior_ranges['CTP'] = (-6,1)
                        prior_ranges['CTP'] = priorRangeTable['CTP']
                        ctp = pm.Uniform('CTP', prior_ranges['CTP'][0], prior_ranges['CTP'][1])
                        nodes.append(ctp)

                        # HAZE SCAT. CROSS SECTION SCALE FACTOR
                        # prior_ranges['HScale'] = (-6,6)
                        prior_ranges['HScale'] = priorRangeTable['HScale']
                        hza = pm.Uniform('HScale', prior_ranges['HScale'][0], prior_ranges['HScale'][1])
                        nodes.append(hza)
                    else:
                        # print('model_params',inputData['model_params'])
                        fixedParams['CTP'] = inputData['model_params']['CTP']
                        fixedParams['HScale'] = inputData['model_params']['HScale']

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
                    if sphshell:
                        if fitCloudParameters:
                            # prior_ranges['HLoc'] = (-6,1)
                            prior_ranges['HLoc'] = priorRangeTable['HLoc']
                            hzloc = pm.Uniform('HLoc',
                                               prior_ranges['HLoc'][0], prior_ranges['HLoc'][1])
                            nodes.append(hzloc)

                            # prior_ranges['HThick'] = (1,20)
                            prior_ranges['HThick'] = priorRangeTable['HThick']
                            hzthick = pm.Uniform('HThick',
                                                 prior_ranges['HThick'][0],prior_ranges['HThick'][1])
                            nodes.append(hzthick)
                        else:
                            fixedParams['HLoc'] = inputData['model_params']['HLoc']
                            fixedParams['HThick'] = inputData['model_params']['HThick']
                    else:
                        if fitCloudParameters:
                            # prior_ranges['HIndex'] = (-4,0)
                            prior_ranges['HIndex'] = priorRangeTable['HIndex']
                            hzi = pm.Uniform('HIndex',
                                             prior_ranges['HIndex'][0], prior_ranges['HIndex'][1])
                            nodes.append(hzi)
                        else:
                            fixedParams['HIndex'] = -2

                    # BOOST TEMPERATURE PRIOR TO [75%, 150%] Teq
                    if fitT:
                        # prior_ranges['T'] = (0.75e0*eqtemp, 1.5e0*eqtemp)
                        prior_ranges['T'] = (priorRangeTable['Tfactor'][0] * eqtemp,
                                             priorRangeTable['Tfactor'][1] * eqtemp)
                        tpr = pm.Uniform('T', prior_ranges['T'][0], prior_ranges['T'][1])
                        nodes.append(tpr)
                    else:
                        fixedParams['T'] = inputData['model_params']['T']

                    # MODEL SPECIFIC ABSORBERS
                    for param in modparlbl[model]:
                        # dexRange = (-6,6)
                        dexRange = priorRangeTable['dexRange']
                        if param=='XtoH':
                            prior_ranges['[X/H]'] = dexRange
                        elif param=='CtoO':
                            prior_ranges['[C/O]'] = dexRange
                        elif param=='NtoO':
                            prior_ranges['[N/O]'] = dexRange
                        else:
                            prior_ranges[param] = dexRange
                    numAbundanceParams = len(modparlbl[model])
                    # make sure that there's at least two parameters here, or the decorator crashes
                    # if fitCtoO: numAbundanceParams += 1
                    numAbundanceParams = max(numAbundanceParams, 2)
                    # print('numAbundanceParams',numAbundanceParams)
                    modelpar = pm.Uniform(model, lower=dexRange[0], upper=dexRange[1],
                                          shape=numAbundanceParams)
                    nodes.append(modelpar)
                    if not fitNtoO:
                        fixedParams['NtoO'] = 0.
                    if not fitCtoO:
                        # print('model params',inputData['model_params'])
                        fixedParams['CtoO'] = inputData['model_params']['C/O']

                    # before calling MCMC, save the fixed-parameter info in the context
                    ctxtupdt(cleanup=cleanup, model=model, p=p, solidr=solidr, orbp=orbp,
                             tspectrum=tspectrum, xsl=xsl, spc=spc, modparlbl=modparlbl,
                             hzlib=crbhzlib, fixedParams=fixedParams)

                    # CERBERUS MCMC
                    if not fitCloudParameters:
                        log.warning('--< RUNNING MCMC - NO CLOUDS! >--')
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
                            log.warning('--< STANDARD MCMC (WITH CLOUDS) >--')
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
                    else:
                        mctrace[key] = trace[tracekeys[0]]
                    pass
                out['data'][p][model]['MCTRACE'] = mctrace

                out['data'][p][model]['prior_ranges'] = prior_ranges
            # out['data'][p]['WAVELENGTH'] = np.array(spc['data'][p]['WB'])
            out['data'][p]['WAVELENGTH'] = np.array(inputData['WB'])
            out['data'][p]['SPECTRUM'] = np.array(inputData['ES'])
            out['data'][p]['ERRORS'] = np.array(inputData['ESerr'])
            if ext=='Ariel-sim':
                if 'true_spectrum' in inputData.keys():

                    out['data'][p]['TRUTH_SPECTRUM'] = np.array(inputData['true_spectrum']['fluxDepth'])
                    # wavelength should be the same as just above, but just in case load it here too
                    out['data'][p]['TRUTH_WAVELENGTH'] = np.array(inputData['true_spectrum']['wavelength_um'])
                    out['data'][p]['TRUTH_MODELPARAMS'] = inputData['model_params']
                    # print('true modelparams in atmos:',inputData['model_params'])
            out['data'][p]['VALID'] = cleanup
            out['STATUS'].append(True)
            am = True
            pass
    return am
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
def results(trgt, filt, fin, anc, xsl, atm, out, verbose=False):
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

    # load in the table of limits used for profiling
    if filt=='HST-WFC3-IR-G141-SCAN':
        profilingLimits = getProfileLimits_HSTG141()
    else:
        profilingLimits = []

    for p in fin['priors']['planets']:
        # print('post-analysis for planet:',p)

        # TEC params - X/H, C/O, N/O
        # disEq params - HCN, CH4, C2H2, CO2, H2CO

        # check whether this planet was analyzed
        # (some planets are skipped, because they have an unbound atmosphere)
        if p not in atm.keys():
            log.warning('>-- CERBERUS.RESULTS: this planet is missing cerb fit: %s %s', trgt, p)

        else:
            out['data'][p] = {}

            # limit results to just the TEC model?  No, not for HST/G141
            # but do verify that TEC exists at least
            if 'TEC' not in atm[p]['MODELPARNAMES'].keys():
                log.warning('>-- %s', 'TROUBLE: theres no TEC fit!?')
                return False

            # there was a bug before where PHOTOCHEM was passed in for Ariel
            # just in case, filter out models that are missing
            models = []
            for modelName in atm[p]['MODELPARNAMES']:
                if modelName in atm[p].keys(): models.append(modelName)
            for modelName in models:
                # for modelName in atm[p]['MODELPARNAMES']:
                allTraces = []
                allKeys = []
                for key in atm[p][modelName]['MCTRACE']:
                    # print('fillin tru the keys again?',key)
                    allTraces.append(atm[p][modelName]['MCTRACE'][key])
                    if modelName=='TEC':
                        if key=='TEC[0]': allKeys.append('[X/H]')
                        elif key=='TEC[1]': allKeys.append('[C/O]')
                        elif key=='TEC[2]': allKeys.append('[N/O]')
                        else: allKeys.append(key)
                    elif modelName=='PHOTOCHEM':
                        if key=='PHOTOCHEM[0]': allKeys.append('HCN')
                        elif key=='PHOTOCHEM[1]': allKeys.append('CH4')
                        elif key=='PHOTOCHEM[2]': allKeys.append('C2H2')
                        elif key=='PHOTOCHEM[3]': allKeys.append('CO2')
                        elif key=='PHOTOCHEM[4]': allKeys.append('H2CO')
                        else: allKeys.append(key)
                    else:
                        allKeys.append(key)
                    # print('allKeys',allKeys)

                # remove the traced phase space that is excluded by profiling
                profileTrace, appliedLimits = applyProfiling(
                    trgt+' '+p, profilingLimits, allTraces, allKeys)
                keepers = np.where(profileTrace==1)
                profiledTraces = []
                for key in atm[p][modelName]['MCTRACE']:
                    profiledTraces.append(atm[p][modelName]['MCTRACE'][key][keepers])
                profiledTraces = np.array(profiledTraces)

                # make note of the bounds placed on each parameter
                if 'prior_ranges' in atm[p][modelName].keys():
                    prior_ranges = atm[p][modelName]['prior_ranges']
                else:
                    prior_ranges = {}

                fitCloudParameters = 'CTP' in allKeys
                fitNtoO = '[N/O]' in allKeys
                fitCtoO = '[C/O]' in allKeys
                fitT = 'T' in allKeys

                # ndim = len(allTraces)
                # nsamples = len(allTraces[0])
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

                # print('results',atm[p][modelName].keys())
                # print('results',atm[p][modelName]['MCTRACE'].keys())

                # print('T',np.median(atm[p][modelName]['MCTRACE']['T']))
                # if fitCloudParameters:
                #     print('CTP',np.median(atm[p][modelName]['MCTRACE']['CTP']))
                #     print('HScale',np.median(atm[p][modelName]['MCTRACE']['HScale']))
                #     print('HLoc  ',np.median(atm[p][modelName]['MCTRACE']['HLoc']))
                #     print('HThick',np.median(atm[p][modelName]['MCTRACE']['HThick']))
                # print('TEC0',np.median(atm[p][modelName]['MCTRACE']['TEC[0]']))
                # print('TEC1',np.median(atm[p][modelName]['MCTRACE']['TEC[1]']))
                # print('TEC2',np.median(atm[p][modelName]['MCTRACE']['TEC[2]']))

                tprtrace = atm[p][modelName]['MCTRACE']['T']
                tprtraceProfiled = atm[p][modelName]['MCTRACE']['T'][keepers]
                mdplist = [key for key in atm[p][modelName]['MCTRACE'] if modelName in key]
                # print('mdplist',mdplist)
                mdptrace = []
                mdptraceProfiled = []
                for key in mdplist: mdptrace.append(atm[p][modelName]['MCTRACE'][key])
                for key in mdplist: mdptraceProfiled.append(atm[p][modelName]['MCTRACE'][key][keepers])
                if fitCloudParameters:
                    ctptrace = atm[p][modelName]['MCTRACE']['CTP']
                    hzatrace = atm[p][modelName]['MCTRACE']['HScale']
                    hloctrace = atm[p][modelName]['MCTRACE']['HLoc']
                    hthicktrace = atm[p][modelName]['MCTRACE']['HThick']
                    ctp = np.median(ctptrace)
                    hza = np.median(hzatrace)
                    hloc = np.median(hloctrace)
                    hthc = np.median(hthicktrace)
                    # print('fit results; CTP:',ctp)
                    # print('fit results; HScale:',hza)
                    # print('fit results; HLoc:',hloc)
                    # print('fit results; HThick:',hthc)
                    ctptraceProfiled = atm[p][modelName]['MCTRACE']['CTP'][keepers]
                    hzatraceProfiled = atm[p][modelName]['MCTRACE']['HScale'][keepers]
                    hloctraceProfiled = atm[p][modelName]['MCTRACE']['HLoc'][keepers]
                    hthicktraceProfiled = atm[p][modelName]['MCTRACE']['HThick'][keepers]
                    ctpProfiled = np.median(ctptraceProfiled)
                    hzaProfiled = np.median(hzatraceProfiled)
                    hlocProfiled = np.median(hloctraceProfiled)
                    hthcProfiled = np.median(hthicktraceProfiled)
                else:
                    # ctp = 3.
                    # hza = 10.
                    # hloc = 0.
                    # hthc = 0.
                    # ctp = -1.52
                    # hza = -2.1
                    # hloc = -2.3
                    # hthc = 9.76
                    ctp = atm[p]['TRUTH_MODELPARAMS']['CTP']
                    hza = atm[p]['TRUTH_MODELPARAMS']['HScale']
                    hloc = atm[p]['TRUTH_MODELPARAMS']['HLoc']
                    hthc = atm[p]['TRUTH_MODELPARAMS']['HThick']
                    ctpProfiled = ctp
                    hzaProfiled = hza
                    hlocProfiled = hloc
                    hthcProfiled = hthc
                    # print(' ctp hza hloc hthc',ctp,hza,hloc,hthc)
                if fitT:
                    tpr = np.median(tprtrace)
                    tprProfiled = np.median(tprtraceProfiled)
                else:
                    tpr = atm[p]['TRUTH_MODELPARAMS']['T']
                    tprProfiled = tpr
                mdp = np.median(np.array(mdptrace), axis=1)
                mdpProfiled = np.median(np.array(mdptraceProfiled), axis=1)
                # print('fit results; T:',tpr)
                # print('fit results; mdplist:',mdp)

                solidr = fin['priors'][p]['rp'] * ssc['Rjup']

                if modelName=='TEC':
                    # if len(mdp)!=3: log.warning('--< Expecting 3 molecules for TEQ model! >--')
                    mixratio = None
                    mixratioProfiled = None
                    tceqdict = {}
                    tceqdictProfiled = {}
                    tceqdict['XtoH'] = float(mdp[0])
                    tceqdictProfiled['XtoH'] = float(mdpProfiled[0])
                    if fitCtoO:
                        tceqdict['CtoO'] = float(mdp[1])
                        tceqdictProfiled['CtoO'] = float(mdpProfiled[1])
                    else:
                        # print('truth params',atm[p]['TRUTH_MODELPARAMS'])
                        tceqdict['CtoO'] = atm[p]['TRUTH_MODELPARAMS']['CtoO']
                        tceqdictProfiled['CtoO'] = tceqdict['CtoO']
                    if fitNtoO:
                        tceqdict['NtoO'] = float(mdp[2])
                        tceqdictProfiled['NtoO'] = float(mdpProfiled[2])
                    else:
                        # print('truth params',atm[p]['TRUTH_MODELPARAMS'])
                        if 'NtoO' in atm[p]['TRUTH_MODELPARAMS'].keys():
                            tceqdict['NtoO'] = atm[p]['TRUTH_MODELPARAMS']['NtoO']
                        else:
                            tceqdict['NtoO'] = 0.
                        tceqdictProfiled['NtoO'] = tceqdict['NtoO']
                elif modelName=='PHOTOCHEM':
                    if len(mdp)!=5: log.warning('--< Expecting 5 molecules for PHOTOCHEM model! >--')
                    tceqdict = None
                    mixratio = {}
                    mixratio['HCN'] = float(mdp[0])
                    mixratio['CH4'] = float(mdp[1])
                    mixratio['C2H2'] = float(mdp[2])
                    mixratio['CO2'] = float(mdp[3])
                    mixratio['H2CO'] = float(mdp[4])
                    tceqdictProfiled = None
                    mixratioProfiled = {}
                    mixratioProfiled['HCN'] = float(mdpProfiled[0])
                    mixratioProfiled['CH4'] = float(mdpProfiled[1])
                    mixratioProfiled['C2H2'] = float(mdpProfiled[2])
                    mixratioProfiled['CO2'] = float(mdpProfiled[3])
                    mixratioProfiled['H2CO'] = float(mdpProfiled[4])
                else:
                    log.warning('--< Expecting TEQ or PHOTOCHEM model! >--')

                # print('CONFIRMING xsl keys',xsl[p].keys())  # (XSECS, QTGRID)

                crbhzlib = {'PROFILE':[]}
                hazedir = os.path.join(excalibur.context['data_dir'], 'CERBERUS/HAZE')
                hazelib(crbhzlib, hazedir=hazedir, verbose=False)

                fmc = np.zeros(transitdata['depth'].size)
                fmc = crbmodel(mixratio, float(hza), float(ctp),
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

                fmcProfiled = np.zeros(transitdata['depth'].size)
                fmcProfiled = crbmodel(mixratioProfiled, float(hzaProfiled), float(ctpProfiled),
                                       solidr, fin['priors'],
                                       xsl[p]['XSECS'], xsl[p]['QTGRID'],
                                       float(tprProfiled),
                                       transitdata['wavelength'],
                                       hzlib=crbhzlib, hzp='AVERAGE', hztop=float(hlocProfiled),
                                       hzwscale=float(hthcProfiled), cheq=tceqdictProfiled, pnet=p,
                                       sphshell=True, verbose=False, debug=False)
                patmos_modelProfiled = fmcProfiled - np.nanmean(fmcProfiled) + np.nanmean(transitdata['depth'])

                # make an array of 10 random walker results
                nrandomwalkers = 10
                nrandomwalkers = 100
                fmcarray = []
                for _ in range(nrandomwalkers):
                    iwalker = int(len(tprtrace) * np.random.rand())
                    # iwalker = max(0, len(tprtrace) - 1 - int(1000* np.random.rand()))
                    if fitCloudParameters:
                        ctp = ctptrace[iwalker]
                        hza = hzatrace[iwalker]
                        hloc = hloctrace[iwalker]
                        hthc = hthicktrace[iwalker]
                    if fitT:
                        tpr = tprtrace[iwalker]
                    mdp = np.array(mdptrace)[:,iwalker]
                    # print('shape mdp',mdp.shape)
                    # if fitCloudParameters:
                    #    print('fit results; CTP:',ctp)
                    #    print('fit results; HScale:',hza)
                    #    print('fit results; HLoc:',hloc)
                    #    print('fit results; HThick:',hthc)
                    # print('fit results; T:',tpr)
                    # print('fit results; mdplist:',mdp)

                    if modelName=='TEC':
                        mixratio = None
                        tceqdict = {}
                        tceqdict['XtoH'] = float(mdp[0])
                        if fitCtoO:
                            tceqdict['CtoO'] = float(mdp[1])
                        else:
                            tceqdict['CtoO'] = atm[p]['TRUTH_MODELPARAMS']['CtoO']
                        if fitNtoO:
                            tceqdict['NtoO'] = float(mdp[2])
                        else:
                            if 'NtoO' in atm[p]['TRUTH_MODELPARAMS'].keys():
                                tceqdict['NtoO'] = atm[p]['TRUTH_MODELPARAMS']['NtoO']
                            else:
                                # log.warning('--< NtoO is missing from TRUTH_MODELPARAMS >--')
                                tceqdict['NtoO'] = 0.
                    elif modelName=='PHOTOCHEM':
                        tceqdict = None
                        mixratio = {}
                        mixratio['HCN'] = float(mdp[0])
                        mixratio['CH4'] = float(mdp[1])
                        mixratio['C2H2'] = float(mdp[2])
                        mixratio['CO2'] = float(mdp[3])
                        mixratio['H2CO'] = float(mdp[4])

                    fmcrand = np.zeros(transitdata['depth'].size)
                    fmcrand = crbmodel(mixratio, float(hza), float(ctp),
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
                saveDir = os.path.join(excalibur.context['data_dir'], 'bryden/')
                # print('saveDir',saveDir)

                # _______________BEST-FIT SPECTRUM PLOT________________
                transitdata = rebinData(transitdata)

                out['data'][p]['plot_spectrum_'+modelName],_ = plot_bestfit(
                    transitdata, patmos_model, patmos_modelProfiled, fmcarray,
                    truth_spectrum,
                    anc['data'][p], atm[p],
                    filt, modelName, trgt, p, saveDir)

                # _______________CORNER PLOT________________
                out['data'][p]['plot_corner_'+modelName],_ = plot_corner(
                    allKeys, allTraces, profiledTraces,
                    truth_params, prior_ranges,
                    filt, modelName, trgt, p, saveDir)

                # _______________WALKER-EVOLUTION PLOT________________
                out['data'][p]['plot_walkerevol_'+modelName],_ = plot_walkerEvolution(
                    allKeys, allTraces, profiledTraces,
                    truth_params, prior_ranges, appliedLimits,
                    filt, modelName, trgt, p, saveDir)

                # _______________VS-PRIOR PLOT________________
                out['data'][p]['plot_vsprior_'+modelName],_ = plot_vsPrior(
                    allKeys, allTraces, profiledTraces,
                    truth_params, prior_ranges, appliedLimits,
                    filt, modelName, trgt, p, saveDir)

            out['target'].append(trgt)
            out['planets'].append(p)
            completed_at_least_one_planet = True
            # print('out-data keys at end of this planet',out['data'][p].keys())

    if completed_at_least_one_planet: out['STATUS'].append(True)
    return out['STATUS'][-1]
# --------------------------------------------------------------------
def analysis(aspects, filt, out, verbose=False):
    '''
    Plot out the population analysis (retrieval vs truth, mass-metallicity, etc)
    aspects: cross-target information
    out [INPUT/OUTPUT]
    verbose [OPTIONAL]: verbosity
    '''
    if verbose: print('cerberus/analysis...')

    aspecttargets = []
    for a in aspects: aspecttargets.append(a)
    # print('aspects check','TrES-3' in aspecttargets)
    # print('aspects check','WASP-33' in aspecttargets)

    # print('aspects',aspects)
    # print('aspect keys',aspects.keys())
    # for a in aspects.keys(): print(a)
    # for a in aspects: print(a)
    # print('aspects check','TrES-3' in aspects)
    # print('aspects check','WASP-33' in aspects)
    # print('filt',filt)
    # exit('core stop')

    svname = 'cerberus.atmos'

    alltargetlists = get_target_lists()

    # allow for analysis of multiple target lists
    analysistargetlists = []

    if filt=='Ariel-sim':
        # analysistargetlists.append({
        #    'targetlistname':'Roudier+ 2022',
        #    'targets':alltargetlists['roudier62']})
        analysistargetlists.append({
            'targetlistname':'MCS Nov.2023 Transit-list',
            'targets':alltargetlists['arielMCS_Nov2023_transit']})
        analysistargetlists.append({
            'targetlistname':'MCS Nov.2023 max-visits=25',
            'targets':alltargetlists['arielMCS_Nov2023_maxVisits25']})
        # analysistargetlists.append({
        #    'targetlistname':'MCS Feb.2024 Transit-list',
        #    'targets':alltargetlists['arielMCS_Feb2024_transit']})
        # analysistargetlists.append({
        #    'targetlistname':'MCS Feb.2024 max-visits=25',
        #    'targets':alltargetlists['arielMCS_Feb2024_maxVisits25']})
    else:
        # analysistargetlists.append({
        #    'targetlistname':'All Excalibur targets',
        #    'targets':alltargetlists['active']})
        analysistargetlists.append({
            'targetlistname':'Roudier+ 2022',
            'targets':alltargetlists['roudier62']})
        analysistargetlists.append({
            'targetlistname':'All G141 targets',
            'targets':alltargetlists['G141']})

    for targetlist in analysistargetlists:
        param_names = []
        masses = []
        truth_values = defaultdict(list)
        fit_values = defaultdict(list)
        fit_errors = defaultdict(list)

        # for trgt in filter(lambda tgt: 'STATUS' in aspects[tgt][svname+'.'+filt], targetlist['targets']):
        # JenkinsPEP8 needs this param outside loop
        # svname_with_filter = svname+'.'+filt
        # for trgt in filter(lambda tgt: 'STATUS' in aspects[tgt][svname_with_filter], targetlist['targets']):
        # nope! still not jenkins compatible. arg!
        for trgt in targetlist['targets']:
            # print('        cycling through targets',trgt)
            # if trgt not in aspects.keys():
            # if trgt not in aspects:
            if trgt not in aspecttargets:
                log.warning('--< CERBERUS ANALYSIS: TARGET NOT IN ASPECT %s %s >--',filt,trgt)
            elif svname+'.'+filt not in aspects[trgt]:
                # some targets don't have this filter; no problem
                # log.warning('--< NO CERB.ATMOS for this FILTER+TARGET %s %s >--',filt,trgt)
                pass
            elif 'STATUS' not in aspects[trgt][svname+'.'+filt]:
                log.warning('--< CERBERUS ANALYSIS: FORMAT ERROR - NO STATUS %s %s >--',filt,trgt)
            else:
                # print('target with valid data format for this filter:',filt,trgt)
                atmosFit = aspects[trgt][svname+'.'+filt]

                # verify SV succeeded for target
                if not atmosFit['STATUS'][-1]:
                    log.warning('--< CERBERUS ANALYSIS: STATUS IS FALSE FOR CERB.ATMOS %s %s >--',filt,trgt)
                else:
                    for planetLetter in atmosFit['data'].keys():
                        if 'TEC' not in atmosFit['data'][planetLetter]['MODELPARNAMES']:
                            log.warning('--< CERBERUS ANALYSIS: BIG PROBLEM theres no TEC model! %s %s >--',filt,trgt)
                        elif 'prior_ranges' not in atmosFit['data'][planetLetter]['TEC']:
                            log.warning('--< CERBERUS ANALYSIS: SKIP (no prior info) - %s %s >--',filt,trgt)
                        else:
                            if 'planet_params' in atmosFit['data'][planetLetter]:
                                masses.append(atmosFit['data'][planetLetter]['planet_params']['mass'])
                            else:
                                masses.append(666)

                            # (prior range should be the same for all the targets)
                            # print('priors',atmosFit['data'][planetLetter]['TEC']['prior_ranges'])
                            # print('   key check',atmosFit['data'][planetLetter]['TEC'].keys())
                            # if 'prior_ranges' not in atmosFit['data'][planetLetter]['TEC']:
                            #    print('check',atmosFit['data'][planetLetter])
                            prior_ranges = atmosFit['data'][planetLetter]['TEC']['prior_ranges']

                            allTraces = []
                            allKeys = []
                            for key in atmosFit['data'][planetLetter]['TEC']['MCTRACE']:
                                allTraces.append(atmosFit['data'][planetLetter]['TEC']['MCTRACE'][key])

                                if key=='TEC[0]': allKeys.append('[X/H]')
                                elif key=='TEC[1]': allKeys.append('[C/O]')
                                elif key=='TEC[2]': allKeys.append('[N/O]')
                                else: allKeys.append(key)

                            for key,trace in zip(allKeys,allTraces):
                                if key not in param_names: param_names.append(key)
                                fit_values[key].append(np.median(trace))
                                lo = np.percentile(np.array(trace), 16)
                                hi = np.percentile(np.array(trace), 84)
                                fit_errors[key].append((hi-lo)/2)

                            if ('TRUTH_MODELPARAMS' in atmosFit['data'][planetLetter].keys()) and \
                               (isinstance(atmosFit['data'][planetLetter]['TRUTH_MODELPARAMS'], dict)):
                                truth_params = atmosFit['data'][planetLetter]['TRUTH_MODELPARAMS'].keys()
                                # print('truth keys:',truth_params)
                            else:
                                truth_params = []

                            for trueparam,fitparam in zip(['Teq','metallicity','C/O','N/O','Mp'],
                                                          ['T','[X/H]','[C/O]','[N/O]','Mp']):
                                if trueparam in truth_params:
                                    true_value = atmosFit['data'][planetLetter]['TRUTH_MODELPARAMS'][trueparam]
                                    # (metallicity and C/O do not have to be converted to log-solar)
                                    # if trueparam=='metallicity':
                                    #    true_value = np.log10(true_value)
                                    # elif trueparam=='C/O':
                                    #    true_value = np.log10(true_value/0.54951)  # solar is C/O=0.55
                                    # elif trueparam=='N/O':
                                    #     true_value = true_value
                                    if fitparam=='[N/O]' and true_value==666:
                                        truth_values[fitparam].append(0)
                                    else:
                                        truth_values[fitparam].append(true_value)

                                    if trueparam=='Teq' and true_value > 3333:
                                        print('strangely high T',trgt,true_value)
                                    if trueparam=='metallicity' and true_value > 66:
                                        print('strangely high [X/H]',trgt,true_value)
                                        print('atmosFit',atmosFit['data'][planetLetter])
                                    if trueparam=='C/O' and true_value > 0.5:
                                        print('strangely high [C/O]',trgt,true_value)

                                elif trueparam=='Mp':
                                    # if the planet mass is not in the Truth dictionary, pull it from system
                                    # print(' input keys',atmosFit['data'][planetLetter]['planet_params'])
                                    # print(' planet mass from system params:',
                                    #      atmosFit['data'][planetLetter]['planet_params']['mass'])
                                    truth_values[fitparam].append(
                                        atmosFit['data'][planetLetter]['planet_params']['mass'])
                                elif trueparam=='N/O':
                                    truth_values[fitparam].append(0)
                                else:
                                    truth_values[fitparam].append(666)
                            # print('fits',dict(fit_values))
                            # print('truths',dict(truth_values))
                            # print('truth_params',truth_params)
                            # print(' input keys',atmosFit['data'][planetLetter].keys())
                            # print(' input keys',atmosFit['data'][planetLetter]['planet_params'])
                            # exit('testtt')
                            # print(' input truth',atmosFit['data'][planetLetter]['TRUTH_MODELPARAMS'])
                            # print()

                        # look out for oddballs
                        # print('T,fit,err',truth_values['T'][-1],
                        #      fit_values['T'][-1],fit_errors['T'][-1],
                        #      (fit_values['T'][-1] - truth_values['T'][-1])/fit_errors['T'][-1],trgt)

        # plot analysis of the results.  save as png and as state vector for states/view
        saveDir = os.path.join(excalibur.context['data_dir'], 'bryden/')
        # jenkins doesn't like to have a triple-packed return here because it's fussy
        if 'sim' in filt:
            # for simulated data, compare retrieval against the truth
            plotarray = plot_fitsVStruths(
                truth_values, fit_values, fit_errors, prior_ranges, filt, saveDir)
            fitTplot, fitMetalplot, fitCOplot, fitNOplot = plotarray[0],plotarray[1],plotarray[2],plotarray[3]
        else:
            # for real data, make a histogram of the retrieved uncertainties
            plotarray = plot_fitUncertainties(
                fit_values, fit_errors, prior_ranges, filt, saveDir)
            fitTplot, fitMetalplot, fitCOplot, fitNOplot = plotarray[0],plotarray[1],plotarray[2],plotarray[3]

        masses = truth_values['Mp']
        massMetalsplot,_ = plot_massVSmetals(
            masses, truth_values, fit_values, fit_errors, prior_ranges, filt, saveDir)

        # save the analysis as .csv file? (in /proj/data/spreadsheets/)
        # savesv(aspects, targetlist)

        # targetlistname = targetlist['targetlistname']

        # Add to SV
        out['data']['truths'] = dict(truth_values)
        out['data']['values'] = dict(fit_values)
        out['data']['errors'] = dict(fit_errors)
        out['data']['plot_massVmetals'] = massMetalsplot
        out['data']['plot_fitT'] = fitTplot
        out['data']['plot_fitMetal'] = fitMetalplot
        out['data']['plot_fitCO'] = fitCOplot
        out['data']['plot_fitNO'] = fitNOplot

    out['data']['params'] = param_names
    out['data']['targetlistnames'] = [
        targetlist['targetlistname'] for targetlist in analysistargetlists]

    out['STATUS'].append(True)
    return out['STATUS'][-1]
