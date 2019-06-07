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
import matplotlib.pyplot as plt

import theano.tensor as tt
import theano.compile.ops as tco

import scipy.constants as cst
from scipy.interpolate import interp1d as itp
# -- GLOBAL CONTEXT FOR PYMC3 DETERMINISTICS ---------------------------------------------
from collections import namedtuple
CONTEXT = namedtuple('CONTEXT', ['cleanup', 'model', 'p', 'solidr', 'orbp', 'tspectrum',
                                 'xsl', 'spc', 'modparlbl'])
ctxt = CONTEXT(cleanup=None, model=None, p=None, solidr=None, orbp=None, tspectrum=None,
               xsl=None, spc=None, modparlbl=None)
def ctxtupdt(cleanup=None, model=None, p=None, solidr=None, orbp=None, tspectrum=None,
             xsl=None, spc=None, modparlbl=None):
    '''
G. ROUDIER: Update context
    '''
    excalibur.cerberus.core.ctxt = CONTEXT(cleanup=cleanup, model=model, p=p,
                                           solidr=solidr, orbp=orbp,
                                           tspectrum=tspectrum, xsl=xsl, spc=spc,
                                           modparlbl=modparlbl)
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
def myxsecs(spc, out,
            hitemp=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HITEMP'),
            tips=os.path.join(excalibur.context['data_dir'], 'CERBERUS/TIPS'),
            ciadir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/HITRAN/CIA'),
            exomoldir=os.path.join(excalibur.context['data_dir'], 'CERBERUS/EXOMOL'),
            knownspecies=['NO', 'OH', 'C2H2', 'N2', 'N2O', 'O3', 'O2'].copy(),
            cialist=['H2-H', 'H2-H2', 'H2-He', 'He-H'].copy(),
            xmspecies=['TIO', 'CH4', 'H2O', 'H2CO', 'HCN', 'CO', 'CO2', 'NH3'].copy(),
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
                    select = ((matnu > (nubin-mydw/2.)) &
                              (matnu <= nubin+mydw/2.))
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
def atmos(fin, xsl, spc, out, mclen=int(1e4), verbose=False):
    '''
G. ROUDIER: Cerberus retrievial
    '''
    am = False
    orbp = fin['priors'].copy()
    ssc = syscore.ssconstants(mks=True)
    modfam = ['TEC', 'PHOTOCHEM', 'HESC']
    modparlbl = {'TEC':['XtoH', 'CtoO'],
                 'PHOTOCHEM':['TIO', 'CH4', 'C2H2', 'NH3'],
                 'HESC':['TIO', 'N2O', 'CO2']}
    for p in spc['data'].keys():
        out['data'][p] = {}
        out['data'][p]['MODELPARNAMES'] = modparlbl
        eqtemp = orbp['T*']*np.sqrt(orbp['R*']*ssc['Rsun/AU']/(2.*orbp[p]['sma']))
        tspc = np.array(spc['data'][p]['ES'])
        terr = np.array(spc['data'][p]['ESerr'])
        tspecerr = abs(tspc**2 - (tspc + terr)**2)
        tspectrum = tspc**2
        cleanup = np.isfinite(tspectrum) & (tspecerr < 1e0)
        solidr = orbp[p]['rp']*ssc['Rjup']  # MKS
        for model in modfam:
            ctxtupdt(cleanup=cleanup, model=model, p=p, solidr=solidr, orbp=orbp,
                     tspectrum=tspectrum, xsl=xsl, spc=spc, modparlbl=modparlbl)
            out['data'][p][model] = {}
            nodes = []
            with pm.Model():
                ctp = pm.Uniform('CTP', -6., 1.)
                hza = pm.Uniform('HScale', -6e0, 6e0)
                hzi = pm.Uniform('HIndex', -4e0, 0e0)
                tpr = pm.Uniform('T', eqtemp/2e0, 2e0*eqtemp)
                modelpar = pm.Uniform(model, lower=-6e0, upper=6e0,
                                      shape=len(modparlbl[model]))
                nodes.append(ctp)
                nodes.append(hza)
                nodes.append(hzi)
                nodes.append(tpr)
                nodes.append(modelpar)
                # CERBERUS MCMC
                _mcdata = pm.Normal('mcdata', mu=fmcerberus(*nodes),
                                    tau=1e0/(np.nanmedian(tspecerr[cleanup])**2),
                                    observed=tspectrum[cleanup])
                log.warning('>-- MCMC nodes: %s', str([n.name for n in nodes]))
                # ALL PRINTS ARE IN THE PM.SAMPLE CALL, CANNOT GET RID OF THEM
                trace = pm.sample(mclen, cores=4, tune=int(mclen/2),
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
            out['data'][p][model]['MCTRACE'] = mctrace
            pass
        out['data'][p]['WAVELENGTH'] = np.array(spc['data'][p]['WB'])
        out['data'][p]['SPECTRUM'] = np.array(spc['data'][p]['ES'])
        out['data'][p]['ERRORS'] = np.array(spc['data'][p]['ESerr'])
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
             hzlib=None, hzp=None, hzslope=-4., hztop=None,
             cheq=None, h2rs=True, logx=False, pnet='b',
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
        mixratio, fH2, fHe = crbce(p, temp, C2Or=cheq['CtoO'], X2Hr=cheq['XtoH'])
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
    tau, wtau = gettau(xsecs, qtgrid, temp, mixratio,
                       z, dz, rho, rp0, p, wgrid,
                       lbroadening, lshifting, cialist, fH2, fHe,
                       xmollist, rayleigh,
                       hzlib, hzp, hzslope, hztop,
                       h2rs=h2rs, debug=debug)
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
           cialist, fH2, fHe,
           xmollist, rayleigh, hzlib, hzp, hzslope, hztop,
           h2rs=True,
           debug=False):
    '''
G. ROUDIER: Builds optical depth matrix
    '''
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
                                 debug=debug)  # cm^2/mol
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
    if (hzlib is None) or (hzp is None) or (hztop is None):
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
    if debug:
        plt.figure()
        plt.imshow(np.log10(tau), aspect='auto')
        plt.title('Total Optical Depth / Pressure Layer')
        plt.colorbar()
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
def crbce(p, temp, C2Or=0., X2Hr=0.):
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
    AN = (10.**X2Hr)*solar['nN']/nH
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
    ctp, hza, hzi, tpr, mdp = crbinputs
    fmc = np.zeros(ctxt.tspectrum.size)
    if ctxt.model == 'TEC':
        tceqdict = {}
        tceqdict['XtoH'] = float(mdp[0])
        tceqdict['CtoO'] = float(mdp[1])
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
# ----------------------------------- --------------------------------
