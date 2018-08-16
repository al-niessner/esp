# -- IMPORTS -- ------------------------------------------------------
import os
import pymc as pm
import numpy as np
import lmfit as lm
import scipy.constants as cst
import matplotlib.pyplot as plt

from pymc.distributions import Uniform as Uniform
from pymc.distributions import Normal as Normal
from scipy.interpolate import interp1d as itp
# ------------- ------------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- X SECTIONS LIBRARY -- -------------------------------------------
def xsecs(spc, fin, out,
          hitemp='/proj/sdp/data/CERBERUS/HITEMP',
          tips='/proj/sdp/data/CERBERUS/TIPS',
          ciadir='/proj/sdp/data/CERBERUS/HITRAN/CIA',
          exomoldir='/proj/sdp/data/CERBERUS/EXOMOL',
          knownspecies=['NO', 'OH', 'C2H2', 'N2', 'N2O', 'O3', 'O2'],
          cialist=['H2-H', 'H2-H2', 'H2-He', 'He-H'],
          xmspecies=['TIO', 'CH4', 'H2O', 'H2CO', 'HCN', 'CO', 'CO2', 'NH3'],
          verbose=False, debug=False):
    cs = False
    for p in spc['data'].keys():
        out['data'][p] = {}
        wgrid = np.array(spc['data'][p]['WB'])
        qtgrid = gettpf(tips, knownspecies, debug=debug)
        library = {}
        nugrid = (1e4/np.copy(wgrid))[::-1]
        dwnu = np.concatenate((np.array([np.diff(nugrid)[0]]), np.diff(nugrid)))
        for myexomol in xmspecies:
            if verbose: print(myexomol)
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
                bint = []
                matnu = np.array(library[myexomol]['nutemp'])[select]
                sigma2 = np.array(library[myexomol]['Itemp'])[select]
                mattemp = np.array(library[myexomol]['Ttemp'])[select]
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
                if debug:
                    plt.plot(x, y, 'o')
                    xp = np.arange(101)/100.*(3000. - np.min(x))+np.min(x)
                    plt.plot(xp, myspl(xp))
                    plt.show()
                    pass
                pass
            if debug:
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
                plt.xlabel('Wavelength $\lambda$[$\mu m$]',
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
            if verbose: print(mycia)
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
                    bint = []
                    matnu = np.array(library[mycia]['nutemp'])[select]
                    sigma2 = np.array(library[mycia]['Itemp'])[select]
                    mattemp = np.array(library[mycia]['Ttemp'])[select]
                    for nubin, mydw in zip(nugrid, dwnu):
                        select = ((matnu > (nubin-mydw/2.)) &
                                  (matnu <= nubin+mydw/2.))
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
                    if debug:
                        plt.plot(x, y, 'o')
                        xp = np.arange(101)/100.*(np.max(x) - np.min(x))+np.min(x)
                        plt.plot(xp, myspl(xp))
                        plt.show()
                        pass
                    pass
                pass
            if debug:
                for temp in set(library[mycia]['T']):
                    select = np.array(library[mycia]['T']) == temp
                    plt.semilogy(1e4/(np.array(library[mycia]['nu'])[select]),
                                 np.array(library[mycia]['I'])[select])
                    pass
                plt.title(mycia)
                plt.xlabel('Wavelength $\lambda$[$\mu m$]')
                plt.ylabel('Line intensity $S(T)$ [$cm^{5}.molecule^{-2}$]')
                plt.show()
                pass
            pass
        for ks in knownspecies:
            if debug: print(ks)
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
                            if ks == 'H2O':
                                cotest = float(line[15:15+10]) > 1e-27
                                pass
                            if ks == 'CO2':
                                cotest = float(line[15:15+10]) > 1e-29
                                pass
                            if ((waveeq < (np.max(wgrid)+dwmax)) and (waveeq > (np.min(wgrid)-dwmin)) and cotest):
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
                else:
                    if debug: print(':'.join(('Skipping', fdata)))
                    pass
                pass
            if debug:
                for i in set(library[ks]['I']):
                    select = np.array(library[ks]['I']) == i
                    plt.semilogy(np.array(library[ks]['MU'])[select],
                                 np.array(library[ks]['S'])[select], '.')
                    pass
                plt.title(ks)
                plt.xlabel('Wavelength $\lambda$[$\mu m$]')
                plt.ylabel('Line intensity $S_{296K}$ [$cm.molecule^{-1}$]')
                plt.show()
                pass
            pass
        out['data'][p]['XSECS'] = library
        out['data'][p]['QTGRID'] = qtgrid
        pass
    if len(out['data'].keys()) > 0: out['STATUS'].append(True)
    return cs
# ------------------------ -------------------------------------------
# -- TOTAL PARTITION FUNCTION -- -------------------------------------
def gettpf(tips, knownspecies, debug=False):
    grid = {}
    # Gamache et al. 2011
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
            if debug:
                plt.plot(tempgrid, myspl(tempgrid))
                plt.plot(tempgrid, y, '+')
                pass
            pass
        if debug:
            plt.title(ks)
            plt.xlabel('Temperature T[K]')
            plt.ylabel('Total Internal Partition Function Q')
            plt.show()
        pass
    return grid
# ------------------------------ -------------------------------------
# -- HAZE DENSITY PROFILE LIBRARY -- ---------------------------------
def hzlib(mcmc, priors, sv,
          hazedir='/proj/sdp/data/CERBERUS/HAZE',
          datafile='Jup-ISS-aerosol.dat',
          verbose=False, debug=False):
    vdensity = {'PRESSURE':[], 'ISOBAR':[],
                'JMAX':[], 'SPLJMAX':[],
                'JMED':[], 'SPLJMED':[],
                'JAV':[], 'SPLJAV':[]}
    with open(os.path.join(hazedir, datafile), 'r') as fp:
        data = fp.readlines()
        fp.close()
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
    pressure = 1e-3*np.array(pressure)  # [bar]
    density = 1e-6*np.array(density)  # [n/m^3]
    jmax = np.nanmax(density, 1)
    jmed = np.median(density, 1)
    jav = np.mean(density, 1)
    isobar = pressure*0. + np.mean(jav)
    sortme = np.argsort(pressure)
    x = pressure[sortme]
    y = jmax[sortme]
    jmaxspl = itp(x, y, s=0, ext=1)
    y = jmed[sortme]
    jmedspl = itp(x, y, s=0, ext=1)
    y = jav[sortme]
    javspl = itp(x, y, s=0, ext=1)
    if verbose:
        plt.figure()
        plt.plot(1e6*jmax, pressure, label='Jupiter $d_{max}(\Phi)$')
        plt.plot(1e6*jmed, pressure, label='Jupiter $d_{median}(\Phi)$')
        plt.plot(1e6*jav, pressure, label='Jupiter $d_{mean}(\Phi)$')
        plt.plot(1e6*isobar, pressure, label='Jupiter isobar')
        plt.legend(loc=3, frameon=False)
        plt.plot(1e6*jmaxspl(pressure), pressure, 'k+')
        plt.plot(1e6*jmedspl(pressure), pressure, 'k+')
        plt.plot(1e6*javspl(pressure), pressure, 'k+')
        plt.xlabel('Aerosol Density [$n.{cm}^{-3}$]')
        plt.ylabel('Pressure [bar]')
        plt.semilogy()
        plt.semilogx()
        plt.gca().invert_yaxis()
        plt.xlim([1e-10, 1e4])
        plt.show()
        pass
    latvdp = []
    for dprofile in density.T:
        latvdp.append(itp(pressure, dprofile,
                           kind='cubic', bounds_error=False,
                           fill_value=0))
        pass
    if verbose:
        plt.figure()
        for vmr, itpvdp in zip(density.T, latvdp):
            plt.plot(1e6*vmr, pressure, '*')
            plt.plot(1e6*itpvdp(pressure), pressure)
            pass
        plt.xlabel('Aerosol Density [$n.{cm}^{-3}$]')
        plt.ylabel('Pressure [bar]')
        plt.semilogy()
        plt.semilogx()
        plt.gca().invert_yaxis()
        plt.xlim([1e-10, 1e4])
        plt.show()
        pass
    vdensity['DTABLE'] = np.array(latvdp)
    vdensity['PRESSURE'] = list(pressure)
    vdensity['ISOBAR'] = list(isobar)
    vdensity['JMAX'] = list(jmax)
    vdensity['SPLJMAX'].append(jmaxspl)
    vdensity['JMED'] = list(jmed)
    vdensity['SPLJMED'].append(jmedspl)
    vdensity['JAV'] = list(jav)
    vdensity['SPLJAV'].append(javspl)
    sv['PROFILE'].append(vdensity)
    return
# ---------------------------------- ---------------------------------
# -- ATMOS -- --------------------------------------------------------
def atmos(mcmc, priors, xsecs, qtgrid, hzlib, sv,
          lbroadening=False, lshifting=False,
          res='high',
          mixratio={'H2O':-10., 'CH4':-10., 'HCN':-10., 'H2CO':-10.,
                    'CO2':-10., 'CO':-10., 'NH3':-10., 'C2H2':-10.,
                    'N2':-10., 'N2O':-10., 'O3':-10., 'O2':-10.},
          cialist=['H2-H2', 'H2-He', 'H2-H', 'He-H'],
          xmollist=['TIO', 'CH4', 'H2O', 'H2CO', 'HCN',
                    'CO', 'CO2', 'NH3'],
          cloudtp=1., rayleigh=0.,
          mclen=int(1e4), nuisancespec=False,
          atmosmc=False, tceq=True,
          verbose=False, debug=False):
    out = {}
    orbp = priors['orbpar'][0]
    pnet = 'b'
    if orbp['sn'] == '55 Cnc': pnet = 'e'
    if orbp['sn'] == 'KOI-314': pnet = 'd'
    eqtemp = orbp['Ts']*np.sqrt(orbp['Rs']/(2.*orbp[pnet]['a']))
    wgrid = np.array(mcmc[res][0]['MU'])
    tspectrum = np.array(mcmc[res][0]['rprs'])
    tspecup = np.array(mcmc[res][0]['rprs_uerr'])
    tspelow = np.array(mcmc[res][0]['rprs_lerr'])
    tspecerr = np.sqrt(tspecup * tspelow)
    svtspec = tspecerr < 1e0
    zmeanspec = tspectrum - np.nanmean(tspectrum[svtspec])

    if atmosmc:
        mcparams = Parameters()
        mcparams.add('CtoO', value=0., vary=False)
        mcparams.add('XtoH', value=0., vary=False)
        mcparams.add('TopP', value=1., vary=False)
        mcparams.add('Haze', value=-10., vary=False)
        mcparams.add('Cloud', value=1., vary=False)
        mcparams.add('Rp0', value=orbp[pnet]['rp'], vary=False)
        mcparams.add('Tp', value=eqtemp, vary=True)
        jsonme = mcparams.dumps()
        tspectrum = lmcrbmodel(mcparams, wgrid, orbp=orbp,
                               xsecs=xsecs, qtgrid=qtgrid,
                               cialist=cialist, xmollist=xmollist,
                               hzlib=hzlib, hzp='SPLJAV',
                               ce=tceq, shape=False, verbose=debug)
        mcmodel = tspectrum.copy()
        tspectrum = (tspectrum + np.random.normal(scale=2e-4,
                                                  size=tspectrum.size))
        tspecerr = tspectrum*0. + 2e-4
        if debug:
            plt.figure()
            plt.errorbar(wgrid, 1e2*tspectrum, yerr=1e2*tspecerr, fmt='*')
            plt.plot(wgrid, 1e2*mcmodel, linewidth=3.)
            plt.xlabel('Wavelength $\lambda$[$\mu m$]')
            plt.ylabel('Transit Depth [%]')
            plt.show()
            pass
        pass
    else:
        # MODEL SELECTION
        tcepar = Parameters()
        tcepar.add('CtoO', value=0., vary=True)
        tcepar.add('XtoH', value=0., vary=True)
        tcepar.add('TopP', value=1., vary=True)
        tcepar.add('Haze', value=0., vary=True)
        tcepar.add('Tp', value=eqtemp, vary=False)
        tcepar.add('Rp0', value=orbp[pnet]['rp'], vary=False)
        lmbest = minimize(lmcrbmodel, tcepar,
                          args=(wgrid, zmeanspec, tspecerr,
                                orbp, xsecs, qtgrid,
                                cialist, xmollist, hzlib, None,
                                True, True, False, svtspec))
        tceqbest = lmcrbmodel(lmbest.params, wgrid, orbp=orbp,
                              xsecs=xsecs, qtgrid=qtgrid,
                              cialist=cialist, xmollist=xmollist,
                              hzlib=hzlib,
                              hzp=None, ce=True, shape=True,
                              verbose=False, cleanup=svtspec)
        mclean = np.isfinite(tceqbest)
        chisq = (tceqbest[mclean] - zmeanspec[svtspec][mclean])**2
        chisq /= (tspecerr[svtspec][mclean])**2
        likeh = np.exp(-chisq/2e0)
        likeh /= np.sqrt(2e0*np.pi*(tspecerr[svtspec][mclean])**2)
        logtclike = np.sum(np.log(likeh))
        logtclike -= (4e0)*np.log(np.sum(mclean))/2e0
        tclike = logtclike
        ccparams = Parameters()
        ccparams.add('CH4', value=0., vary=True, min=-4, max=4)
        ccparams.add('C2H2', value=0., vary=True, min=-4, max=4)
        ccparams.add('NH3', value=0., vary=True, min=-4, max=4)
        ccparams.add('N2O', value=0., vary=True, min=-4, max=4)
        ccparams.add('O2', value=0., vary=True, min=-4, max=4)
        ccparams.add('TopP', value=1., vary=True)
        ccparams.add('Haze', value=0., vary=True)
        ccparams.add('H2CO', value=-10., vary=False)
        ccparams.add('CO', value=-10., vary=False)
        ccparams.add('H2O', value=-10., vary=False)
        ccparams.add('HCN', value=-10., vary=False)
        ccparams.add('OH', value=-10., vary=False)
        ccparams.add('N2', value=-10., vary=False)
        ccparams.add('O3', value=-10., vary=False)
        ccparams.add('CO2', value=-10., vary=False)
        ccparams.add('Rp0', value=orbp[pnet]['rp'], vary=False)
        ccparams.add('Tp', value=eqtemp, vary=False)
        ccbest = minimize(lmcrbmodel, ccparams,
                          args=(wgrid, zmeanspec, tspecerr,
                                orbp, xsecs, qtgrid,
                                cialist, xmollist, hzlib, None,
                                False, True, False, svtspec))
        ccnebest = lmcrbmodel(ccbest.params, wgrid,
                              orbp=orbp, xsecs=xsecs,
                              qtgrid=qtgrid, cialist=cialist,
                              xmollist=xmollist, hzlib=hzlib,
                              hzp=None, ce=False, shape=True,
                              verbose=False, cleanup=svtspec)
        mclean = np.isfinite(ccnebest)
        chisq = (ccnebest[mclean] - zmeanspec[svtspec][mclean])**2
        chisq /= (tspecerr[svtspec][mclean])**2
        likeh = np.exp(-chisq/2e0)
        likeh /= np.sqrt(2e0*np.pi*(tspecerr[svtspec][mclean])**2)
        logcclike = np.sum(np.log(likeh))
        logcclike -= (5e0)*np.log(np.sum(mclean))/2e0
        cclike = logcclike
        mwparams = Parameters()
        mwparams.add('H2O', value=0., vary=True, min=-4, max=4)
        mwparams.add('NH3', value=0., vary=True, min=-4, max=4)
        mwparams.add('H2CO', value=0., vary=True, min=-4, max=4)
        mwparams.add('CO', value=0., vary=True, min=-4, max=4)
        mwparams.add('CO2', value=0., vary=True, min=-4, max=4)
        mwparams.add('Haze', value=0., vary=True)
        mwparams.add('TopP', value=1., vary=True)
        mwparams.add('N2', value=-10., vary=False)
        mwparams.add('HCN', value=-10., vary=False)
        mwparams.add('C2H2', value=-10., vary=False)
        mwparams.add('OH', value=-10., vary=False)
        mwparams.add('CH4', value=-10., vary=False)
        mwparams.add('N2O', value=-10., vary=False)
        mwparams.add('O3', value=-10., vary=False)
        mwparams.add('O2', value=-10., vary=False)
        mwparams.add('Rp0', value=orbp[pnet]['rp'], vary=False)
        mwparams.add('Tp', value=eqtemp, vary=False)
        mwbest = minimize(lmcrbmodel, mwparams,
                          args=(wgrid, zmeanspec, tspecerr,
                                orbp, xsecs, qtgrid,
                                cialist, xmollist, hzlib, None,
                                False, True, False, svtspec))
        mwnebest = lmcrbmodel(mwbest.params, wgrid, orbp=orbp,
                              xsecs=xsecs, qtgrid=qtgrid,
                              cialist=cialist, xmollist=xmollist,
                              hzlib=hzlib, hzp=None,
                              ce=False, shape=True,
                              verbose=False, cleanup=svtspec)
        mclean = np.isfinite(mwnebest)
        chisq = (mwnebest[mclean] - zmeanspec[svtspec][mclean])**2
        chisq /= (tspecerr[svtspec][mclean])**2
        likeh = np.exp(-chisq/2e0)
        likeh /= np.sqrt(2e0*np.pi*(tspecerr[svtspec][mclean])**2)
        logmwlike = np.sum(np.log(likeh))
        logmwlike -= (5e0)*np.log(np.sum(mclean))/2e0
        mwlike = logmwlike
        tclike = int(np.round(tclike))
        cclike = int(np.round(cclike))
        mwlike = int(np.round(mwlike))
        if verbose: print('Scores:', tclike, cclike, mwlike)
        mw = True
        if abs(tclike - mwlike) < 2: mw = False
        if verbose:
            fts = 24
            plt.figure(figsize=(16,12))
            plt.errorbar(wgrid[svtspec], 1e2*tspectrum[svtspec],
                         yerr=[1e2*tspelow[svtspec], 1e2*tspecup[svtspec]])
            plt.plot(wgrid[svtspec],
                     1e2*(tceqbest + np.nanmean(tspectrum[svtspec])),
                     linewidth=5,
                     label='TCE '+str(tclike))
            plt.plot(wgrid[svtspec],
                     1e2*(ccnebest + np.nanmean(tspectrum[svtspec])),
                     linewidth=5,
                     label='CH4 NE '+str(cclike))
            if mw:
                plt.plot(wgrid[svtspec],
                         1e2*(mwnebest + np.nanmean(tspectrum[svtspec])),
                         linewidth=5,
                         label='H2O NE '+str(mwlike))
                pass
            plt.xlabel('Wavelength [$\mu m$]', fontsize=fts)
            plt.ylabel('$(R_p/R_*)^2$ [%]', fontsize=fts)
            plt.tick_params(axis='both', labelsize=fts-4)
            plt.title(orbp['sn'], fontsize=fts)
            plt.legend()
            plt.show()
            pass

        if mw: scores = [tclike, cclike, mwlike]
        else: scores = [tclike, cclike]
        preference = [False]*4
        if tclike == np.max(scores):
            mcparams = Parameters()
            mcparams.add('CtoO', value=0., vary=False)
            mcparams.add('XtoH', value=0., vary=False)
            mcparams.add('TopP', value=1., vary=False)
            mcparams.add('Haze', value=-6., vary=False)
            mcparams.add('Cloud', value=1., vary=False)
            mcparams.add('Rp0', value=orbp[pnet]['rp'], vary=False)
            mcparams.add('Tp', value=eqtemp, vary=False)
            jsonme = mcparams.dumps()
            preference[0] = True
            pass
        if cclike == np.max(scores):
            mcparams = Parameters()
            mcparams.add('CH4', value=0., vary=False)
            mcparams.add('C2H2', value=0., vary=False)
            mcparams.add('N2O', value=0., vary=False)
            mcparams.add('O2', value=0., vary=False)
            mcparams.add('NH3', value=0., vary=False)
            mcparams.add('TopP', value=1., vary=False)
            mcparams.add('Haze', value=-6., vary=False)
            mcparams.add('Rp0', value=orbp[pnet]['rp'], vary=False)
            mcparams.add('Tp', value=eqtemp, vary=False)
            preference[1] = True
            jsonme = mcparams.dumps()
            pass
        if mw:
            if mwlike == np.max(scores):
                mcparams = Parameters()
                mcparams.add('CO', value=0., vary=False)
                mcparams.add('H2O', value=0., vary=False)
                mcparams.add('CO2', value=0., vary=False)
                mcparams.add('H2CO', value=0., vary=False)
                mcparams.add('NH3', value=0., vary=False)
                mcparams.add('TopP', value=1., vary=False)
                mcparams.add('Haze', value=-6., vary=False)
                mcparams.add('Rp0', value=orbp[pnet]['rp'], vary=False)
                mcparams.add('Tp', value=eqtemp, vary=False)
                preference[2] = True
                jsonme = mcparams.dumps()
                pass
            pass
        if np.sum(abs(scores - np.max(scores)) < 2) > 1:
            mcparams = Parameters()
            mcparams.add('C2H2', value=0., vary=False)
            mcparams.add('CH4', value=0., vary=False)
            mcparams.add('H2O', value=0., vary=False)
            mcparams.add('H2CO', value=0., vary=False)
            mcparams.add('NH3', value=0., vary=False)
            mcparams.add('N2O', value=0., vary=False)
            mcparams.add('O2', value=0., vary=False)
            mcparams.add('CO', value=0., vary=False)
            mcparams.add('CO2', value=0., vary=False)
            mcparams.add('TopP', value=1., vary=False)
            mcparams.add('Haze', value=-6., vary=False)
            mcparams.add('Rp0', value=orbp[pnet]['rp'], vary=False)
            mcparams.add('Tp', value=eqtemp, vary=False)
            preference[3] = True
            preference[2] = False
            preference[1] = False
            preference[0] = False
            jsonme = mcparams.dumps()
            pass
        pass
    # SETUP
    solidr = orbp[pnet]['rp']
    # PRIORS
    rayleigh = Uniform('Haze', -10., 6.)
    atmtp = Uniform('Tp', 0.5*eqtemp, 2.*eqtemp)
    cloud = Uniform('TopP', -5., 1.)
    found = False
    if preference[0]:
        mtlct = Uniform('XtoH', -6., 3.)
        nrch = Uniform('CtoO', -6., 6.)
        h2o = None
        ch4 = None
        c2h2 = None
        h2co = None
        co = None
        co2 = None
        nh3 = None
        o2 = None
        n2o = None
        found = True
        preference[1] = False
        preference[2] = False
        preference[3] = False
        if verbose: print(scores, ' --< TCE >--')
        pass
    if preference[1] and not(found):
        mtlct = None
        nrch = None
        ch4 = Uniform('CH4', -4., 4.)
        c2h2 = Uniform('C2H2', -4., 4.)
        nh3 = Uniform('NH3', -4., 4.)
        n2o = Uniform('N2O', -4., 4.)
        o2 = Uniform('O2', -4., 4.)
        h2o = None
        co = None
        co2 = None
        h2co = None
        found = True
        preference[0] = False
        preference[2] = False
        preference[3] = False
        if verbose: print(scores, ' --< CH4+ >--')
        pass
    if preference[2] and not(found):
        mtlct = None
        nrch = None
        h2o = Uniform('H2O', -4., 4.)
        nh3 = Uniform('NH3', -4., 4.)
        h2co = Uniform('H2CO', -4., 4.)
        co = Uniform('CO', -4., 4.)
        co2 = Uniform('CO2', -4., 4.)
        ch4 = None
        c2h2 = None
        n2o = None
        o2 = None
        found = True
        preference[0] = False
        preference[1] = False
        preference[3] = False
        if verbose: print(scores, ' --< H2O+ >--')
        pass
    if preference[3] and not(found):
        mtlct = None
        nrch = None
        h2o = Uniform('H2O', -4., 4.)
        ch4 = Uniform('CH4', -4., 4.)
        nh3 = Uniform('NH3', -4., 4.)
        c2h2 = Uniform('C2H2', -4., 4.)
        h2co = Uniform('H2CO', -4., 4.)
        co = Uniform('CO', -4., 4.)
        co2 = Uniform('CO2', -4., 4.)
        n2o = Uniform('N2O', -4., 4.)
        o2 = Uniform('O2', -4., 4.)
        found = True
        preference[0] = False
        preference[1] = False
        preference[2] = False
        if verbose: print(scores, ' --< NE >--')
        pass
    cleanup = np.isfinite(tspectrum) & svtspec
    # CERBERUS FM CALL
    @pm.deterministic
    def ccerberus(mtlct=mtlct, nrch=nrch,
                  ch4=ch4, c2h2=c2h2, h2co=h2co, co=co,
                  h2o=h2o, nh3=nh3, o2=o2, n2o=n2o, co2=co2,
                  atmtp=atmtp,
                  rayleigh=rayleigh, cloud=cloud):
        if preference[0]:
            tceqdict = {}
            tceqdict['CtoO'] = float(nrch)
            tceqdict['XtoH'] = float(mtlct)
            cmodel = crbmodel(None, float(rayleigh),
                              float(cloud), solidr,
                              orbp, xsecs, qtgrid,
                              float(atmtp), wgrid,
                              False, False, cialist, xmollist,
                              hzlib=hzlib, hzp=None,
                              cheq=tceqdict,
                              verbose=False, debug=False)
            pass
        if preference[1]:
            mixratio={'CH4':float(ch4), 'O2':float(o2),
                      'NH3':float(nh3),
                      'N2O':float(n2o), 'C2H2':float(c2h2)}
            cmodel = crbmodel(mixratio, float(rayleigh),
                              float(cloud), solidr,
                              orbp, xsecs, qtgrid,
                              float(atmtp), wgrid,
                              False, False, cialist, xmollist,
                              hzlib=hzlib, hzp=None,
                              cheq=None,
                              verbose=False, debug=False)
            pass
        if preference[2]:
            mixratio={'H2O':float(h2o), 'CO':float(co),
                      'H2CO':float(h2co),
                      'NH3':float(nh3), 'CO2':float(co2)}
            cmodel = crbmodel(mixratio, float(rayleigh),
                              float(cloud), solidr,
                              orbp, xsecs, qtgrid,
                              float(atmtp), wgrid,
                              False, False, cialist, xmollist,
                              hzlib=hzlib, hzp=None,
                              cheq=None,
                              verbose=False, debug=False)
        if preference[3]:
            mixratio={'H2O':float(h2o), 'CO2':float(co2),
                      'H2CO':float(h2co), 'CH4':float(ch4),
                      'CO':float(co), 'N2O':float(n2o), 'O2':float(o2),
                      'NH3':float(nh3), 'C2H2':float(c2h2)}
            cmodel = crbmodel(mixratio, float(rayleigh),
                              float(cloud), solidr,
                              orbp, xsecs, qtgrid,
                              float(atmtp), wgrid,
                              False, False, cialist, xmollist,
                              hzlib=hzlib, hzp=None,
                              cheq=None,
                              verbose=False, debug=False)
            pass
        cmodel = cmodel[cleanup]
        cmodel = cmodel - np.nanmean(cmodel) + np.nanmean(tspectrum[cleanup])
        return cmodel
    # CERBERUS MCMC
    sigma = np.sqrt(np.nanmean(tspecerr[cleanup]**2))
    mcdata = Normal('mcdata', mu=ccerberus, tau=1./(sigma**2),
                    value=tspectrum[cleanup], observed=True)
    nodes = [mcdata, atmtp, rayleigh, cloud]
    if preference[0]:
        nodes.append(mtlct)
        nodes.append(nrch)
        pass
    if preference[1]:
        nodes.append(ch4)
        nodes.append(c2h2)
        nodes.append(n2o)
        nodes.append(o2)
        nodes.append(nh3)
        pass
    if preference[2]:
        nodes.append(h2o)
        nodes.append(co)
        nodes.append(co2)
        nodes.append(nh3)
        nodes.append(h2co)
        pass
    if preference[3]:
        nodes.append(ch4)
        nodes.append(n2o)
        nodes.append(o2)
        nodes.append(h2o)
        nodes.append(c2h2)
        nodes.append(nh3)
        nodes.append(h2co)
        nodes.append(co)
        nodes.append(co2)
        pass
    mcmcmodel = pm.Model(nodes)
    markovc = pm.MCMC(mcmcmodel)
    burnin = int(np.sqrt(mclen)*4*len(nodes))
    if burnin > mclen: burnin = 0
    markovc.sample(mclen, burn=burnin, progress_bar=verbose)
    if verbose: print('')

    mcpost = markovc.stats()
    bestparams = Parameters()
    bestparams.loads(jsonme)
    bestvalues = {}
    chains = {}
    for thiskey in mcpost:
        thischain = markovc.trace(thiskey)[:]
        chains[thiskey] = thischain
        chainout = np.median(thischain)
        bestparams[thiskey].value = chainout['mostlikely']
        bestvalues[thiskey] = chainout
        nbbn = int(np.sqrt(len(thischain)))
        if verbose:
            plt.figure(figsize=(6, 9))
            plt.subplot(2, 1, 1)
            plt.plot(thischain)
            plt.title(thiskey)
            plt.subplot(2, 1, 2)
            plt.hist(thischain, bins=nbbn, histtype='stepfilled')
            plt.xlabel(thiskey)
            plt.title(str(chainout['mostlikely']))
            pass
        pass
    if verbose: plt.show()
    # 10 bars Solid Radius
    bestparams['Rp0'].vary = True
    if verbose:
        print('Adjusting 10 bars radius')
        print(bestparams['Rp0'].value)
        pass
    if preference[0]:
        lmbest = minimize(lmcrbmodel, bestparams,
                          args=(wgrid, tspectrum,
                                tspecerr, orbp, xsecs,
                                qtgrid, cialist, xmollist,
                                hzlib, None,
                                tceq, False, False, cleanup))
        crbbest = lmcrbmodel(lmbest.params, wgrid, orbp=orbp,
                             xsecs=xsecs, qtgrid=qtgrid,
                             cialist=cialist, xmollist=xmollist,
                             hzlib=hzlib, hzp=None,
                             ce=tceq, shape=False, verbose=verbose,
                             cleanup=cleanup)
        pass
    else:
        lmbest = minimize(lmcrbmodel, bestparams,
                          args=(wgrid, tspectrum,
                                tspecerr, orbp, xsecs,
                                qtgrid, cialist, xmollist,
                                hzlib, None,
                                False, False, False, cleanup))
        crbbest = lmcrbmodel(lmbest.params, wgrid, orbp=orbp,
                             xsecs=xsecs, qtgrid=qtgrid,
                             cialist=cialist, xmollist=xmollist,
                             hzlib=hzlib, hzp=None,
                             ce=False, shape=False, verbose=verbose,
                             cleanup=cleanup)
        pass
    if verbose: print(lmbest.params['Rp0'].value)
    if verbose:
        plt.figure()
        plt.errorbar(wgrid[cleanup],
                     1e2*tspectrum[cleanup],
                     yerr=1e2*tspecerr[cleanup], fmt='*')
        if atmosmc: plt.plot(wgrid, 1e2*mcmodel, '--', linewidth=3.)
        plt.plot(wgrid[cleanup], 1e2*crbbest[cleanup], linewidth=3.)
        plt.xlabel('Wavelength $\lambda$[$\mu m$]')
        plt.ylabel('Transit Depth [%]')
        plt.show()
        pass
    crbdata = {}
    crbdata['WAVE'] = wgrid
    crbdata['SPECTRUM'] = tspectrum
    crbdata['ERRORS'] = tspecerr
    crbdata['MODELS'] = scores
    crbdata['SELECTION'] = preference
    crbdata['MODNAMES'] = ['TCE', 'CH4 NE', 'H2O NE', 'FREE']
    sv['DATA'].append(crbdata)  # SPECTRUM + ERRORS
    sv['MODEL'].append(crbbest)  # BEST MODEL
    sv['LMPARAMS'].append(lmbest.params)  # ASSOCIATED PARAMS
    sv['MCPOST'].append(bestvalues)  # CHAINS SUMMARY
    sv['MCCHAINS'].append(chains)  # CHAINS
    return
# ----------- --------------------------------------------------------
# -- CERBERUS MODEL -- -----------------------------------------------
def crbmodel(mixratio, rayleigh, cloudtp, rp0, orbp, xsecs, qtgrid,
             temp, wgrid, lbroadening, lshifting, cialist, xmollist,
             nlevels=100, Hsmax=15., solrad=10.,
             hzlib=None, hzp=None, hzslope=-4., hztop=None,
             cheq=None, h2rs=True, logx=False, ihmghaze=False,
             verbose=False, debug=False):
    # Probing up to 'Hsmax' scale heights from solid radius 'solrad'
    # evenly log divided amongst 'nlevels' steps
    tbmodel = time.time()
    if verbose: print('- Building Cerberus model ...')
    pnet = 'b'
    if orbp['sn'] == '55 Cnc': pnet = 'e'
    if orbp['sn'] == 'KOI-314': pnet = 'd'
    pgrid = np.arange(np.log(solrad)-Hsmax,
                      np.log(solrad)+Hsmax/nlevels,
                      Hsmax/(nlevels-1))
    pgrid = np.exp(pgrid)
    dp = np.diff(pgrid[::-1])
    p = pgrid[::-1]
    z = [0.]
    dz = []
    addz = []
    if cheq is not None:
        mixratio, fH2, fHe = crbce(p, temp,
                                   C2Or=cheq['CtoO'], X2Hr=cheq['XtoH'])
        mmw, fH2, fHe = getmmw(mixratio,
                               protosolar=False, fH2=fH2, fHe=fHe)
        pass
    else: mmw, fH2, fHe = getmmw(mixratio)
    mmw = mmw*cst.m_p  # [kg]
    Hs = cst.Boltzmann*temp/(mmw*1e-2*(10.**orbp[pnet]['logg']))  # [m]
    for press, dpress in zip(p[:-1], dp):
        rdz = abs(Hs/2.*np.log(1. + dpress/press))
        if len(addz) > 0: dz.append(addz[-1]/2. + rdz)
        else: dz.append(2.*rdz)
        addz.append(2.*rdz)
        z.append(z[-1]+addz[-1])
        pass
    dz.append(addz[-1])
    rho = p*1e5/(cst.Boltzmann*temp)
    # https://www.cfa.harvard.edu/~dfabricant/huchra/ay145/constants.html
    ua2m = 1.4960e11
    if ihmghaze:
        tottau = []
        for thisprofile in hzp:
            tau, wtau = gettau(orbp, xsecs, qtgrid, temp, mixratio,
                               z, dz, rho, rp0*ua2m, p, dp, wgrid,
                               lbroadening, lshifting, cialist, fH2, fHe,
                               xmollist, rayleigh,
                               hzlib, thisprofile, hzslope, hztop,
                               mmw, h2rs=h2rs, verbose=verbose, debug=debug)
            tottau.append(tau)
            pass
        tottau = np.array(tottau).T
        totauu = np.mean(tottau, 0)
        pass
    else:
        tau, wtau = gettau(orbp, xsecs, qtgrid, temp, mixratio,
                           z, dz, rho, rp0*ua2m, p, dp, wgrid,
                           lbroadening, lshifting, cialist, fH2, fHe,
                           xmollist, rayleigh,
                           hzlib, hzp, hzslope, hztop,
                           mmw, h2rs=h2rs, verbose=verbose, debug=debug)
        pass
    reversep = np.array(p[::-1])
    selectcloud = p > 10.**cloudtp
    blocked = False
    if all(selectcloud):
        tau = tau*0
        blocked = True
        pass
    if not(all(~selectcloud)) and not(blocked):
        cloudindex = np.max(np.arange(len(p))[selectcloud])+1
        for index in np.arange(wtau.size):
            myspl = itp(reversep, tau[:,index])
            tau[cloudindex,index] = myspl(10.**cloudtp)
            tau[:cloudindex,index] = 0.
            pass
        ctpdpress = 10.**cloudtp - p[cloudindex]
        ctpdz = abs(Hs/2.*np.log(1. + ctpdpress/p[cloudindex]))
        rp0 += (z[cloudindex]+ctpdz)/ua2m
        pass
    atmdepth = 2.*np.array(np.mat((rp0*ua2m+np.array(z))*np.array(dz))*
                           np.mat(1. - np.exp(-tau))).flatten()
    model = ((rp0*ua2m)**2 + atmdepth)/(orbp['Rs']*ua2m)**2
    temodel = time.time()
    if verbose: print('- Done in', temodel - tbmodel, 's')
    noatm = rp0**2/orbp['Rs']**2
    noatm = np.min(model)
    rp0hs = np.sqrt(noatm*orbp['Rs']**2)
    if verbose:
        fig, ax = plt.subplots(figsize=(10,6))
        axes = [ax, ax.twinx(), ax.twinx()]
        fig.subplots_adjust(left=0.125, right=0.775)
        axes[-1].spines['right'].set_position(('axes', 1.2))
        axes[-1].set_frame_on(True)
        axes[-1].patch.set_visible(False)
        axes[0].plot(wtau, 1e2*model)
        axes[0].plot(wtau, model*0+1e2*noatm, '--')
        axes[0].set_xlabel('Wavelength $\lambda$[$\mu m$]')
        axes[0].set_ylabel('Transit Depth [%]')
        axes[0].get_yaxis().get_major_formatter().set_useOffset(False)
        yaxmin, yaxmax = axes[0].get_ylim()
        ax2min = (np.sqrt(1e-2*yaxmin)*orbp['Rs'] - rp0hs)*ua2m/Hs
        ax2max = (np.sqrt(1e-2*yaxmax)*orbp['Rs'] - rp0hs)*ua2m/Hs
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
    # Proton mass dominated
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
def gettau(orbp, xsecs, qtgrid, temp, mixratio,
           z, dz, rho, rp0, p, dp, wgrid, lbroadening, lshifting,
           cialist, fH2, fHe,
           xmollist, rayleigh, hzlib, hzp, hzslope, hztop, mmw,
           h2rs=True,
           verbose=False, debug=False):
    firstelem = True
    vectauelem = []
    for myz in list(z):
        tauelem = 0.
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
                                 p, dp, mmr, lbroadening, lshifting, wgrid,
                                 verbose=verbose, debug=debug)  # cm^2/mol
            sigma = np.array(sigma)
            if np.sum(sigma < 0) > 0:
                sigma[sigma < 0] = 0
                pass
            if np.sum(~np.isfinite(sigma)) > 0:
                sigma[~np.isfinite(sigma)] = 0
                pass
            sigma = np.array(sigma)*1e-4  # m^2/mol
            if sigma.shape[0] < 2: sigma = sigma*np.array([np.ones(len(z))]).T
            pass
        else:
            # Hill et al. 2013
            sigma, lsig = getxmolxs(temp, xsecs[elem], wgrid)  # cm^2/mol
            sigma = np.array(sigma)
            if np.sum(sigma < 0) > 0:
                sigma[sigma < 0] = 0
                pass
            if np.sum(~np.isfinite(sigma)) > 0:
                sigma[~np.isfinite(sigma)] = 0
                pass
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
        sigma, lsig = getciaxs(temp, xsecs[cia], wgrid)  # cm^5/mol^2
        sigma = np.array(sigma)*1e-10  # m^5/mol^2
        if np.sum(sigma < 0) > 0:
            sigma[sigma < 0] = 0
            pass
        if np.sum(~np.isfinite(sigma)) > 0:
            sigma[~np.isfinite(sigma)] = 0
            pass
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
        # Lecavelier et al. 2008
        sray = sray0*(wgrid[::-1]/slambda0)**(-4)
        tauz = vectauelem*sray
        if firstelem:
            tau = np.array(tauz)*fH2
            firstelem = False
            pass
        else: tau = tau+np.array(tauz)*fH2
        pass
    if hzlib is not None:
        # West et al. 2004
        tauaero = (0.0083*(wgrid[::-1])**(hzslope)*
                   (1+
                    0.014*(wgrid[::-1])**(-2)+
                    0.00027*(wgrid[::-1])**(-4)))
        vectauhaze = []
        if hzp is not None:
            frh = hzlib['PROFILE'][0][hzp][0]
            rh = frh(p)
            rh[rh < 0] = 0.
            pass
        else: rh = np.array(hzlib['PROFILE'][0]['ISOBAR'])
        if hztop is not None:
            refhzp = float(p[rh == np.max(rh)])  # hztop = -1.632
            hzshift = hztop - np.log10(refhzp)
            splp = np.log10(p[::-1]) + hzshift
            splrh = rh[::-1]
            thisfrh = itp(splp, splrh)
            rh = thisfrh(np.log10(p))
            rh[rh < 0] = 0.
            pass
        if debug:
            jdata = np.array(hzlib['PROFILE'][0]['JAV'])
            jpres = np.array(hzlib['PROFILE'][0]['PRESSURE'])
            plt.figure()
            plt.plot(1e6*jdata, jpres)
            plt.plot(1e6*rh, p, '*')
            plt.plot(1e6*rh, rh*0.+10.**(-1.632), '--')
            plt.semilogy()
            plt.semilogx()
            plt.gca().invert_yaxis()
            plt.xlim([1e-10, 1e4])
            plt.show()
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
        tau = tau + np.array(tauz)*(10.**rayleigh)
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
def absorb(xsecs, qtgrid, T, p, dp, mmr, lbroadening, lshifting, wgrid,
           iso=0, Tref=296.,
           verbose=True, debug=False):
    select = (np.array(xsecs['I']) == iso+1)
    S = np.array(xsecs['S'])[select]
    E = np.array(xsecs['Epp'])[select]
    gself = np.array(xsecs['g_self'])[select]
    nu = np.array(xsecs['nu'])[select]
    delta = np.array(xsecs['delta'])[select]
    eta = np.array(xsecs['eta'])[select]
    MU = np.array(xsecs['MU'])[select]
    gair = np.array(xsecs['g_air'])[select]
    Qref = float(qtgrid['SPL'][iso](Tref))
    Q = float(qtgrid['SPL'][iso](T))
    c2 = 1e2*cst.h*cst.c/cst.Boltzmann
    tips = ((Qref*np.exp(-c2*E/T)*(1.-np.exp(-c2*nu/T)))/
            (Q*np.exp(-c2*E/Tref)*(1.-np.exp(-c2*nu/Tref))))
    sigma = S*tips
    ps = mmr*p
    gamma = np.array((np.mat(p-ps).T*np.mat(gair*(Tref/T)**eta) +
                      (np.mat(ps)).T*np.mat(gself)))
    if lbroadening:
        if lshifting:
            matnu = np.array((np.mat(np.ones(p.size)).T*np.mat(nu) +
                              np.mat(p).T*np.mat(delta)))
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
        plt.xlabel('Wavelength $\lambda$[$\mu m$]')
        plt.ylabel('Absorption Coeff [$cm^{2}.molecule^{-1}$]')
        plt.show()
        pass
    return absgrid, nugrid
# ----------------------------- --------------------------------------
# -- PRESSURE BROADENING -- ------------------------------------------
def flor(wave, nu, gamma, norm):
    f = norm*gamma/(np.pi*(gamma**2.+(wave - nu)**2.))
    return f
def intflor(wave, dwave, nu, gamma):
    f = 1./np.pi*(np.arctan((wave+dwave - nu)/gamma) -
                  np.arctan((wave-dwave - nu)/gamma))
    return f
# ------------------------- ------------------------------------------
# -- CIA -- ----------------------------------------------------------
def getciaxs(temp, xsecs, wgrid):
    sigma = np.array([thisspl(temp) for thisspl in xsecs['SPL']])
    nu = np.array(xsecs['SPLNU'])
    select = np.argsort(nu)
    nu = nu[select]
    sigma = sigma[select]
    return sigma, nu
# --------- ----------------------------------------------------------
# -- EXOMOL -- -------------------------------------------------------
def getxmolxs(temp, xsecs, wgrid):
    sigma = np.array([thisspl(temp) for thisspl in xsecs['SPL']])
    nu = np.array(xsecs['SPLNU'])
    select = np.argsort(nu)
    nu = nu[select]
    sigma = sigma[select]
    return sigma, nu
# ------------ -------------------------------------------------------
# -- CRB MODEL FOR LMFIT -- ------------------------------------------
def lmcrbmodel(params, wgrid,
               data=None, weights=None,
               orbp=None, xsecs=None, qtgrid=None,
               cialist=None, xmollist=None, hzlib=None, hzp=None,
               ce=False, shape=False, verbose=False, cleanup=None):
    knownspecies = ['NO', 'OH', 'C2H2', 'N2', 'N2O', 'O3', 'O2']
    if ce:
        mixratio = None
        cheq = {}
        cheq['CtoO'] = params['CtoO'].value
        cheq['XtoH'] = params['XtoH'].value
        pass
    else:
        cheq = None
        mixratio = {}
        for gazspec in params.keys():
            if (gazspec in xmollist) or (gazspec in knownspecies):
                mixratio[gazspec] = params[gazspec].value
                pass
            pass
        pass
    model = crbmodel(mixratio, params['Haze'].value,
                     params['TopP'].value,
                     params['Rp0'].value, orbp, xsecs, qtgrid,
                     params['Tp'].value, wgrid,
                     False, False, cialist, xmollist,
                     hzlib=hzlib, hzp=hzp, cheq=cheq,
                     verbose=verbose, debug=False)
    if shape: model = model - np.mean(model)
    if data is None:
        if cleanup is None: out = model
        else: out = model[cleanup]
        pass
    else:
        if cleanup is None:
            if weights is None: out = data - model
            else: out = (data - model)/weights
            pass
        else:
            if weights is None: out = data[cleanup] - model[cleanup]
            else: out = (data[cleanup] - model[cleanup])/weights[cleanup]
            pass
        pass
    return out
# ------------------------- ------------------------------------------
# -- CHEMICAL EQUILIBRIUM -- -----------------------------------------
# BURROWS AND SHARP 1998
def crbce(p, temp, C2Or=0., X2Hr=0.):
    # SOLAR ABUNDANCES (ANDERS & GREVESSE 1989)
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
    K1 = np.exp((a1/temp + b1 + c1*temp + d1*temp**2 + e1*temp**3)/
                (RcalpmolpK*temp))
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
    K2 = np.exp((a2/temp + b2 + c2*temp + d2*temp**2 + e2*temp**3)/
                (RcalpmolpK*temp))
    AN = (10.**X2Hr)*solar['nN']/nH
    AH2 = (pH2**2.)/(8.*K2)
    BN2 = AN + AH2 - np.sqrt((AN + AH2)**2. - (AN)**2.)
    BNH3 = 2.*(AN - BN2)
    nN2 = np.mean(BN2*pH2/p)
    if nN2 <= 0: nN2 = 1e-16
    nNH3 = np.mean(BNH3*pH2/p)
    if nNH3 <= 0: nNH3 = 1e-16
    mixratio = {'H2O':np.log10(nH2O)+6.,
                'CH4':np.log10(nCH4)+6.,
                'NH3':np.log10(nNH3)+6.,
                'N2':np.log10(nN2)+6.,
                'CO':np.log10(nCO)+6.}
    return mixratio, nH2, nHe
# -------------------------- -----------------------------------------
# -------------------------- -----------------------------------------
def getwgrid(fnsdir, res, filename='wavegrid.txt',
             verbose=False, debug=False):
    with open(os.path.join(fnsdir, filename), 'r') as fp:
        fnsgrid = fp.readlines()
        fp.close()
        pass
    wgrid = []
    starsnr = []
    for line in fnsgrid:
        line = np.array(line.split(' '))
        line = line[line != '']
        wgrid.append(line[0].astype(float))
        starsnr.append(line[1].astype(float))
        pass
    wgrid = np.array(wgrid)
    starsnr = np.array(starsnr)
    dwgrid = np.concatenate((np.diff(wgrid),
                             np.array([np.diff(wgrid)[-1]])))
    fnsres = wgrid/dwgrid
    thisres = float(res.replace('R', ''))
    scaleres = fnsres*(thisres/np.min(fnsres))
    refpoint = float(wgrid[fnsres == np.min(fnsres)])
    splscale = itp(wgrid, scaleres)
    splsnr = itp(wgrid, starsnr)
    spldw = itp(wgrid, dwgrid)
    newgrid = [np.min(wgrid)]
    appendme = True
    while appendme:
        thispoint = newgrid[-1]
        newpoint = thispoint + thispoint/splscale(thispoint)
        if newpoint > np.max(wgrid): appendme = False
        if appendme: newgrid.append(newpoint)
        pass
    dnewgrid = np.concatenate((np.diff(newgrid),
                               np.array([np.diff(newgrid)[-1]])))
    if verbose:
        plt.figure()
        plt.plot(wgrid, fnsres, 'k+', label='Reference')
        plt.plot(wgrid, splscale(wgrid), 'g', label=res+' Scaled')
        plt.plot(newgrid, newgrid/dnewgrid, 'g+', label=res+' New Grid')
        plt.legend(frameon=False, loc=4)
        plt.xlabel('Wavelength $\lambda$[$\mu m$]')
        plt.ylabel('Resolving Power $R$')
        plt.yscale('log')
        plt.show()
        pass
    newstarsnr = []
    for myw, mydw in zip(newgrid, dnewgrid):
        thissnr = float(splsnr(myw)*np.sqrt(mydw)/np.sqrt(spldw(myw)))
        newstarsnr.append(thissnr)
        pass
    newgrid = np.array(newgrid)
    newstarsnr = np.array(newstarsnr)
    if verbose:
        plt.figure()
        plt.plot(wgrid, starsnr, 'k+', label='Reference')
        plt.plot(newgrid, newstarsnr, 'g+', label=res)
        plt.legend(frameon=False, loc=4)
        plt.xlabel('Wavelength $\lambda$[$\mu m$]')
        plt.ylabel('Stellar SNR')
        plt.yscale('log')
        plt.show()
        pass
    return newgrid, newstarsnr
# -------------------------- -----------------------------------------
# -- ATMOS FOR FINESSE -- --------------------------------------------
def fatmos(priors, xsecs, qtgrid, hzlib, sv,
           res='R60',
           cialist=['H2-H2', 'H2-He', 'H2-H', 'He-H'],
           xmollist=['CH4', 'H2O', 'HCN', 'H2CO', 'CO', 'CO2', 'NH3'],
           mclen=int(1e4),
           snrdir='/proj/sdp/data/CERBERUS/FINESSE/SNR',
           verbose=False, debug=False, domcmc=False):
    orbp = priors['orbpar'][0]
    pnet = 'b'
    if orbp['sn'] == '55 Cnc': pnet = 'e'
    if orbp['sn'] == 'KOI-314': pnet = 'd'
    rp0hss = orbp[pnet]['rp']
    temp = orbp['Ts']*np.sqrt(orbp['Rs']/(2.*orbp[pnet]['a']))
    tceq = True
    target = orbp['sn']
    wgrid, starsnr = getwgrid(snrdir, res, filename=target+'.dat',
                              verbose=False)
    params = Parameters()
    params.add('CtoO', value=0., min=-10., max=10., vary=True)
    params.add('XtoH', value=0., min=-10, max=2.8, vary=True)
    params.add('TopP', value=1., min=-6., max=1., vary=False)
    params.add('Haze', value=-10, min=-10., max=10., vary=False)
    params.add('Rp0', value=rp0hss, vary=False)
    params.add('Tp', value=temp, min=296., max=orbp['Ts'], vary=False)
    model = lmcrbmodel(params, wgrid, orbp=orbp,
                       xsecs=xsecs, qtgrid=qtgrid,
                       cialist=cialist, xmollist=xmollist,
                       hzlib=hzlib, hzp='SPLJAV',
                       ce=tceq, shape=False, verbose=False)
    starsnr *= np.sqrt(2.)  # Equivalent SNR for the transit duration
    dataerr = np.random.randn(model.size)/starsnr
    data = model + dataerr
    shapedata = data - np.mean(data)
    if debug:
        plt.figure()
        plt.errorbar(wgrid, 1e2*data, yerr=1e2*dataerr, fmt='*')
        plt.plot(wgrid, 1e2*model, linewidth=3.)
        plt.xlabel('Wavelength $\lambda$[$\mu m$]')
        plt.ylabel('Transit Depth [%]')
        plt.show()
        pass
    if domcmc:
        # PRIORS
        mcmcmtlct = Normal('XtoH', 0., 1./1.**2)
        mcmcnrch = Normal('CtoO', 0., 1./1.**2)
        # CERBERUS MODEL
        @pm.deterministic
        def ccerberus(mtlct=mcmcmtlct, nrch=mcmcnrch, solidr=rp0hss,
                      tpr=temp, crborbp=orbp, crbxsecs=xsecs,
                      crbqtgrid=qtgrid, crbwgrid=wgrid,
                      crbhzlib=hzlib,
                      crbcialist=cialist, crbxmollist=xmollist):
            tceq = {'CtoO':float(nrch), 'XtoH':float(mtlct)}
            cmodel = crbmodel(None, -10., 1., solidr,
                              crborbp, crbxsecs, crbqtgrid,
                              tpr, crbwgrid,
                              False, False, crbcialist, crbxmollist,
                              hzlib=crbhzlib, hzp='SPLJAV', cheq=tceq)
            cmodel = cmodel - np.mean(cmodel)
            return cmodel
        # CERBERUS MCMC
        mcdata = Normal('mcdata',
                        mu=ccerberus, tau=1./(np.mean(dataerr**2)),
                        value=shapedata, observed=True)
        mcmcmodel = pm.Model([mcdata, mcmcmtlct, mcmcnrch])
        markovc = pm.MCMC(mcmcmodel)
        markovc.sample(mclen, burn=2*int(mclen/1e1), progress_bar=verbose)
        if verbose: print('')
        mcpost = markovc.stats()
        bestvalues = {}
        chains = {}
        for thiskey in mcpost:
            thischain = markovc.trace(thiskey)[:]
            chains[thiskey] = thischain
            chainout = np.median(thischain)
            bestvalues[thiskey] = chainout
            pass
        sv['MODEL'].append(bestvalues)
        sv['MCPOST'].append(mcpost)
        sv['MCCHAINS'].append(chains)
        pass
    else:
        out = minimize(lmcrbmodel, params,
                       args=(wgrid, shapedata,
                             dataerr, orbp, xsecs,
                             qtgrid, cialist, xmollist,
                             hzlib, 'SPLJAV',
                             tceq, True))
        sv['MODEL'].append(out.params)
        lmpost = {}
        for thiskey in out.params:
            lmpost[thiskey] = {'VALUE':out.params[thiskey].value,
                               'ERR':out.params[thiskey].stderr}
            pass
        sv['MCPOST'].append(lmpost)
        pass
    return
# ----------------------- --------------------------------------------
if __name__ == "__main__":
    pass
