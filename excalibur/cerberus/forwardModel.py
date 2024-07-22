'''cerberus forwardModel ds'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst
from scipy.interpolate import interp1d as itp
import logging; log = logging.getLogger(__name__)

import excalibur
# pylint: disable=import-self
import excalibur.cerberus.forwardModel  # is this needed for the context updater?
import excalibur.system.core as syscore
from excalibur.util.cerberus import crbce,getmmw

import theano.tensor as tt
import theano.compile.ops as tco

# -- GLOBAL CONTEXT FOR PYMC3 DETERMINISTICS ---------------------------------------------
from collections import namedtuple
CONTEXT = namedtuple('CONTEXT', ['cleanup', 'model', 'p', 'solidr', 'orbp', 'tspectrum',
                                 'xsl', 'spc', 'modparlbl', 'hzlib', 'fixedParams'])
ctxt = CONTEXT(cleanup=None, model=None, p=None, solidr=None, orbp=None, tspectrum=None,
               xsl=None, spc=None, modparlbl=None, hzlib=None, fixedParams=None)
def ctxtupdt(cleanup=None, model=None, p=None, solidr=None, orbp=None, tspectrum=None,
             xsl=None, spc=None, modparlbl=None, hzlib=None, fixedParams=None):
    '''
    G. ROUDIER: Update context
    '''
    excalibur.cerberus.forwardModel.ctxt = CONTEXT(cleanup=cleanup, model=model, p=p,
                                                   solidr=solidr, orbp=orbp,
                                                   tspectrum=tspectrum, xsl=xsl, spc=spc,
                                                   fixedParams=fixedParams,
                                                   modparlbl=modparlbl, hzlib=hzlib)
    return
# ----------- --------------------------------------------------------
# -- CERBERUS MODEL -- -----------------------------------------------
def crbmodel(mixratio, rayleigh, cloudtp, rp0, orbp, xsecs, qtgrid,
             temp, wgrid, lbroadening=False, lshifting=False,
             cialist=['H2-H', 'H2-H2', 'H2-He', 'He-H'].copy(),
             xmollist=['TIO', 'CH4', 'H2O', 'H2CO', 'HCN', 'CO', 'CO2', 'NH3'].copy(),
             # nlevels=100, Hsmax=15., solrad=10.,
             # increase the number of scale heights from 15 to 20, to match the Ariel forward model
             nlevels=100, Hsmax=20., solrad=10.,
             hzlib=None, hzp=None, hzslope=-4., hztop=None, hzwscale=1e0,
             cheq=None, logx=False, pnet='b',
             break_down_by_molecule=False,
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
    if not mixratio:
        if cheq is None: log.warning('neither mixratio nor cheq are defined')
        mixratio, fH2, fHe = crbce(p, temp,
                                   C2Or=cheq['CtoO'], X2Hr=cheq['XtoH'],
                                   N2Or=cheq['NtoO'])
        # mixratio['CO2'] = mixratio['NH3']
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
    tau, tau_by_molecule, wtau = gettau(
        xsecs, qtgrid, temp, mixratio, z, dz, rho, rp0, p, wgrid,
        lbroadening, lshifting, cialist, fH2, fHe, xmollist, rayleigh,
        hzlib, hzp, hzslope, hztop, hzwscale=hzwscale,
        debug=debug)
    if not break_down_by_molecule:
        tau_by_molecule = {}
    molecules = tau_by_molecule.keys()
    # SEMI FINITE CLOUD ------------------------------------------------------------------
    reversep = np.array(p[::-1])
    selectcloud = p > 10.**cloudtp
    blocked = False
    if all(selectcloud):
        tau = tau*0
        for molecule in molecules:
            tau_by_molecule[molecule] = tau_by_molecule[molecule]*0
        blocked = True
        pass
    if not all(~selectcloud) and not blocked:
        cloudindex = np.max(np.arange(len(p))[selectcloud])+1
        for index in np.arange(wtau.size):
            myspl = itp(reversep, tau[:,index])
            tau[cloudindex,index] = myspl(10.**cloudtp)
            tau[:cloudindex,index] = 0.
            for molecule in molecules:
                myspl = itp(reversep, tau_by_molecule[molecule][:,index])
                tau_by_molecule[molecule][cloudindex,index] = myspl(10.**cloudtp)
                tau_by_molecule[molecule][:cloudindex,index] = 0.
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
    models_by_molecule = {}
    for molecule in molecules:
        atmdepth = 2e0*np.array(np.mat((rp0+np.array(z))*np.array(dz))*
                                np.mat(1. - np.exp(-tau_by_molecule[molecule]))).flatten()
        models_by_molecule[molecule] = (rp0**2 + atmdepth)/(orbp['R*']*ssc['Rsun'])**2
        models_by_molecule[molecule] = models_by_molecule[molecule][::-1]
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
    if break_down_by_molecule:
        return model[::-1], models_by_molecule
    return model[::-1]

# --------------------------- ----------------------------------------
# -- TAU -- ----------------------------------------------------------
def gettau(xsecs, qtgrid, temp, mixratio,
           z, dz, rho, rp0, p, wgrid, lbroadening, lshifting,
           cialist, fH2, fHe, xmollist, rayleigh, hzlib, hzp, hzslope, hztop,
           isothermal=True, hzwscale=1e0,
           debug=False):
    '''
    G. ROUDIER: Builds optical depth matrix
    '''
    # SPHERICAL SHELL (PLANE-PARALLEL REMOVED) -------------------------------------------
    # MATRICES INIT ------------------------------------------------------------------
    tau = np.zeros((len(z), wgrid.size))
    tau_by_molecule = {}
    # DL ARRAY, Z VERSUS ZPRIME ------------------------------------------------------
    dlarray = []
    zprime = np.array(z)
    dzprime = np.array(dz)
    for iz, thisz in enumerate(z):
        dl = np.sqrt((rp0 + zprime + dzprime)**2 - (rp0 + thisz)**2)
        dl[:iz] = 0e0
        # oof nasty bug here!
        #  sometimes equal terms are off by the instrument precision
        #  so sqrt(1e15 - 1e15) = sqrt(-1) = NaN
        # take absolute value, just to be sure there's no problem
        # dl[iz:] = dl[iz:] - np.sqrt((rp0 + zprime[iz:])**2 - (rp0 + thisz)**2)
        dl[iz:] = dl[iz:] - np.sqrt(np.abs((rp0 + zprime[iz:])**2 - (rp0 + thisz)**2))
        dlarray.append(dl)
        pass
    dlarray = np.array(dlarray)
    # GAS ARRAY, ZPRIME VERSUS WAVELENGTH  -------------------------------------------
    for elem in mixratio:
        # tau_by_molecule[elem] = np.zeros((len(z), wgrid.size))
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
        if isothermal:
            tau = tau + mmr*sigma*np.array([rho]).T
            tau_by_molecule[elem] = mmr*sigma*np.array([rho]).T
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
        tau_by_molecule[cia] = f1*f2*sigma*np.array([rho**2]).T
        pass
    # RAYLEIGH ARRAY, ZPRIME VERSUS WAVELENGTH  --------------------------------------
    # NAUS & UBACHS 2000
    slambda0 = 750.*1e-3  # microns
    sray0 = 2.52*1e-28*1e-4  # m^2/mol
    sigma = sray0*(wgrid[::-1]/slambda0)**(-4)
    tau = tau + fH2*sigma*np.array([rho]).T
    tau_by_molecule['rayleigh'] = fH2*sigma*np.array([rho]).T
    # HAZE ARRAY, ZPRIME VERSUS WAVELENGTH  ------------------------------------------
    if hzlib is None:
        slambda0 = 750.*1e-3  # microns
        sray0 = 2.52*1e-28*1e-4  # m^2/mol
        sigma = sray0*(wgrid[::-1]/slambda0)**(hzslope)
        hazedensity = np.ones(len(z))
        tau = tau + (10.**rayleigh)*sigma*np.array([hazedensity]).T
        tau_by_molecule['haze'] = (10.**rayleigh)*sigma*np.array([hazedensity]).T
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
        tau_by_molecule['haze'] = (10.**rayleigh)*sigma*np.array([rh]).T
        pass
    tau = 2e0*np.array(np.mat(dlarray)*np.mat(tau))
    molecules = tau_by_molecule.keys()
    for molecule in molecules:
        tau_by_molecule[molecule] = \
            2e0*np.array(np.mat(dlarray)*np.mat(tau_by_molecule[molecule]))

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
    return tau, tau_by_molecule, 1e4/lsig
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
# ----------------------------- --------------------------------------
# -- PRESSURE BROADENING -- ------------------------------------------
def intflor(wave, dwave, nu, gamma):
    '''
G. ROUDIER: Pressure Broadening
    '''
    f = 1e0/np.pi*(np.arctan((wave+dwave - nu)/gamma) -
                   np.arctan((wave-dwave - nu)/gamma))
    return f
# -------------------------- -----------------------------------------
# -- PYMC3 DETERMINISTIC FUNCTIONS -- --------------------------------
@tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar,
                   tt.dvector],
           otypes=[tt.dvector])
def cloudyfmcerberus(*crbinputs):
    '''
    G. ROUDIER: Wrapper around Cerberus forward model, spherical shell symmetry
    '''
    ctp, hza, hzloc, hzthick, tpr, mdp = crbinputs
    # ctp, hza, hzloc, hzthick, tpr, mdp = crbinputs
    # print(' not-fixed cloud parameters (cloudy):',ctp,hza,hzloc,hzthick)
    fmc = np.zeros(ctxt.tspectrum.size)
    if ctxt.model == 'TEC':
        tceqdict = {}
        mdpindex = 0
        if 'XtoH' in ctxt.fixedParams:
            tceqdict['XtoH'] = ctxt.fixedParams['XtoH']
        else:
            tceqdict['XtoH'] = float(mdp[mdpindex])
            mdpindex += 1

        if 'CtoO' in ctxt.fixedParams:
            tceqdict['CtoO'] = ctxt.fixedParams['CtoO']
        else:
            tceqdict['CtoO'] = float(mdp[mdpindex])
            mdpindex += 1

        if 'NtoO' in ctxt.fixedParams:
            tceqdict['NtoO'] = ctxt.fixedParams['NtoO']
        else:
            tceqdict['NtoO'] = float(mdp[mdpindex])
        # print('XtoH,CtoO,NtoO =',tceqdict['XtoH'],tceqdict['CtoO'],tceqdict['NtoO'])

        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
    # ctp = 3.    # cloud deck is very deep - 1000 bars
    # hza = -10.  # small number means essentially no haze
    # hzloc = 0.
    # hzthick = 0.
    ctp = ctxt.fixedParams['CTP']
    hza = ctxt.fixedParams['HScale']
    hzloc = ctxt.fixedParams['HLoc']
    hzthick = ctxt.fixedParams['HThick']
    # print(' fixed cloud parameters (clear):',ctp,hza,hzloc,hzthick)

    tpr, mdp = crbinputs

    # if you don't want to fit Teq, fix it here. otherwise modify decorators somehow
    # tpr = 593.5
    # if 'T' in ctxt.fixedParams: exit('fixing T is tricky; need to adjust the decorators')
    # if 'T' in ctxt.fixedParams:
    #     tpr = ctxt.fixedParams['T']
    #     mdp = crbinputs
    # else:
    #    tpr, mdp = crbinputs

    fmc = np.zeros(ctxt.tspectrum.size)
    if ctxt.model == 'TEC':
        tceqdict = {}
        mdpindex = 0
        if 'XtoH' in ctxt.fixedParams:
            tceqdict['XtoH'] = ctxt.fixedParams['XtoH']
        else:
            tceqdict['XtoH'] = float(mdp[mdpindex])
            mdpindex += 1

        if 'CtoO' in ctxt.fixedParams:
            tceqdict['CtoO'] = ctxt.fixedParams['CtoO']
        else:
            tceqdict['CtoO'] = float(mdp[mdpindex])
            mdpindex += 1

        if 'NtoO' in ctxt.fixedParams:
            tceqdict['NtoO'] = ctxt.fixedParams['NtoO']
        else:
            tceqdict['NtoO'] = float(mdp[mdpindex])
        # print('XtoH,CtoO,NtoO =',tceqdict['XtoH'],tceqdict['CtoO'],tceqdict['NtoO'])

        fmc = crbmodel(None, float(hza), float(ctp), ctxt.solidr, ctxt.orbp,
                       ctxt.xsl['data'][ctxt.p]['XSECS'],
                       ctxt.xsl['data'][ctxt.p]['QTGRID'],
                       float(tpr), np.array(ctxt.spc['data'][ctxt.p]['WB']),
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=tceqdict, pnet=ctxt.p,
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
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
                       hzlib=ctxt.hzlib,  hzp='AVERAGE', hztop=float(hzloc),
                       hzwscale=float(hzthick), cheq=None, pnet=ctxt.p,
                       verbose=False, debug=False)
        pass
    fmc = fmc[ctxt.cleanup] - np.nanmean(fmc[ctxt.cleanup])
    fmc = fmc + np.nanmean(ctxt.tspectrum[ctxt.cleanup])
    ww = wbb
    ww = ww[ctxt.cleanup]
    cond_G102 = 'HST-WFC3-IR-G102-SCAN' in flt
    fmc[cond_G102] = fmc[cond_G102] + 1e-2*float(off0)
    return fmc
