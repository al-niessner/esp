# -- IMPORTS -- ------------------------------------------------------
# import excalibur.data.core as datcore
import excalibur.system.core as syscore
# import excalibur.cerberus.core as crbcore
# pylint: disable=import-self
import excalibur.transit.core
from excalibur.transit.core import transit, weightedflux, gaussian_weights, sigma_clip, get_ld

# import re
# import requests
import logging
import numpy as np
# import lmfit as lm
# import time as timer

import pymc3 as pm
log = logging.getLogger(__name__)
pymc3log = logging.getLogger('pymc3')
pymc3log.setLevel(logging.ERROR)

# import scipy.constants as cst
import matplotlib.pyplot as plt
# from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

# import ctypes
import copy
# from os import environ

import theano
import theano.tensor as tt
import theano.compile.ops as tco

theano.config.exception_verbosity = 'high'

import os
# COMMENT FOLLOWING WHEN MERGING TO MASTER
os.environ["LDTK_ROOT"] = '/home/rzellem/.ldtk'


from collections import namedtuple
CONTEXT = namedtuple('CONTEXT', ['alt', 'ald', 'allz', 'orbp', 'commonoim', 'ecc',
                                 'g1', 'g2', 'g3', 'g4', 'ootoindex', 'ootorbits',
                                 'orbits', 'period', 'selectfit', 'smaors', 'time',
                                 'tmjd', 'ttv', 'valid', 'visits', 'aos', 'avi'])
ctxt = CONTEXT(alt=None, ald=None, allz=None, orbp=None, commonoim=None, ecc=None,
               g1=None, g2=None, g3=None, g4=None, ootoindex=None, ootorbits=None,
               orbits=None, period=None, selectfit=None, smaors=None, time=None,
               tmjd=None, ttv=None, valid=None, visits=None, aos=None, avi=None)
def ctxtupdt(alt=None, ald=None, allz=None, orbp=None, commonoim=None, ecc=None,
             g1=None, g2=None, g3=None, g4=None, ootoindex=None, ootorbits=None,
             orbits=None, period=None, selectfit=None, smaors=None, time=None,
             tmjd=None, ttv=None, valid=None, visits=None, aos=None, avi=None):
    '''
    G. ROUDIER: Update global context for pymc3 deterministics
    '''
    excalibur.transit.core.ctxt = CONTEXT(alt=alt, ald=ald, allz=allz, orbp=orbp,
                                          commonoim=commonoim, ecc=ecc, g1=g1, g2=g2,
                                          g3=g3, g4=g4, ootoindex=ootoindex,
                                          ootorbits=ootorbits, orbits=orbits,
                                          period=period, selectfit=selectfit,
                                          smaors=smaors, time=time, tmjd=tmjd, ttv=ttv,
                                          valid=valid, visits=visits, aos=aos, avi=avi)
    return

import ldtk
# from ldtk import LDPSetCreator, BoxcarFilter
# from ldtk.ldmodel import LinearModel, QuadraticModel, NonlinearModel
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

# ########################################################
# # LOAD IN TRANSIT FUNCTION FROM C

# # define 1d array pointer in python
# array_1d_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double,ndim=1,flags=['C_CONTIGUOUS','aligned'])

# # load library
# try:
#     lib_trans = np.ctypeslib.load_library('lib_transit.so','/proj/src/ae/excalibur/transit/')
# except:
#     # for jupyter notebook stuff
#     lib_trans = np.ctypeslib.load_library('lib_transit.so','/home/kpearson/esp/excalibur/transit/')
# # load fn from library and define inputs
# occultquadC = lib_trans.occultquad

# # inputs
# occultquadC.argtypes = [array_1d_double, ctypes.c_double, ctypes.c_double,
#                         ctypes.c_double, ctypes.c_double, ctypes.c_double,
#                         ctypes.c_double, ctypes.c_double, ctypes.c_double,
#                         ctypes.c_double, ctypes.c_double, array_1d_double]

# # no outputs, last *double input is saved over in C
# occultquadC.restype = None

def rampmodel(time, pars):
    return 1 + pars['a1']*np.exp(-1*time/pars["a2"])

def phasecurvemodel(time, pars):
    # R. Zellem's phasecurve model
    phase = (time - pars["tm"])/pars["per"]
    phase = phase - int(np.nanmin(phase))

    # Make the model lightcurves
    # Transit
    lct = transit(time=time, values=pars)

    # Eclipse1
    # Zero out ld for eclipse model
    epars = pars.copy()
    epars['u1'] = 0
    epars['u2'] = 0
    epars['tm'] = pars['Tmide']
    lce1 = transit(time=time, values=epars)

    # Inclusions of phase curve variations
    c1 = pars["c1"]
    c2 = pars["c2"]
    c3 = pars["c3"]
    c4 = pars["c4"]
    tP = phase

    # Create the phasecurve portion
    phasecurve = c1*np.cos(2.*np.pi*tP) + c2*np.sin(2.*np.pi*tP) + c3*np.cos(4.*np.pi*tP) + c4*np.sin(4.*np.pi*tP)

    # Subtract off the minimum so that the lightcurve's min = 0
    lce1 = lce1 - np.nanmin(lce1)

    # Normalize the lightcurve between 0 and 1
    lce1 = lce1/(pars["rp"]**2.)

    idxe1, = np.where(lce1 < np.nanmedian(lce1))

    try:
        phasecurveoffset1 = np.mean(phasecurve[idxe1[0]:idxe1[-1]])
    except IndexError:
        phasecurveoffset1=0

    # Add in phasecurve curvature to each eclipse model
    lce1 = lce1*(phasecurve + pars["FpFs1"] + (-1.)*phasecurveoffset1)

    lce = lce1

    model = lce + lct

    return model

# ----------------- --------------------------------------------------
# -- NORMALIZATION -- ------------------------------------------------
def norm_spitzer(cal, tme, fin, ext, out, selftype, verbose=False, debug=False):
    '''
    K. PEARSON: prep data for light curve fitting
        remove nans, remove zeros, 3 sigma clip time series
    '''
    normed = False
    priors = fin['priors'].copy()

    planetloop = [pnet for pnet in tme['data'].keys()
                  if (pnet in priors.keys()) and tme['data'][pnet][selftype]]

    for p in planetloop:
        if verbose or debug:
            print("Working on planet: ", p)
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
        visits = tme['data'][p]['pcvisits']
        orbits = tme['data'][p]['orbits']

        # remove nans and zeros
        mask = np.isnan(flux) | np.isnan(pflux) | (pflux == 0) | (flux == 0)
        keys = [
            'TIME','WX','WY',
            'G_PSF_ERR', 'G_PSF', 'G_XCENT', 'G_YCENT',
            'G_SIGMAX', 'G_SIGMAY', 'G_ROT', 'G_MODEL',
        ]
        for k in keys:
            out['data'][p][k] = np.array(cal['data'][k])[~mask]
        out['data'][p]['pcvisits'] = visits[~mask]
        out['data'][p]['orbits'] = orbits[~mask]
        out['data'][p]['NOISEPIXEL'] = noisep[~mask]
        out['data'][p]['PHOT'] = flux[~mask]

        # time order things
        ordt = np.argsort(out['data'][p]['TIME'])
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][ordt]

        # remove the first hour of data
        idxchopped = (out['data'][p]['TIME'] - np.nanmin(out['data'][p]['TIME']))*24. > 1
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][idxchopped]

        # 3 sigma clip flux time series
        badmask = np.zeros(out['data'][p]['TIME'].shape).astype(bool)
        for i in tme['data'][p]['phasecurve']:
            print(ext, i)
            omask = out['data'][p]['pcvisits'] == i

            dt = np.nanmean(np.diff(out['data'][p]['TIME'][omask]))*24*60
            medf = median_filter(out['data'][p]['PHOT'][omask], int(15/dt)*2+1)
            res = out['data'][p]['PHOT'][omask] - medf
            photmask = np.abs(res) > 3*np.std(res)

            medf = median_filter(out['data'][p]['G_PSF'][omask], int(15/dt)*2+1)
            res = out['data'][p]['G_PSF'][omask] - medf
            psfmask = np.abs(res) > 3*np.std(res)

            badmask[omask] = photmask | psfmask

        if debug:
            plt.plot(out['data'][p]['TIME'][~badmask], out['data'][p]['PHOT'][~badmask], 'k.',label='Filtered')
            plt.plot(out['data'][p]['TIME'][badmask], out['data'][p]['PHOT'][badmask], 'r.',label='Raw')
            plt.legend()
            plt.show()

        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][~badmask]

        # pass information along
        out['data'][p]['phasecurve'] = tme['data'][p]['phasecurve']
        if debug:
            plt.plot(out['data'][p]['TIME'], out['data'][p]['G_PSF'],'g.',label='G_PSF')
            plt.plot(out['data'][p]['TIME'], out['data'][p]['PHOT'],'k.',label='PHOT')
            plt.xlabel('Time')
            plt.ylabel('Flux')
            plt.legend()
            plt.show()
        if out['data'][p][selftype]:
            normed = True
            out['STATUS'].append(True)
    return normed

# from excalibur.transit.core import time_bin

def phasecurve_spitzer(nrm, fin, out, selftype, fltr, chainlen=25000, verbose=False, debug=False):
    wl= False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = [pnet for pnet in nrm['data'].keys()
                  if (pnet in priors.keys()) and nrm['data'][pnet][selftype]]

    for p in planetloop:
        if verbose or debug:
            print("Working on planet: ", p)

        time = nrm['data'][p]['TIME']
        visits = nrm['data'][p]['pcvisits']
        out['data'][p] = []

        # loop through epochs
        ec = 0  # event counter
        for event in nrm['data'][p]['phasecurve']:
            # # To run only the first event
            # if ec > 0:
            #     break
            # To run only the second event
            if ec == 0:
                out['data'][p].append({})
                ec = ec +1
                continue
            print('processing event:',event)
            emask = visits == event
            out['data'][p].append({})

            # get data
            time = nrm['data'][p]['TIME'][emask]
            tmask = time < 2400000.5  # remove later
            time[tmask] += 2400000.5

            # compute phase + priors
            smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
            # z, phase = datcore.time2z(time, priors[p]['inc'], priors[p]['t0'], smaors, priors[p]['period'], priors[p]['ecc'])
            # tdur = priors[p]['period']/(2*np.pi)/smaors  # rough estimate
            rprs = (priors[p]['rp']*7.1492e7) / (priors['R*']*6.955e8)
            inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
            fpfs = 0.01
            tmid = priors[p]['t0']
            # Sometimes the prior mid-transit time is after the current data...and for some reason, that messes up
            # the function "transit" - so just roll back the clock....
            while tmid >= max(time):
                tmid = tmid - priors[p]['period']
            tmid_err = np.sqrt(np.abs(priors[p]['t0_lowerr']*priors[p]['t0_uperr']))
            # tmide = tmid-priors[p]['period']/2.
            w = priors[p].get('omega',0)
            tmide = priors[p]['t0'] + priors[p]['period']*0.5 * (1 + priors[p]['ecc']*(4./np.pi)*np.cos(np.deg2rad(w)))  # to account for elliptical orbits
            # tmide_err = np.sqrt(priors[p]['t0_lowerr']*priors[p]['t0_uperr'])
            # Use Monte Carlo to estimate the uncertainty on tmide ***ASSUMES GAUSSIAN PRIORS***
            t0_draws = np.random.normal(priors[p]['t0'], tmid_err, 100000)
            period_draws = np.random.normal(priors[p]['period'], np.sqrt(np.abs(priors[p]['period_lowerr']*priors[p]['period_uperr'])), 100000)
            ecc_draws = np.random.normal(priors[p]['ecc'], np.sqrt(np.abs(priors[p]['ecc_lowerr']*priors[p]['ecc_uperr'])), 100000)
            w_draws = np.random.normal(w, np.sqrt(np.abs(priors[p].get('omega_lowerr',0)*priors[p].get('omega_uperr',0))), 100000)
            tmide_err_draws = t0_draws + period_draws*0.5 * (1 + ecc_draws*(4./np.pi)*np.cos(np.deg2rad(w_draws)))  # to account for elliptical orbits
            tmide_err = np.nanmax([np.nanstd(tmide_err_draws), 1./24.])  # uses either MC uncertainty or 1 hr, whichever is greater

            # extract aperture photometry data
            subt = time
            aper = nrm['data'][p]['PHOT'][emask]
            aper_err = np.sqrt(aper)
            gpsf = nrm['data'][p]['G_PSF'][emask]
            # gpsf_err = np.sqrt(gpsf)

            if '3.6' in fltr:
                lin,quad = get_ld(priors,'Spit36')
            elif '4.5' in fltr:
                lin,quad = get_ld(priors,'Spit45')

            # can't solve for wavelengths greater than below
            # whiteld = createldgrid([2.5],[2.6], priors, segmentation=int(10), verbose=verbose)
            # whiteld = createldgrid([wmin],[wmax], priors, segmentation=int(1), verbose=verbose)
            # '''
            # # LDTK breaks for Spitzer https://github.com/hpparvi/ldtk/issues/11 4
            # filters = [BoxcarFilter('a', 3150, 3950)]
            # tstar = priors['T*']
            # terr = np.sqrt(abs(priors['T*_uperr']*priors['T*_lowerr']))
            # fehstar = priors['FEH*']
            # feherr = np.sqrt(abs(priors['FEH*_uperr']*priors['FEH*_lowerr']))
            # loggstar = priors['LOGG*']
            # loggerr = np.sqrt(abs(priors['LOGG*_uperr']*priors['LOGG*_lowerr']))
            # sc = LDPSetCreator(teff=(tstar, terr), logg=(loggstar, loggerr), z=(fehstar, feherr), filters=filters)
            # ps = sc.create_profiles(nsamples=int(1e4))
            # cq,eq = ps.coeffs_qd(do_mc=True)
            # '''

            tpars = {
                'rp': rprs,
                'tm': tmid,  # np.median(subt),
                'tm_err': tmid_err,
                'tmide': tmide,
                'tmide_err': tmide_err,
                'ar':smaors,
                'per':priors[p]['period'],
                'inc':priors[p]['inc'],
                'inc_lim': inc_lim,
                'u1':lin, 'u2': quad,
                'ecc':priors[p]['ecc'],
                'ome': 0,
                'a0':1, 'a1':0, 'a2':0,
                'c1':1, 'c2':0, 'c3':0, 'c4':0,
                'FpFs1':fpfs,
            }

            if verbose:
                data = transit(time=subt, values=tpars)
                plt.plot(subt,aper/np.median(aper),'ko')
                plt.plot(subt,data,'g--')
                plt.show()

            # perform a quick sigma clip
            dt = np.nanmean(np.diff(subt))*24*60  # minutes
            medf = median_filter(aper, int(15/dt)*2+1)  # needs to be odd
            res = aper - medf
            photmask = np.abs(res) < 3*np.std(res)

            # resize aperture data
            ta = subt[photmask]
            aper_err = aper_err[photmask]/np.nanmedian(aper[photmask])  # normalize the flux
            aper = aper[photmask]/np.nanmedian(aper[photmask])  # normalize the flux
            wxa = nrm['data'][p]['WX'][emask][photmask]
            wya = nrm['data'][p]['WY'][emask][photmask]
            npp = nrm['data'][p]['NOISEPIXEL'][emask][photmask]

            # sigma clip PSF data
            dt = np.nanmean(np.diff(subt))*24*60  # minutes
            medf = median_filter(gpsf, int(15/dt)*2+1)  # needs to be odd
            res = gpsf - medf
            # psfmask = np.abs(res) < 3*np.std(res)

            # PSF photometry
            # tp = subt[psfmask]
            # psf_err = gpsf_err[psfmask]/np.nanmedian(gpsf[psfmask])  # normalize the flux
            # psf = gpsf[psfmask]/np.nanmedian(gpsf[psfmask])  # normalize the flux
            # wxp = nrm['data'][p]['G_XCENT'][emask][psfmask]
            # wyp = nrm['data'][p]['G_YCENT'][emask][psfmask]
            # sxp = nrm['data'][p]['G_SIGMAX'][emask][psfmask]
            # syp = nrm['data'][p]['G_SIGMAY'][emask][psfmask]
            # rot = nrm['data'][p]['G_ROT'][emask][psfmask]

            # if verbose:
            #     # quick decorrelation to assess priors + phase mask
            #     wf = weightedflux(aper, gw, nearest)
            #     frac = len(aper) % 20
            #     btt = subt[:-1*frac].reshape(-1,20).mean(1)
            #     baper = (aper/wf /np.median(aper/wf))[:-1*frac].reshape(-1,20).mean(1)

            #     plt.plot(subt, (aper/wf)/np.median(aper/wf), 'ko')
            #     plt.plot(btt, baper, 'go')
            #     plt.show()

            # aperture selection
            # estimate quadratic coefficients
            # A = np.vstack([(subt-min(subt))**2, subt-min(subt), np.ones(subt.shape)]).T
            # a2,a1,a0 = np.linalg.lstsq(A, aper)[0]
            # quad = np.matmul(A,np.array([a2,a1,a0]))

            def fit_data(time,flux,fluxerr, syspars, tpars):

                gw, nearest = gaussian_weights(
                    np.array(syspars).T,
                    # w=np.array([1,1])
                )

                gw[np.isnan(gw)] = 0.01

                @tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar],otypes=[tt.dvector])
                def phasecurve2min(*pars):
                    # rprs, tmid, inc, fpfs, Tmide, norm, c1, c2, c3, c4 = pars
                    rprs, tmid, inc, fpfs, Tmide, c1, c2, c3, c4 = pars
                    tpars['rp'] = float(rprs)
                    tpars['tm'] = float(tmid)
                    tpars['inc']= float(inc)
                    tpars['FpFs1'] = float(fpfs)
                    tpars['Tmide'] = float(Tmide)
                    tpars['c1'] = float(c1)
                    tpars['c2'] = float(c2)
                    tpars['c3'] = float(c3)
                    tpars['c4'] = float(c4)
                    lcmodel = phasecurvemodel(time, tpars)
                    detrended = flux/lcmodel
                    wf=np.sum(detrended[nearest]*gw,axis=-1)
                    return lcmodel*wf  # *float(norm)

                with pm.Model():
                    priors = [
                        pm.Uniform('rprs', lower=0.5*tpars['rp'],  upper=1.5*tpars['rp']),
                        # pm.Uniform('tmid', lower=min(time), upper=max(time)),
                        pm.Normal('tmid', mu=tpars['tm'], tau=1./(tpars['tm_err']**2.)),
                        pm.Uniform('inc',  lower=tpars['inc_lim'], upper=90),
                        pm.Uniform('FpFs1',  lower=0, upper=0.01),
                        # pm.Uniform('Tmide',  lower=min(time), upper=max(time)),
                        pm.Normal('Tmide',  mu=tpars['tmide'], tau=1./(tpars['tmide_err']**2)),
                        # pm.Uniform('norm', lower=0.9,  upper=1.1 ),
                        pm.Uniform('c1', lower=0, upper=0.01),  # upper=0.003),
                        pm.Uniform('c2', lower=0, upper=0.01),  # upper=0.003),
                        pm.Uniform('c3', lower=0, upper=0.01),  # upper=0.003),
                        pm.Uniform('c4', lower=0, upper=0.01)]  # upper=0.003),

                    pm.Normal('likelihood',
                              mu=phasecurve2min(*priors),
                              tau=(1./fluxerr)**2,
                              observed=flux
                             )

                    trace = pm.sample(chainlen,  # 12500
                                      pm.Metropolis(),
                                      cores=8,  # multi-processing
                                      chains=4,
                                      tune=0,  # 500,
                                      progress_bar=True
                                     )

                    # logp = model.logp
                    # values = np.array([logp(point) for point in trace.points()])

                return trace

            def fit_data_ramp(time,flux,fluxerr, syspars, tpars):

                gw, nearest = gaussian_weights(
                    np.array(syspars).T,
                    # w=np.array([1,1])
                )

                gw[np.isnan(gw)] = 0.01

                @tco.as_op(itypes=[tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar],otypes=[tt.dvector])
                def phasecurve2min(*pars):
                    # rprs, tmid, inc, fpfs, Tmide, norm, c1, c2, c3, c4 = pars
                    rprs, tmid, inc, fpfs, Tmide, c1, c2, c3, c4, a1, a2 = pars
                    tpars['rp'] = float(rprs)
                    tpars['tm'] = float(tmid)
                    tpars['inc']= float(inc)
                    tpars['FpFs1'] = float(fpfs)
                    tpars['Tmide'] = float(Tmide)
                    tpars['c1'] = float(c1)
                    tpars['c2'] = float(c2)
                    tpars['c3'] = float(c3)
                    tpars['c4'] = float(c4)
                    tpars['a1'] = float(a1)
                    tpars['a2'] = float(a2)
                    lcmodel = phasecurvemodel(time, tpars)*rampmodel(time, tpars)
                    detrended = flux/lcmodel
                    wf=np.sum(detrended[nearest]*gw,axis=-1)
                    # import pdb; pdb.set_trace()
                    return lcmodel*wf  # *float(norm)

                with pm.Model():
                    priors = [
                        pm.Uniform('rprs', lower=0.5*tpars['rp'],  upper=1.5*tpars['rp']),
                        # pm.Uniform('tmid', lower=min(time), upper=max(time)),
                        pm.Normal('tmid', mu=tpars['tm'], tau=1./(tpars['tm_err']**2.)),
                        pm.Uniform('inc',  lower=tpars['inc_lim'], upper=90),
                        pm.Uniform('FpFs1',  lower=0, upper=0.01),
                        # pm.Uniform('Tmide',  lower=min(time), upper=max(time)),
                        pm.Normal('Tmide',  mu=tpars['tmide'], tau=1./(tpars['tmide_err']**2)),
                        # pm.Uniform('norm', lower=0.9,  upper=1.1 ),
                        pm.Uniform('c1', lower=0, upper=0.01),  # upper=0.003),
                        pm.Uniform('c2', lower=0, upper=0.01),  # upper=0.003),
                        pm.Uniform('c3', lower=0, upper=0.01),  # upper=0.003),
                        pm.Uniform('c4', lower=0, upper=0.01),  # upper=0.003),
                        pm.Normal('a1', mu=1, tau=1./(np.nanstd(flux)**2.)),
                        pm.Uniform('a2', lower=0, upper=100)
                        # pm.Uniform('w1',   lower=0, upper=1), # weights for instrument model
                        # pm.Uniform('w2',   lower=0, upper=1)
                        ]

                    pm.Normal('likelihood',
                              mu=phasecurve2min(*priors),
                              tau=(1./fluxerr)**2,
                              observed=flux
                             )

                    trace = pm.sample(chainlen,  # 12500
                                      pm.Metropolis(),
                                      cores=8,  # multi-processing
                                      chains=4,
                                      tune=0,  # 500,
                                      progress_bar=True
                                     )

                    # logp = model.logp
                    # values = np.array([logp(point) for point in trace.points()])

                return trace

            # analyze different extraction techniques
            print("Optimal aperture.")
            print("Sparse sampling of the data.")
            print("No ramp.")
            trace_aper = fit_data(ta[::100],aper[::100],aper_err[::100], [wxa[::100],wya[::100]], tpars)

            print("Analyzing chains.")
            tpars['tm'] = np.median(trace_aper['tmid'])
            # tpars['a0'] = np.median(trace_aper['norm'])
            tpars['rp'] = np.median(trace_aper['rprs'])
            tpars['inc']= np.median(trace_aper['inc'])
            tpars['FpFs1'] = np.median(trace_aper['FpFs1'])
            tpars['Tmide'] = np.median(trace_aper['Tmide'])
            tpars['c1'] = np.median(trace_aper['c1'])
            tpars['c2'] = np.median(trace_aper['c2'])
            tpars['c3'] = np.median(trace_aper['c3'])
            tpars['c4'] = np.median(trace_aper['c4'])

            print("Constructing best-fit model.")
            lcmodel = phasecurvemodel(ta[::100], tpars)

            detrended = aper[::100]/lcmodel
            gw, nearest = gaussian_weights(np.array([wxa[::100],wya[::100]]).T)
            wf = weightedflux(detrended, gw, nearest)

            chi2 = np.nansum(aper[::100]/wf/lcmodel)

            print("Ramp.")
            trace_aper_ramp = fit_data_ramp(ta[::100],aper[::100],aper_err[::100], [wxa[::100],wya[::100]], tpars)

            print("Analyzing chains.")
            tpars['tm'] = np.median(trace_aper_ramp['tmid'])
            # tpars['a0'] = np.median(trace_aper['norm'])
            tpars['rp'] = np.median(trace_aper_ramp['rprs'])
            tpars['inc']= np.median(trace_aper_ramp['inc'])
            tpars['FpFs1'] = np.median(trace_aper_ramp['FpFs1'])
            tpars['Tmide'] = np.median(trace_aper_ramp['Tmide'])
            tpars['c1'] = np.median(trace_aper_ramp['c1'])
            tpars['c2'] = np.median(trace_aper_ramp['c2'])
            tpars['c3'] = np.median(trace_aper_ramp['c3'])
            tpars['c4'] = np.median(trace_aper_ramp['c4'])
            tpars['a1'] = np.median(trace_aper_ramp['a1'])
            tpars['a2'] = np.median(trace_aper_ramp['a2'])

            print("Constructing best-fit model.")
            lcmodel_ramp = phasecurvemodel(ta[::100], tpars)*rampmodel(ta[::100], tpars)

            detrended = aper[::100]/lcmodel_ramp
            gw, nearest = gaussian_weights(np.array([wxa[::100],wya[::100]]).T)
            wf_ramp = weightedflux(detrended, gw, nearest)

            chi2_ramp = np.nansum(aper[::100]/wf_ramp/lcmodel_ramp)

            print("Full sampling of the winner.")
            print(chi2, chi2_ramp)

            if chi2 < chi2_ramp:
                print("No ramp needed.")
                trace_aper = fit_data(ta,aper,aper_err, [wxa,wya], tpars)
            else:
                print("Ramp needed.")
                trace_aper = fit_data_ramp(ta,aper,aper_err, [wxa,wya], tpars)

            print("Analyzing chains.")
            tpars['tm'] = np.median(trace_aper['tmid'])
            # tpars['a0'] = np.median(trace_aper['norm'])
            tpars['rp'] = np.median(trace_aper['rprs'])
            tpars['inc']= np.median(trace_aper['inc'])
            tpars['FpFs1'] = np.median(trace_aper['FpFs1'])
            tpars['Tmide'] = np.median(trace_aper['Tmide'])
            tpars['c1'] = np.median(trace_aper['c1'])
            tpars['c2'] = np.median(trace_aper['c2'])
            tpars['c3'] = np.median(trace_aper['c3'])
            tpars['c4'] = np.median(trace_aper['c4'])

            if chi2 < chi2_ramp:
                print("Constructing best-fit model.")
                lcmodel = phasecurvemodel(ta, tpars)
            else:
                tpars['a1'] = np.median(trace_aper['a1'])
                tpars['a2'] = np.median(trace_aper['a2'])
                print("Constructing best-fit model.")
                lcmodel = phasecurvemodel(ta, tpars)*rampmodel(ta, tpars)

            detrended = aper/lcmodel
            gw, nearest = gaussian_weights(np.array([wxa,wya]).T)
            wf = weightedflux(detrended, gw, nearest)

            print("Writing state vectors.")
            out['data'][p][ec]['aper_time'] = ta
            out['data'][p][ec]['aper_flux'] = aper
            out['data'][p][ec]['aper_err'] = aper_err
            out['data'][p][ec]['aper_xcent'] = wxa
            out['data'][p][ec]['aper_ycent'] = wya
            out['data'][p][ec]['aper_trace'] = pm.trace_to_dataframe(trace_aper)
            out['data'][p][ec]['aper_wf'] = wf
            out['data'][p][ec]['aper_model'] = lcmodel
            out['data'][p][ec]['aper_pars'] = copy.deepcopy(tpars)
            out['data'][p][ec]['noise_pixel'] = npp
            if chi2 < chi2_ramp:
                out['data'][p][ec]['noise_pixel'] = np.ones(len(ta))
            else:
                out['data'][p][ec]['noise_pixel'] = rampmodel(ta, tpars)

            ec += 1
            out['STATUS'].append(True)
            wl = True

            pass
        pass
    return wl

# '''
# (Pdb) z, phase = datcore.time2z( time, priors[p]['inc'], priors[p]['t0'], smaors, priors[p]['period'], priors[p]['ecc'] )
# (Pdb) plt.plot( phase[pmask][photmask], aper/wf, 'ko'); plt.show()

# f,ax = plt.subplots(3)
# ax[0].plot(subt, aper,'ko')
# ax[0].set_ylabel('Counts [ADU]')
# ax[1].plot(subt, wx,'ro')
# ax[1].set_ylabel('X Centroid [px]')
# ax[2].plot(subt, wy,'bo')
# ax[2].set_xlabel('Time [JD]')
# ax[2].set_ylabel('Y Centroid [px]')
# plt.show()
# f,ax = plt.subplots(2,2)
# ax[0,0].plot(wx-np.median(wx), aper/np.median(aper), 'go', alpha=0.5)
# ax[0,0].set_xlabel('X Centroid [px]')
# ax[0,0].set_ylabel('Relative Flux')
# ax[0,0].set_title('Aperture Photometry')
# ax[0,1].plot(wx-np.median(wx), gpsfm/np.median(gpsfm), 'bo', alpha=0.5)
# ax[0,1].set_xlabel('X Centroid [px]')
# ax[0,1].set_ylabel('Relative Flux')
# ax[0,1].set_title('PSF Photometry')
# ax[1,0].plot(wy-np.median(wy), aper/np.median(aper), 'go', alpha=0.5)
# ax[1,0].set_xlabel('Y Centroid [px]')
# ax[1,0].set_ylabel('Relative Flux')
# ax[1,1].plot(wy-np.median(wy), gpsfm/np.median(gpsfm), 'bo', alpha=0.5)
# ax[1,1].set_xlabel('Y Centroid [px]')
# ax[1,1].set_ylabel('Relative Flux')
# plt.show()
# '''
