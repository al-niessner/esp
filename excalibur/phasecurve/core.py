'''phasecurve core ds'''
# -- IMPORTS -- ------------------------------------------------------
import io
import copy
import numpy as np
import matplotlib.pyplot as plt

import excalibur.system.core as syscore
import excalibur.transit.core as trncore

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
    trncore.ctxt = CONTEXT(alt=alt, ald=ald, allz=allz, orbp=orbp,
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
    def is_mime():
        '''is_mime ds'''
        return True

    @property
    def profile_mu(self):
        '''profile_mu ds'''
        return self._mu
    pass
setattr(ldtk, 'LDPSet', LDPSet)
setattr(ldtk.ldtk, 'LDPSet', LDPSet)

# ----------------- --------------------------------------------------
# -- NORMALIZATION -- ------------------------------------------------
def norm_spitzer(cal, tme, fin, _ext, out, selftype):
    '''
    K. PEARSON: aperture selection, remove nans, remove zeros, 3 sigma clip time series
    '''
    normed = False
    priors = fin['priors'].copy()

    planetloop = [pnet for pnet in tme['data'].keys() if
                  (pnet in priors.keys()) and tme['data'][pnet][selftype]]

    for p in planetloop:
        out['data'][p] = {}

        keys = [
            'TIME','WX','WY','FRAME',
        ]
        for k in keys:
            out['data'][p][k] = np.array(cal['data'][k])

        # is set later during aperture selection
        out['data'][p]['PHOT'] = np.zeros(len(cal['data']['TIME']))
        out['data'][p]['NOISEPIXEL'] = np.zeros(len(cal['data']['TIME']))

        # remove nans
        nanmask = np.isnan(out['data'][p]['PHOT'])
        for k in out['data'][p].keys():
            nanmask = nanmask | np.isnan(out['data'][p][k])

        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][~nanmask]

        # time order things
        ordt = np.argsort(out['data'][p]['TIME'])
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][ordt]
        cflux = np.array(cal['data']['PHOT'])[~nanmask][ordt]
        cnpp = np.array(cal['data']['NOISEPIXEL'])[~nanmask][ordt]

        # 3 sigma clip flux time series
        phase = (out['data'][p]['TIME'] - fin['priors'][p]['t0'])/fin['priors'][p]['period']
        badmask = np.zeros(out['data'][p]['TIME'].shape).astype(bool)
        for i in np.unique(tme['data'][p][selftype]):
            # mask out phase curve data (previous eclipse | transit | eclipse)
            omask = (phase > i-0.75) & (phase < i+0.75) & ~np.isnan(out['data'][p]['TIME'])

            if omask.sum() == 0:
                continue

            dt = np.nanmean(np.diff(out['data'][p]['TIME'][omask]))*24*60
            ndt = int(7/dt)*2+1
            # aperture selection
            stds = []
            for j in range(cflux.shape[1]):
                stds.append(np.nanstd(trncore.sigma_clip(cflux[omask,j], ndt)))

            bi = np.argmin(stds)
            out['data'][p]['PHOT'][omask] = cflux[omask,bi]
            out['data'][p]['NOISEPIXEL'][omask] = cnpp[omask,bi]

            # sigma clip and remove nans
            photmask = np.isnan(trncore.sigma_clip(out['data'][p]['PHOT'][omask], ndt))
            xmask = np.isnan(trncore.sigma_clip(out['data'][p]['WX'][omask], ndt))
            ymask = np.isnan(trncore.sigma_clip(out['data'][p]['WY'][omask], ndt))
            nmask = np.isnan(trncore.sigma_clip(out['data'][p]['NOISEPIXEL'][omask], ndt))

            badmask[omask] = photmask | xmask | ymask | nmask

        # remove outliers
        for k in out['data'][p].keys():
            out['data'][p][k] = out['data'][p][k][~badmask]

        # pass information along
        out['data'][p]['transit'] = tme['data'][p]['transit']
        out['data'][p]['eclipse'] = tme['data'][p]['eclipse']
        out['data'][p]['phasecurve'] = tme['data'][p]['phasecurve']

        if out['data'][p][selftype]:
            normed = True
            out['STATUS'].append(True)

    return normed

def phasecurve_spitzer(nrm, fin, out, selftype, fltr):
    '''
    K. PEARSON: modeling of phase curves
    '''
    wl= False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = list(nrm['data'].keys())

    for p in planetloop:

        out['data'][p] = []

        # extract data based on phase
        if selftype == 'transit':
            phase = (nrm['data'][p]['TIME'] - fin['priors'][p]['t0'])/fin['priors'][p]['period']
        elif selftype == 'eclipse':
            priors = fin['priors']
            w = priors[p].get('omega',0)
            tme = priors[p]['t0']+ priors[p]['period']*0.5 * (1 + priors[p]['ecc']*(4./np.pi)*np.cos(np.deg2rad(w)))
            phase = (nrm['data'][p]['TIME'] - tme)/fin['priors'][p]['period']
        if selftype == 'phasecurve':
            phase = (nrm['data'][p]['TIME'] - fin['priors'][p]['t0'])/fin['priors'][p]['period']

        # loop through epochs
        ec = 0  # event counter
        for event in nrm['data'][p][selftype]:
            print('processing event:',event)

            # compute phase + priors
            smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
            smaors_up = (priors[p]['sma']+priors[p]['sma_uperr'])/(priors['R*']-abs(priors['R*_lowerr']))/ssc['Rsun/AU']
            smaors_lo = (priors[p]['sma']-abs(priors[p]['sma_lowerr']))/(priors['R*']+priors['R*_uperr'])/ssc['Rsun/AU']
            priors[p]['ars'] = smaors

            tmid = priors[p]['t0'] + event*priors[p]['period']
            rprs = (priors[p]['rp']*7.1492e7) / (priors['R*']*6.955e8)
            # inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
            w = priors[p].get('omega',0)

            # mask out data by event type
            pmask = (phase > event-0.65) & (phase < event+0.65)  # should check for eccentric orbits

            if pmask.sum() == 0:
                continue

            # extract aperture photometry data
            subt = nrm['data'][p]['TIME'][pmask]
            aper = nrm['data'][p]['PHOT'][pmask]
            aper_err = np.sqrt(aper)

            fpfs = trncore.eclipse_ratio(priors, p, fltr)
            edepth = fpfs*rprs**2

            tpars = {
                # Star
                'T*':priors['T*'],

                # transit
                'rprs': rprs,
                'ars': smaors,
                'tmid':tmid,
                'per': priors[p]['period'],
                'inc': priors[p]['inc'],

                # eclipse
                'fpfs': fpfs,
                'omega': priors['b'].get('omega',0),
                'ecc': priors['b']['ecc'],

                # limb darkening (nonlinear - exotethys - pylightcurve)
                'u0':priors[p].get('u0',0),
                'u1':priors[p].get('u1',0),
                'u2':priors[p].get('u2',0),
                'u3':priors[p].get('u3',0),

                # phase curve amplitudes
                'c0':0, 'c1':edepth*0.25, 'c2':0, 'c3':0, 'c4':0
            }

            # gather detrending parameters
            wxa = nrm['data'][p]['WX'][pmask]
            wya = nrm['data'][p]['WY'][pmask]
            npp = nrm['data'][p]['NOISEPIXEL'][pmask]
            syspars = np.array([wxa,wya,npp]).T

            # only report params with bounds, all others will be fixed to initial value
            mybounds = {
                'rprs':[0.5*rprs,1.5*rprs],
                'tmid':[tmid-0.01,tmid+0.01],
                'ars':[smaors_lo,smaors_up],

                'fpfs':[0,fpfs*2],
                # 'omega':[priors[p]['omega']-25,priors[p]['omega']+25],
                # 'ecc': [0,priors[p]['ecc']+0.1],

                # 'c0':[-0.025,0.025], gets set in code
                'c1':[0, 0.5*edepth],
                'c2':[-0.15*edepth, 0.15*edepth]
                }

            # 10 minute time scale
            nneighbors = int(10./24./60./np.mean(np.diff(subt)))
            print(" N neighbors:",nneighbors)
            print(" N datapoints:", len(subt))

            myfit = trncore.pc_fitter(subt, aper, aper_err, tpars, mybounds, syspars,
                                      neighbors=nneighbors)

            # copy best fit parameters and uncertainties
            for k in myfit.bounds.keys():
                print(f" {k} = {myfit.parameters[k]:.6f} +/- {myfit.errors[k]:.6f}")
                pass
            out['data'][p].append({})
            out['data'][p][ec]['time'] = subt
            out['data'][p][ec]['flux'] = aper
            out['data'][p][ec]['err'] = aper_err
            out['data'][p][ec]['xcent'] = wxa
            out['data'][p][ec]['ycent'] = wya
            out['data'][p][ec]['npp'] = npp
            out['data'][p][ec]['wf'] = myfit.wf
            out['data'][p][ec]['model'] = myfit.model
            out['data'][p][ec]['transit'] = myfit.transit
            out['data'][p][ec]['residuals'] = myfit.residuals
            out['data'][p][ec]['detrended'] = myfit.detrended
            out['data'][p][ec]['filter'] = fltr
            out['data'][p][ec]['final_pars'] = copy.deepcopy(myfit.parameters)
            out['data'][p][ec]['final_errs'] = copy.deepcopy(myfit.errors)

            # extract plot data for states.py
            def save_plot(plotfn):
                fig,_ = plotfn()
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                return buf.getvalue()

            out['data'][p][ec]['plot_bestfit'] = save_plot(myfit.plot_bestfit)
            out['data'][p][ec]['plot_posterior'] = save_plot(myfit.plot_posterior)
            out['data'][p][ec]['plot_pixelmap'] = save_plot(myfit.plot_pixelmap)
            out['data'][p][ec]['plot_btempcurve'] = save_plot(myfit.plot_btempcurve)

            ec += 1
            out['STATUS'].append(True)
            wl = True

            pass
        pass
    return wl
