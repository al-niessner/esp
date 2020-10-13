# -- IMPORTS -- ------------------------------------------------------
import copy
import ctypes
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

import dynesty
from dynesty.utils import resample_equal

from scipy.optimize import least_squares, brentq
from scipy.stats import gaussian_kde

try:
    import astropy.constants
    import astropy.units
    from astropy.modeling.models import BlackBody
except ImportError:
    from astropy.modeling.blackbody import blackbody_lambda as BlackBody

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
    def is_mime(): return True

    @property
    def profile_mu(self): return self._mu
    pass
setattr(ldtk, 'LDPSet', LDPSet)
setattr(ldtk.ldtk, 'LDPSet', LDPSet)

########################################################
# LOAD IN FUNCTIONS FROM C
try:
    # main pipeline
    lib_trans = np.ctypeslib.load_library('lib_transit.so','/lib')
except OSError:
    # local pipeline
    # /proj/sdp/lib/MandelTransit.c
    lib_trans = np.ctypeslib.load_library('lib_transit.so','/proj/sdp/lib')

# define 1d array pointer in python
array_1d_double = np.ctypeslib.ndpointer(dtype=ctypes.c_double,ndim=1,flags=['C_CONTIGUOUS','aligned'])

# transit, orbital radius, anomaly
input_type1 = [array_1d_double, ctypes.c_double, ctypes.c_double,
               ctypes.c_double, ctypes.c_double, ctypes.c_double,
               ctypes.c_double, ctypes.c_double, ctypes.c_double,
               ctypes.c_double, ctypes.c_double, array_1d_double]

# phasecurve, brightness, eclipse
input_type2 = [array_1d_double, array_1d_double, ctypes.c_double, ctypes.c_double,
               ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
               ctypes.c_double, ctypes.c_double, ctypes.c_double,
               ctypes.c_double, ctypes.c_double, array_1d_double]

# transit
occultquadC = lib_trans.occultquad
occultquadC.argtypes = input_type1
occultquadC.restype = None

# orbital radius
orbitalRadius = lib_trans.orbitalradius
orbitalRadius.argtypes = input_type1
orbitalRadius.restype = None

# true anomaloy
orbitalAnomaly = lib_trans.orbitalanomaly
orbitalAnomaly.argtypes = input_type1
orbitalAnomaly.restype = None

# phase curve
phaseCurve = lib_trans.phasecurve
phaseCurve.argtypes = input_type2
phaseCurve.restype = None

# phase curve without eclipse
brightnessCurve = lib_trans.brightness
brightnessCurve.argtypes = input_type2
brightnessCurve.restype = None

# eclipse
eclipseC = lib_trans.eclipse
eclipseC.argtypes = input_type2
eclipseC.restype = None

# cast arrays into C compatible format with pythonic magic
def format_args(f):
    @wraps(f)
    def wrapper(*args):
        if len(args) == 2:
            t = args[0]
            values = args[1]
        data = {}
        data['time'] = np.require(t,dtype=ctypes.c_double,requirements='C')
        data['model'] = np.require(np.ones(len(t)),dtype=ctypes.c_double,requirements='C')
        data['cvals'] = np.require(np.zeros(5),dtype=ctypes.c_double,requirements='C')
        if 'transit' in f.__name__ or 'orbit' in f.__name__ or 'anomaly' in f.__name__:
            keys=['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
        else:
            keys=['fpfs','rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
        data['vals'] = [values[k] for k in keys]
        for i,k in enumerate(['c0','c1','c2','c3','c4']): data['cvals'][i] = values[k]
        return f(*args, **data)
    return wrapper

@format_args
def brightness(_t, _values, **kwargs):
    '''
    :INPUT:
        t - ndarray
        values - dictionary with
            keys=['fpfs','rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
    :OUTPUT:
        ndarray
    '''
    brightnessCurve(kwargs['time'], kwargs['cvals'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']

@format_args
def phasecurve(_t, _values, **kwargs):
    phaseCurve(kwargs['time'], kwargs['cvals'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']

@format_args
def eclipse(_t, _values, **kwargs):
    eclipseC(kwargs['time'], kwargs['cvals'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']

@format_args
def orbitalradius(_t, _values, **kwargs):
    # keys=['rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
    orbitalRadius(kwargs['time'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']

@format_args
def trueanomaly(_t, _values, **kwargs):
    orbitalAnomaly(kwargs['time'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']

@format_args
def transit(_t, _values, **kwargs):
    occultquadC(kwargs['time'], *kwargs['vals'], len(kwargs['time']), kwargs['model'])
    return kwargs['model']
########################################################


class lc_fitter:
    '''
    K. PEARSON class to fit phase curves with LM or nested sampling
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self, time, data, dataerr, prior, bounds, syspars, neighbors=100, mode='ns'):
        self.time = time
        self.data = data
        self.dataerr = dataerr
        self.prior = prior
        self.bounds = bounds
        self.syspars = syspars
        self.neighbors = neighbors

        if mode == 'ns':
            self.fit_nested()
        else:
            self.fit_lm()

    def fit_lm(self):

        freekeys = list(self.bounds.keys())
        boundarray = np.array([self.bounds[k] for k in freekeys])

        # trim data around predicted transit/eclipse time
        self.gw, self.nearest = trncore.gaussian_weights(self.syspars, neighbors=self.neighbors)

        # alloc arrays for C
        time = np.require(self.time,dtype=ctypes.c_double,requirements='C')
        lightcurve = np.require(np.zeros(len(time)),dtype=ctypes.c_double,requirements='C')
        cvals = np.require(np.zeros(5),dtype=ctypes.c_double,requirements='C')

        def lc2min(pars):
            for i,par in enumerate(pars):
                self.prior[freekeys[i]] = par

            # call C function
            keys = ['fpfs', 'rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
            vals = [self.prior[k] for k in keys]
            for i,k in enumerate(['c0','c1','c2','c3','c4']): cvals[i] = self.prior[k]
            phaseCurve(self.time, cvals, *vals, len(self.time), lightcurve)

            detrended = self.data/lightcurve
            wf = trncore.weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            return ((self.data-model)/self.dataerr)**2

        res = least_squares(lc2min, x0=[self.prior[k] for k in freekeys],
                            bounds=[boundarray[:,0], boundarray[:,1]], jac='3-point',
                            loss='linear', method='dogbox', ftol=1e-4, tr_options='exact')

        self.parameters = copy.deepcopy(self.prior)
        self.errors = {}

        for i,k in enumerate(freekeys):
            self.parameters[k] = res.x[i]
            self.errors[k] = 0

        # best fit model
        self.transit = phasecurve(self.time, self.parameters)
        detrended = self.data / self.transit
        self.wf = trncore.weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.residuals = self.data - self.model
        self.detrended = self.data/self.wf
        self.phase = (self.time-self.parameters['tmid'])/self.parameters['per']

    def fit_nested(self):
        freekeys = list(self.bounds.keys())

        # trim data around predicted transit/eclipse time
        self.gw, self.nearest = trncore.gaussian_weights(self.syspars, neighbors=self.neighbors)

        # alloc arrays for C
        time = np.require(self.time,dtype=ctypes.c_double,requirements='C')
        lightcurve = np.require(np.zeros(len(time)),dtype=ctypes.c_double,requirements='C')
        cvals = np.require(np.zeros(5),dtype=ctypes.c_double,requirements='C')

        def loglike(pars):
            # update free parameters
            for i, par in enumerate(pars):
                self.prior[freekeys[i]] = par

            # call C function
            keys = ['fpfs', 'rprs','ars','per','inc','u1','u2','ecc','omega','tmid']
            vals = [self.prior[k] for k in keys]
            for i,k in enumerate(['c0','c1','c2','c3','c4']): cvals[i] = self.prior[k]
            phaseCurve(time, cvals, *vals, len(time), lightcurve)

            detrended = self.data/lightcurve
            wf = trncore.weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            return -0.5 * np.sum(((self.data-model)**2/self.dataerr**2))

        def prior_transform(upars):
            freekeys = list(self.bounds.keys())
            boundarray = np.array([self.bounds[k] for k in freekeys])
            bounddiff = np.diff(boundarray,1).reshape(-1)
            vals = (boundarray[:,0] + bounddiff*upars)

            # set limits of phase amplitude to be less than eclipse depth
            edepth = vals[freekeys.index('rprs')]**2 * vals[freekeys.index('fpfs')]
            for k in ['c1','c2','c3','c4']:
                ki = freekeys.index(k)
                vals[ki] = upars[ki] * edepth - 0.5*edepth

            return vals

        dsampler = dynesty.NestedSampler(loglike, prior_transform, len(freekeys), sample='unif', bound='multi', nlive=1000)
        dsampler.run_nested(maxiter=2e6, print_progress=False, maxcall=2e6)
        self.results = dsampler.results
        del self.results['bound']

        # alloc data for best fit + error
        self.errors = {}
        self.quantiles = {}
        self.parameters = copy.deepcopy(self.prior)

        tests = [copy.deepcopy(self.prior) for i in range(6)]

        # Derive kernel density estimate for best fit
        weights = np.exp(self.results.logwt - self.results.logz[-1])
        samples = self.results['samples']
        logvol = self.results['logvol']
        wt_kde = gaussian_kde(resample_equal(-logvol, weights))  # KDE
        logvol_grid = np.linspace(logvol[0], logvol[-1], 1000)  # resample
        wt_grid = wt_kde.pdf(-1*logvol_grid)  # evaluate KDE PDF
        self.weights = np.interp(-logvol, -1*logvol_grid, wt_grid)  # interpolate

        # errors + final values
        mean, cov = dynesty.utils.mean_and_cov(self.results.samples, weights)
        mean2, _cov2 = dynesty.utils.mean_and_cov(self.results.samples, self.weights)
        for i,fkey in enumerate(freekeys):
            self.errors[freekeys[i]] = cov[i,i]**0.5
            tests[0][fkey] = mean[i]
            tests[1][fkey] = mean2[i]

            counts, bins = np.histogram(samples[:,i], bins=100, weights=weights)
            mi = np.argmax(counts)
            tests[5][freekeys[i]] = bins[mi] + 0.5*np.mean(np.diff(bins))

            # finds median and +- 2sigma, will vary from mode if non-gaussian
            self.quantiles[freekeys[i]] = dynesty.utils.quantile(self.results.samples[:,i], [0.025, 0.5, 0.975], weights=weights)
            tests[2][freekeys[i]] = self.quantiles[freekeys[i]][1]

        # find minimum near weighted mean
        mask = (samples[:,0] < self.parameters[freekeys[0]]+2*self.errors[freekeys[0]]) & (samples[:,0] > self.parameters[freekeys[0]]-2*self.errors[freekeys[0]])
        bi = np.argmin(self.weights[mask])

        for i, fkey in enumerate(freekeys):
            tests[3][fkey] = samples[mask][bi,i]
            tests[4][fkey] = np.average(samples[mask][:,i],weights=self.weights[mask],axis=0)

        # find best fit
        chis = []
        res = []
        for i, test in enumerate(tests):
            lightcurve = phasecurve(self.time, test)
            detrended = self.data / lightcurve
            wf = trncore.weightedflux(detrended, self.gw, self.nearest)
            model = lightcurve*wf
            residuals = self.data - model
            res.append(residuals)
            btime, br = trncore.time_bin(self.time, residuals)
            blc = transit(btime, tests[i])
            mask = np.ones(blc.shape,dtype=bool)
            # Future add more ephemesis on in transit fits
            duration = btime[mask].max() - btime[mask].min()
            tmask = ((btime - tests[i]['tmid']) < duration) & ((btime - tests[i]['tmid']) > -1*duration)
            chis.append(np.mean(br[tmask]**2))

        mi = np.argmin(chis)
        self.parameters = copy.deepcopy(tests[mi])
        # plt.scatter(samples[mask,0], samples[mask,1], c=weights[mask]); plt.show()

        # best fit model
        self.transit = phasecurve(self.time, self.parameters)
        detrended = self.data / self.transit
        self.wf = trncore.weightedflux(detrended, self.gw, self.nearest)
        self.model = self.transit*self.wf
        self.residuals = self.data - self.model
        self.detrended = self.data/self.wf
        self.phase = (self.time-self.parameters['tmid'])/self.parameters['per']

    def plot_bestfit(self, bin_dt=10./(60*24), zoom=False, phase=True):
        f = plt.figure(figsize=(12,7))
        # f.subplots_adjust(top=0.94,bottom=0.08,left=0.07,right=0.96)
        ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
        ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
        axs = [ax_lc, ax_res]

        bt, bf = trncore.time_bin(self.time, self.detrended, bin_dt)
        bp = (bt-self.parameters['tmid'])/self.parameters['per']

        if phase:
            axs[0].plot(bp,bf,'co',alpha=0.5,zorder=2)
            axs[0].plot(self.phase, self.transit, 'r-', zorder=3)
            axs[0].set_xlim([min(self.phase), max(self.phase)])
            axs[0].set_xlabel("Phase ")
        else:
            axs[0].plot(bt,bf,'co',alpha=0.5,zorder=2)
            axs[0].plot(self.time, self.transit, 'r-', zorder=3)
            axs[0].set_xlim([min(self.time), max(self.time)])
            axs[0].set_xlabel("Time [day]")

        axs[0].set_ylabel("Relative Flux")
        axs[0].grid(True,ls='--')

        if zoom:
            axs[0].set_ylim([1-1.15*self.parameters['rprs']**2, 1+0.25*self.parameters['rprs']**2])
        else:
            if phase:
                axs[0].errorbar(self.phase, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.01)
            else:
                axs[0].errorbar(self.time, self.detrended, yerr=np.std(self.residuals)/np.median(self.data), ls='none', marker='.', color='black', zorder=1, alpha=0.01)

        bt, br = trncore.time_bin(self.time, self.residuals/np.median(self.data)*1e6, bin_dt)
        bp = (bt-self.parameters['tmid'])/self.parameters['per']

        if phase:
            axs[1].plot(self.phase, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.15, label=r'$\sigma$ = {:.0f} ppm'.format(np.std(self.residuals/np.median(self.data)*1e6)))
            axs[1].plot(bp,br,'c.',alpha=0.5,zorder=2,label=r'$\sigma$ = {:.0f} ppm'.format(np.std(br)))
            axs[1].set_xlim([min(self.phase), max(self.phase)])
            axs[1].set_xlabel("Phase")

        else:
            axs[1].plot(self.time, self.residuals/np.median(self.data)*1e6, 'k.', alpha=0.15, label=r'$\sigma$ = {:.0f} ppm'.format(np.std(self.residuals/np.median(self.data)*1e6)))
            axs[1].plot(bt,br,'c.',alpha=0.5,zorder=2,label=r'$\sigma$ = {:.0f} ppm'.format(np.std(br)))
            axs[1].set_xlim([min(self.time), max(self.time)])
            axs[1].set_xlabel("Time [day]")

        axs[1].legend(loc='best')
        axs[1].set_ylabel("Residuals [ppm]")
        axs[1].grid(True,ls='--')
        plt.tight_layout()
        return f,axs

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

def plot_phasecurve(sv):

    fig = plt.figure(figsize=(13,7))
    ax_lc = plt.subplot2grid((4,5), (0,0), colspan=5,rowspan=3)
    ax_res = plt.subplot2grid((4,5), (3,0), colspan=5, rowspan=1)
    axs = [ax_lc, ax_res]

    phase = (sv['aper_time']-sv['aper_pars']['tmid'])/sv['aper_pars']['per']
    bin_dt = 10./24./60.
    bt, bf = trncore.time_bin(sv['aper_time'], sv['aper_detrended'], bin_dt)
    bp = (bt-sv['aper_pars']['tmid'])/sv['aper_pars']['per']
    bt, br = trncore.time_bin(sv['aper_time'], sv['aper_residuals'], bin_dt)

    # Tb = brightnessTemp(sv['aper_pars'],sv['aper_filter'])
    # compute the brightness temperature over the orbit
    bcurve = brightness(bt,sv['aper_pars'])
    tbcurve = np.ones(bcurve.shape)
    for i,bc in enumerate(bcurve):
        sv['aper_pars']['fpfs'] = max((bc-1)/sv['aper_pars']['rprs']**2, 0.00001)
        tbcurve[i] = brightnessTemp(sv['aper_pars'],sv['aper_filter'])

    # residuals
    axs[1].plot(phase, sv['aper_residuals']/np.median(sv['aper_flux'])*1e6, 'k.', alpha=0.15, label=r'$\sigma$ = {:.0f} ppm'.format(np.std(sv['aper_residuals']/np.median(sv['aper_flux'])*1e6)))

    axs[1].plot(bp,1e6*br/np.median(sv['aper_flux']),'w.',zorder=2,label=r'$\sigma$ = {:.0f} ppm'.format(np.std(1e6*br/np.median(sv['aper_flux']))))

    axs[1].set_xlim([min(phase), max(phase)])
    axs[1].set_xlabel("Phase")
    axs[1].legend(loc='best')
    axs[1].set_ylabel("Residuals [ppm]")
    axs[1].grid(True,ls='--')

    axs[0].errorbar(phase, sv['aper_detrended'], yerr=np.std(sv['aper_residuals'])/np.median(sv['aper_flux']), ls='none', marker='.', color='black',alpha=0.1, zorder=1)

    # map color to equilibrium temperature
    im = axs[0].scatter(bp,bf,marker='o',c=tbcurve,vmin=500,vmax=2750,cmap='jet', zorder=2, s=20)
    cbar = plt.colorbar(im)
    cbar.ax.set_xlabel("B. Temp. [K]")

    axs[0].plot(phase, sv['aper_transit'], 'w--', zorder=3)
    axs[0].set_xlim([min(phase), max(phase)])
    axs[0].set_xlabel("Phase ")

    axs[0].set_ylabel("Relative Flux")
    axs[0].grid(True,ls='--')
    axs[0].set_ylim([0.955,1.03])

    plt.tight_layout()
    return fig

def brightnessTemp(priors,f='IRAC 3.6um'):
    # Solve for Tb using Fp/Fs, Ts and a filter bandpass
    if '3.6' in f or '36' in f:
        waveset = np.linspace(3.15, 3.9, 1000) * astropy.units.micron
    else:
        waveset = np.linspace(4,5,1000) * astropy.units.micron

    def f2min(T, *args):
        fpfs,tstar,waveset = args
        fstar = BlackBody(waveset, tstar * astropy.units.K)
        fplanet = BlackBody(waveset, T * astropy.units.K)
        fp = np.trapz(fplanet, waveset)
        fs = np.trapz(fstar, waveset)
        return (fp/fs) - fpfs

    tb = brentq(f2min, 1,3500, args=(priors['fpfs'],priors['T*'],waveset))
    return tb

def eclipse_ratio(priors,p='b',f='IRAC 3.6um', verbose=True):

    Te = priors['T*']*(1-0.1)**0.25 * np.sqrt(0.5/priors[p]['ars'])

    rprs = priors[p]['rp'] * astropy.constants.R_jup / (priors['R*'] * astropy.constants.R_sun)
    tdepth = rprs.value**2

    # bandpass integrated flux for planet
    wave36 = np.linspace(3.15,3.95,1000) * astropy.units.micron
    wave45 = np.linspace(4,5,1000) * astropy.units.micron

    try:
        fplanet = BlackBody(Te*astropy.units.K)(wave36)
        fstar = BlackBody(priors['T*']*astropy.units.K)(wave36)
    except TypeError:
        fplanet = BlackBody(wave36, Te*astropy.units.K)
        fstar = BlackBody(wave36, priors['T*']*astropy.units.K)

    fp36 = np.trapz(fplanet, wave36)
    fs36 = np.trapz(fstar, wave36)

    try:
        fplanet = BlackBody(Te*astropy.units.K)(wave45)
        fstar = BlackBody(priors['T*']*astropy.units.K)(wave45)
    except TypeError:
        fplanet = BlackBody(wave45, Te*astropy.units.K)
        fstar = BlackBody(wave45, priors['T*']*astropy.units.K)

    fp45 = np.trapz(fplanet, wave45)
    fs45 = np.trapz(fstar, wave45)

    if verbose:
        print(" Stellar temp: {:.1f} K".format(priors['T*']))
        print(" Transit Depth: {:.4f} %".format(tdepth*100))

    if '3.6' in f or '36' in f:
        if verbose:
            print(" Eclipse Depth @ IRAC 1 (3.6um): ~{:.0f} ppm".format(tdepth*fp36/fs36*1e6))
            print("         Fp/Fs @ IRAC 1 (3.6um): ~{:.4f}".format(fp36/fs36))
        return float(fp36/fs36)
    else:
        if verbose:
            print(" Eclipse Depth @ IRAC 2 (4.5um): ~{:.0f} ppm".format(tdepth*fp45/fs45*1e6))
            print("         Fp/Fs @ IRAC 2 (4.5um): ~{:.4f}".format(fp45/fs45))
        return float(fp45/fs45)

def phasecurve_spitzer(nrm, fin, out, selftype, fltr, mode='ns'):
    '''
    K. PEARSON: modeling of phase curves
    '''
    wl= False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = [pnet for pnet in nrm['data'].keys()]

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
            # to do: update duration for eccentric orbits
            # https://arxiv.org/pdf/1001.2010.pdf eq 16
            tmid = priors[p]['t0'] + event*priors[p]['period']
            # tdur = priors[p]['period']/(np.pi)/smaors
            rprs = (priors[p]['rp']*7.1492e7) / (priors['R*']*6.955e8)
            # inc_lim = 90 - np.rad2deg(np.arctan((priors[p]['rp'] * ssc['Rjup/Rsun'] + priors['R*']) / (priors[p]['sma']/ssc['Rsun/AU'])))
            w = priors[p].get('omega',0)

            # mask out data by event type
            # pmask = (phase > event-1.5*tdur/priors[p]['period']) & (phase < event+1.5*tdur/priors[p]['period'])
            pmask = (phase > event-0.65) & (phase < event+0.65)  # should check for eccentric orbits

            if pmask.sum() == 0:
                continue

            # check if previous eclipse is present
            emask = (phase > event-0.65) & (phase < event-0.55)
            if emask.sum() > 10: tmid = priors[p]['t0'] + (event-1)*priors[p]['period']  # adjust prior for transit model

            # extract aperture photometry data
            subt = nrm['data'][p]['TIME'][pmask]
            aper = nrm['data'][p]['PHOT'][pmask]
            aper_err = np.sqrt(aper)

            try:
                if '36' in fltr:
                    lin,quad = trncore.get_ld(priors,'Spit36')
                elif '45' in fltr:
                    lin,quad = trncore.get_ld(priors,'Spit45')
            except ValueError:
                lin,quad = 0,0

            # can't solve for wavelengths greater than below
            # whiteld = createldgrid([2.5],[2.6], priors, segmentation=int(10), verbose=verbose)
            # whiteld = createldgrid([wmin],[wmax], priors, segmentation=int(1), verbose=verbose)

            # LDTK breaks for Spitzer https://github.com/hpparvi/ldtk/issues/11
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

            fpfs = eclipse_ratio(priors, p, fltr)
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

                # limb darkening (linear, quadratic)
                'u1': lin, 'u2': quad,

                # phase curve amplitudes
                'c0':0, 'c1':-edepth*0.25, 'c2':0, 'c3':0, 'c4':0
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
                'c1':[-edepth*0.35, 0.35*edepth],
                'c2':[-edepth*0.35, 0.35*edepth],
                'c3':[-edepth*0.35, 0.35*edepth],
                'c4':[-edepth*0.35, 0.35*edepth]
                }

            # 10 minute time scale
            nneighbors = int(10./24./60./np.mean(np.diff(subt)))
            print(" N neighbors:",nneighbors)
            print(" N datapoints:", len(subt))
            myfit = lc_fitter(subt, aper, aper_err, tpars, mybounds, syspars,neighbors=nneighbors, mode=mode)

            # copy best fit parameters and uncertainties
            terrs = {}
            for k in myfit.bounds.keys():
                print(" {} = {:.6f} +/- {:.6f}".format(k, myfit.parameters[k], myfit.errors[k]))
                tpars[k] = myfit.parameters[k]
                terrs[k] = myfit.errors[k]

            out['data'][p].append({})
            out['data'][p][ec]['aper_time'] = subt
            out['data'][p][ec]['aper_flux'] = aper
            out['data'][p][ec]['aper_err'] = aper_err
            out['data'][p][ec]['aper_xcent'] = wxa
            out['data'][p][ec]['aper_ycent'] = wya
            out['data'][p][ec]['aper_npp'] = npp
            try:
                # nested sampling only
                out['data'][p][ec]['aper_weights'] = myfit.weights
                out['data'][p][ec]['aper_results'] = myfit.results
                out['data'][p][ec]['aper_quantiles'] = myfit.quantiles
            except AttributeError:
                pass
            out['data'][p][ec]['aper_wf'] = myfit.wf
            out['data'][p][ec]['aper_model'] = myfit.model
            out['data'][p][ec]['aper_transit'] = myfit.transit
            out['data'][p][ec]['aper_residuals'] = myfit.residuals
            out['data'][p][ec]['aper_detrended'] = myfit.detrended
            out['data'][p][ec]['aper_filter'] = fltr
            out['data'][p][ec]['aper_pars'] = copy.deepcopy(tpars)
            out['data'][p][ec]['aper_errs'] = copy.deepcopy(terrs)

            # state vectors for classifer
            z, _phase = trncore.time2z(subt, tpars['inc'], tpars['tmid'], tpars['ars'], tpars['per'], tpars['ecc'])
            out['data'][p][ec]['postsep'] = z
            out['data'][p][ec]['allwhite'] = myfit.detrended
            out['data'][p][ec]['postlc'] = myfit.transit
            ec += 1
            out['STATUS'].append(True)
            wl = True

            # plot_phasecurve(out['data'][p][ec-1])
            # plt.show()
            # trncore.plot_pixelmap(out['data'][p][ec-1])
            # plt.show()
            # trncore.plot_posterior(out['data'][p][ec-1], fltr)
            # plt.show()
            pass
        pass
    return wl
