import excalibur.system.core as syscore

import numpy as np
import math

# ------------- ------------------------------------------------------
# -- ESTIMATOR PROTOTYPES  -- ----------------------------------------
class Estimator:
    def __init__(self, name, descr, plot='hist', units=None, scale=None, ref=None):
        self._name = name
        self._descr = descr
        self._plot = plot
        self._units = units
        self._scale = scale
        self._ref = ref

    def name(self): return self._name
    def descr(self): return self._descr
    def plot(self): return self._plot
    def units(self): return self._units
    def scale(self): return self._scale
    def ref(self): return self._ref

class PlEstimator(Estimator):
    def __init__(self, name, descr, plot='hist', units=None, scale=None,
                 method=None, ref=None):
        Estimator.__init__(self, name, descr, plot, units, scale, ref)
        self._method = method

    def run(self, priors, ests, pl):
        return self._method(priors, ests, pl)

class StEstimator(Estimator):
    def __init__(self, name, descr, plot='hist', units=None, scale=None,
                 method=None, ref=None):
        Estimator.__init__(self, name, descr, plot, units, scale, ref)
        self._method = method

    def run(self, priors, ests):
        return self._method(priors, ests)

# ------------- ------------------------------------------------------
# -- COLLECTION OF ESTIMATORS ----------------------------------------
# ------------- ------------------------------------------------------

class StellarTypeEstimator(StEstimator):
    def __init__(self):
        StEstimator.__init__(self, name='stellar_type', descr='Harvard system stellar type',
                             scale=['M', 'K', 'G', 'F', 'A', 'B', 'unknown'],
                             plot='bar')

    def run(self, priors, ests):
        # Estimates the Harvard system spectral type using the
        # prior stellar temperature from NExSci
        # Temperature ranges taken from:
        # https://en.wikipedia.org/wiki/Stellar_classification
        st_type = 'unknown'  # default value
        st_temp = priors['T*']
        if st_temp >= 30e3: st_type = 'O'
        elif st_temp >= 10e3: st_type = 'B'
        elif st_temp >= 7.5e3: st_type = 'A'
        elif st_temp >= 6e3: st_type = 'F'
        elif st_temp >= 5.2e3: st_type = 'G'
        elif st_temp >= 3.7e3: st_type = 'K'
        elif st_temp >= 2.4e3: st_type = 'M'
        return st_type

class TeqEstimator(PlEstimator):
    def __init__(self):
        PlEstimator.__init__(self, name='Teq', descr='Equilibrium temperature',
                             units='K')

    def run(self, priors, ests, pl):
        sscmks = syscore.ssconstants(mks=True)
        eqtemp = priors['T*']*np.sqrt(priors['R*']*sscmks['Rsun/AU']/
                                      (2.*priors[pl]['sma']))
        return eqtemp

def pl_density(priors, _ests, pl):
    sscmks = syscore.ssconstants(cgs=True)
    volume = (4.0/3)*math.pi*priors[pl]['rp']**3
    if priors[pl]['mass'] == '':  # abort for targets without mass estimates
        return None
    density = priors[pl]['mass']/volume
    # now convert from Jupiter masses per Jupiter radii to g/cm^3
    conversion = sscmks['Mjup']/(sscmks['Rjup']**3)
    density = density*conversion
    return density

def st_luminosity(priors, _ests):
    sscmks = syscore.ssconstants(mks=True)
    Tsun = sscmks['Tsun']
    est = priors['R*']**2*(priors['T*']/Tsun)**4
    return est

def pl_insolation(priors, ests, pl):
    ins = ests['luminosity']*(priors[pl]['sma']**-2)
    # apply eccentricity correction
    ins *= math.sqrt(1/(1-priors[pl]['ecc']**2))
    return ins
