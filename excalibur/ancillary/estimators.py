import excalibur.system.core as syscore

import numpy as np
import scipy
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
                             plot='bar', ref='from T_star')

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
                             units='K', ref='albedo = 0')

    def run(self, priors, ests, pl):
        sscmks = syscore.ssconstants(mks=True)
        if 'sma' not in priors[pl].keys():
            return 'missing semi-major axis'
        elif priors[pl]['sma']=='':
            return 'missing semi-major axis'
        else:
            eqtemp = priors['T*']*np.sqrt(priors['R*']*sscmks['Rsun/AU']/
                                          (2.*priors[pl]['sma']))
        return eqtemp

class HEstimator(PlEstimator):
    def __init__(self):
        PlEstimator.__init__(self, name='H', descr='Atmospheric scale height (CBE)',
                             units='km',ref='Fortney metallicity')

    def run(self, priors, ests, pl):
        sscmks = syscore.ssconstants(cgs=True)

        if 'sma' not in priors[pl].keys():
            return 'missing semi-major axis'
        elif priors[pl]['sma']=='':
            return 'missing semi-major axis'
        else:
            eqtemp = priors['T*']*np.sqrt(priors['R*']*sscmks['Rsun/AU']/
                                          (2.*priors[pl]['sma']))

        if priors[pl]['mass'] == '':  # abort for targets without mass estimates
            return None
        if priors[pl]['logg'] == '':
            g = sscmks['G'] * priors[pl]['mass']*sscmks['Mjup'] / \
                (priors[pl]['rp']*sscmks['Rjup'])**2
        else:
            g = 10.**priors[pl]['logg']

        mmw = pl_mmw(priors, ests, pl)

        H = sscmks['Rgas'] * eqtemp / mmw / g

        # convert cm to km
        return H / 1.e5

class HmaxEstimator(PlEstimator):
    def __init__(self):
        PlEstimator.__init__(self, name='H_max', descr='Atmospheric scale height (max)',
                             units='km',ref='solar composition')

    def run(self, priors, ests, pl):
        sscmks = syscore.ssconstants(cgs=True)

        if 'sma' not in priors[pl].keys():
            return 'missing semi-major axis'
        elif priors[pl]['sma']=='':
            return 'missing semi-major axis'
        else:
            eqtemp = priors['T*']*np.sqrt(priors['R*']*sscmks['Rsun/AU']/
                                          (2.*priors[pl]['sma']))

        if priors[pl]['mass'] == '':  # abort for targets without mass estimates
            return None
        if priors[pl]['logg'] == '':
            g = sscmks['G'] * priors[pl]['mass']*sscmks['Mjup'] / \
                (priors[pl]['rp']*sscmks['Rjup'])**2
        else:
            g = 10.**priors[pl]['logg']

        mmw = pl_mmwmin(priors, ests, pl)

        H = sscmks['Rgas'] * eqtemp / mmw / g

        # convert cm to km
        return H / 1.e5

def pl_metals(priors, _ests, pl):
    if priors[pl]['mass'] == '':  # abort for targets without mass estimates
        # print('no mass for this planet')
        return None

    # 318 Earth masses per Jupiter mass
    # pivot point (where metallicity is max value) is at 10 Earth masses
    metallicity = 3 - np.log10(318.*priors[pl]['mass'])
    if metallicity > 2: metallicity = 2
    return metallicity

def pl_mmw(priors, _ests, pl):

    metallicity = pl_metals(priors, _ests, pl)
    if metallicity is None:
        return None

    (a,b,c) = (2.274,0.02671737,2.195719)
    mmw = a + b*np.exp(c*metallicity)
    return mmw

def pl_mmwmin(_priors, _ests, _pl):

    return 2.274

def pl_ZFOM(priors, _ests, pl):
    if priors[pl]['mass'] == '':  # abort for targets without mass estimates
        # print('no mass for this planet')
        return None

    sscmks = syscore.ssconstants(cgs=True)

    if 'sma' not in priors[pl].keys():
        return 'missing semi-major axis'
    elif priors[pl]['sma']=='':
        return 'missing semi-major axis'
    else:
        eqtemp = priors['T*']*np.sqrt(priors['R*']*sscmks['Rsun/AU']/
                                      (2.*priors[pl]['sma']))

    # g = sscmks['G'] * priors[pl]['mass']*sscmks['Mjup'] / \
    #             (priors[pl]['rp']*sscmks['Rjup'])**2
    g = 10.**priors[pl]['logg']

    # metallicity = pl_metals(priors, _ests, pl)

    mmw = pl_mmw(priors, _ests, pl)

    H = sscmks['Rgas'] * eqtemp / mmw / g

    if 'Hmag' in priors.keys():
        if priors['Hmag']=='':
            # print('missing Hmag1',priors['Hmag'])
            return 'blank Hmag'
        else:
            Hmag = priors['Hmag']
    else:
        # print('missing Hmag2')
        return 'no Hmag'

    # calculate Zellem Figure-of-Merit (Zellem 2017, Eq.10)
    ZFOM = 2 * H * priors[pl]['rp'] / priors['R*']**2 / 10**(Hmag/5) \
           *sscmks['Rjup'] /sscmks['Rsun']**2
    # print('Mplanet,metals,mmw,H,ZFOM', pl,',',
    #      priors[pl]['mass'],',',metallicity,',',mmw,',',H,',',ZFOM*1.e10)
    # normalization to make it easier to read
    return ZFOM * 1.e10

def pl_ZFOMmax(priors, _ests, pl):
    if priors[pl]['mass'] == '':  # abort for targets without mass estimates
        # print('no mass for this planet')
        return None

    sscmks = syscore.ssconstants(cgs=True)

    if 'sma' not in priors[pl].keys():
        return 'missing semi-major axis'
    elif priors[pl]['sma']=='':
        return 'missing semi-major axis'
    else:
        eqtemp = priors['T*']*np.sqrt(priors['R*']*sscmks['Rsun/AU']/
                                      (2.*priors[pl]['sma']))

    # g = sscmks['G'] * priors[pl]['mass']*sscmks['Mjup'] / \
    #             (priors[pl]['rp']*sscmks['Rjup'])**2
    g = 10.**priors[pl]['logg']

    # H/He-dominant atmosphere (for minimum mmw case)
    mmw = pl_mmwmin(priors, _ests, pl)

    H = sscmks['Rgas'] * eqtemp / mmw / g

    if 'Hmag' in priors.keys():
        Hmag = priors['Hmag']
    else:
        # print('Hmag missing for ZFOM2')
        return 'no Hmag'

    # calculate Zellem Figure-of-Merit (Zellem 2017, Eq.10)
    ZFOM = 2 * H * priors[pl]['rp'] / priors['R*']**2 / 10**(Hmag/5) \
           *sscmks['Rjup'] /sscmks['Rsun']**2
    # normalization to make it easier to read
    return ZFOM * 1.e10

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
    if 'L*' in priors.keys() and priors['L*']!='':
        est = 10.**priors['L*']
        if priors['R*']!='' and priors['T*']!='':
            # print('L* vs T*R* consistency?',
            #      est/(priors['R*']**2*(priors['T*']/Tsun)**4),
            #      est, priors['R*']**2*(priors['T*']/Tsun)**4)
            pass
    else:
        if priors['R*']=='':
            est = ' R_star missing'
        elif priors['T*']=='':
            est = 'T_star missing'
        else:
            est = priors['R*']**2*(priors['T*']/Tsun)**4
    return est

def st_spTyp(priors, _ests):
    if 'spTyp' in priors.keys():
        est = priors['spTyp']
    else:
        est = 'N/A'
    return est

def pl_insolation(priors, ests, pl):
    if 'sma' not in priors[pl].keys():
        return 'missing semi-major axis'
    elif priors[pl]['sma']=='':
        return 'missing semi-major axis'
    else:
        insolation = ests['luminosity']*(priors[pl]['sma']**-2)

    # apply eccentricity correction
    insolation *= math.sqrt(1/(1-priors[pl]['ecc']**2))

    return insolation

def st_rotationPeriod(priors, _ests):

    if 'AGE*' in priors.keys():
        Age = priors['AGE*']
    else:
        Age = 5
        # *** important assumption - arbitrary guess for the age if it's blank ***
    if Age=='':
        Age = 5  # Gyr

    if priors['T*'] < 3500:
        # rotation period for M2.5-6.0 V stars
        # applies for GJ 1132, GJ 3053, K2-18, GJ 1214
        if Age < 0.012+0.061:
            Prot = 1
        else:
            Prot = (Age - 0.012) / 0.061
    elif priors['T*'] < 4000:
        # rotation period for M0 V stars
        # applies for Kepler 138, K2-3
        if Age < 0.365+0.019:
            Prot = 1
        else:
            Prot = ((Age - 0.365) / 0.019)**(1/1.457)
    else:
        # rotation period for K-type stars
        # applies for 55 Cnc, GJ 9827, GJ 97658
        Prot = 25 * (Age / 4.6)**0.5

    return Prot

def st_coronalTemp(priors, _ests):

    Prot = st_rotationPeriod(priors, _ests)

    # this formula needs work; it is not continuous at the 18.9-day break
    if Prot > 18.9:
        Tcorona = 1.5e6 * (27 / Prot)**1.2
    else:
        Tcorona = 1.98e6 * (27 / Prot)**0.37

    return Tcorona

def pl_windVelocity(priors, ests, pl):
    sscmks = syscore.ssconstants(cgs=True)

    if 'M*' not in priors.keys():
        if priors['LOGG*'] == '':
            return None
        else:
            priors['M*'] = 10.**priors['LOGG*'] / sscmks['G'] /sscmks['Msun'] \
                * (priors['R*']*sscmks['Rsun'])**2
    if priors['M*'] == '':
        if priors['LOGG*'] == '':
            # print('no stellar mass')
            return None
        else:
            priors['M*'] = 10.**priors['LOGG*'] / sscmks['G'] /sscmks['Msun'] \
                * (priors['R*']*sscmks['Rsun'])**2

    mmw = 0.67

    Tcorona = st_coronalTemp(priors, ests)

    v_crit = np.sqrt(sscmks['Rgas'] * Tcorona / mmw)

    r_crit = sscmks['G'] * sscmks['Msun'] * priors['M*'] / 2 / v_crit**2

    # Parker solar wind solution: v^2 - ln(v^2) = 4(1/r + ln(r)) - 3
    #   comes from  dv/dr(v - 1/v) = 2(1/r - 1/r^2)
    # (r,v are normalized by r_crit,v_crit)
    def parkerSolution(v,r):
        return v**2 - np.log(v**2) - 4/r - 4*np.log(r) + 3

    if 'sma' not in priors[pl].keys():
        return 'missing semi-major axis'
    elif priors[pl]['sma']=='':
        return 'missing semi-major axis'
    else:
        r = priors[pl]['sma']*sscmks['AU'] / r_crit
    # (the second field is an initial guess for v)
    v = scipy.optimize.root(parkerSolution, r, args=r)['x'][0]

    # convert cm/s to km/s
    windVelocity = v * v_crit / 1.e5
    return windVelocity

def pl_windDensity(priors, ests, pl):
    sscmks = syscore.ssconstants(cgs=True)

    if 'sma' not in priors[pl].keys():
        return 'missing semi-major axis'
    elif priors[pl]['sma']=='':
        return 'missing semi-major axis'
    else:
        r = priors[pl]['sma'] * sscmks['AU']/sscmks['Rsun']   # RSun units

    numberDensity = 3.3e5/r**2 + 4.1e6/r**4 + 8.0e7/r**6

    mmw = 0.67
    m_H = 1.6735e-24
    windDensity = numberDensity * m_H*mmw

    # adjustment based on rotation.  fast spin fives higher density
    Prot = st_rotationPeriod(priors, ests)
    windDensity *= (18.9/Prot)**0.6

    return windDensity

def pl_windMassLoss(priors, ests, pl):
    sscmks = syscore.ssconstants(cgs=True)

    entrainmentEfficiency = 0.3

    windVelocity = pl_windVelocity(priors, ests, pl)
    # convert velocity from km/s to cm/s
    if isinstance(windVelocity, str):
        return 'undefined wind velocity'
    else:
        windVelocity *= 1.e5
        windDensity = pl_windDensity(priors, ests, pl)

        massLossRate = 2 * np.pi * entrainmentEfficiency * \
            (priors[pl]['rp'] * sscmks['Rjup'])**2 * \
            windDensity * windVelocity

    # convert to Jupiter masses per gigayear
    return massLossRate / sscmks['Mjup'] * 3.16e7*1.e9

# X-ray flux based on stellar age and spectral type (T_* actually)
def Lxray(priors):

    st_temp = priors['T*']
    if st_temp < 4000:
        pass
    elif st_temp < 5000:
        pass
    else:
        pass
    L_X = 10
    return L_X

def pl_evapMassLoss(priors, _ests, pl):
    sscmks = syscore.ssconstants(cgs=True)

    Age = priors['AGE*']
    if Age=='':
        # *** important assumption - arbitrary guess for the age ***
        Age = 5  # Gyr

    heatingEfficiency = 0.1

    # X-ray flux based on stellar age and spectral type (T_* actually)
    L_X = Lxray(priors)

    # assumed relationship between X-ray and EUV fluxes
    L_EUV = 425 * (L_X/sscmks['Lsun'])**0.58 * sscmks['Lsun']

    if 'sma' not in priors[pl].keys():
        return 'missing semi-major axis'
    elif priors[pl]['sma']=='':
        return 'missing semi-major axis'
    else:
        # F_X = L_X / (4 * np.pi * (priors[pl]['sma']*sscmks['AU'])**2)
        F_EUV = L_EUV / (4 * np.pi * (priors[pl]['sma']*sscmks['AU'])**2)

    # assume the based of the photoevap flow is just the planet radius
    R_base = priors[pl]['rp'] * sscmks['Rjup']

    massLossRate = np.pi * heatingEfficiency * F_EUV * R_base**3 / \
                   (sscmks['G'] * priors[pl]['mass'] * sscmks['Mjup'])

    # convert to Jupiter masses per gigayear
    return massLossRate / sscmks['Mjup'] * 3.16e7*1.e9

def st_COratio(priors, _ests):
    if 'FEH*' in priors.keys() and priors['FEH*'] != '':
        # this is equation 2 from Nissen 2013
        est = -0.002 + 0.22 * priors['FEH*']
    else:
        est = 'N/A'
    return est
