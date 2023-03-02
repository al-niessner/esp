'''ancillary core ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

# import estimators
from excalibur.ancillary.estimators import StEstimator, PlEstimator
import excalibur.ancillary.estimators as ancestor

import logging; log = logging.getLogger(__name__)

SV_EXTS = ['_descr', '_units', '_uperr', '_lowerr', '_ref']

# ------------- ------------------------------------------------------
# -- ESTIMATOR DEFINITIONS -- ----------------------------------------
# View README.md for instructions on defining and adding an estimator.
# NOTE: Estimators are evaluated in the order they are listed..
def getestimators():
    '''getestimators ds'''
    # defined as function so it can reference functions defined
    # later in file
    st_estimators = [
        StEstimator(name='luminosity', units='L_sun', descr='Stellar luminosity',
                    method=ancestor.st_luminosity,
                    ref='from R_star & T_star'),
        ancestor.StellarTypeEstimator(),
        StEstimator(name='spTyp', units='', descr='Spectral type',
                    method=ancestor.st_spTyp,
                    ref='Exoplanet Archive'),
        StEstimator(name='CO*', units='log', descr='Stellar [C/O]',
                    method=ancestor.st_COratio,
                    ref='Nissen 2013; Eq.2'),
        StEstimator(name='P_rot', units='days', descr='Stellar rotation period',
                    method=ancestor.st_rotationPeriod, ref='Engle & Guinan 2018'),
        StEstimator(name='T_corona', units='K', descr='Stellar corona temperature',
                    method=ancestor.st_coronalTemp, ref='')
        ]
    pl_estimators = [
        PlEstimator(name='density', units='g/cm^3', descr='Density of planet',
                    method=ancestor.pl_density, ref='Mr. Fisher'),
        PlEstimator(name='insolation', units='Earth insolation',
                    descr='Incident stellar flux', method=ancestor.pl_insolation,
                    ref='Weiss & Marcy 2014'),
        ancestor.TeqEstimator(),
        PlEstimator(name='metallicity', units='logarithmic',
                    descr='metallicity', method=ancestor.pl_metals,
                    ref='Fortney et al 2013'),
        PlEstimator(name='mmw', units='AMU',
                    descr='mean molecular weight (CBE)', method=ancestor.pl_mmw,
                    ref='CEA (T=1000K;C/O=solar)'),
        PlEstimator(name='mmw_min', units='AMU',
                    descr='mean molecular weight (min)', method=ancestor.pl_mmwmin,
                    ref='CEA (T=1000K;C/O=solar)'),
        ancestor.HEstimator(),
        ancestor.HmaxEstimator(),
        PlEstimator(name='modulation', units='dimensionless',
                    descr='spectral modulation (CBE)', method=ancestor.pl_modulation,
                    ref='Zellem et al 2017'),
        PlEstimator(name='modulation_max', units='dimensionless',
                    descr='spectral modulation (max)', method=ancestor.pl_modulationmax,
                    ref='Zellem et al 2017'),
        PlEstimator(name='ZFOM', units='ppm',
                    descr='Zellem Figure-of-Merit (CBE)', method=ancestor.pl_ZFOM,
                    ref='Zellem et al 2017'),
        PlEstimator(name='ZFOM_max', units='ppm',
                    descr='Zellem Figure-of-Merit (max)', method=ancestor.pl_ZFOMmax,
                    ref='Zellem et al 2017'),
        PlEstimator(name='v_wind', units='km/s',
                    descr='Stellar wind velocity', method=ancestor.pl_windVelocity,
                    ref='Parker solution'),
        PlEstimator(name='rho_wind', units='g/cm^3',
                    descr='Stellar wind density', method=ancestor.pl_windDensity,
                    ref='Leblanc et al 1998'),
        PlEstimator(name='M_loss_rate_wind', units='M_Jup/Gyr',
                    descr='Wind-driven mass loss rate', method=ancestor.pl_windMassLoss,
                    ref='Canto et al 1991'),
        PlEstimator(name='M_loss_rate_evap', units='M_Jup/Gyr',
                    descr='EUV-driven mass loss rate', method=ancestor.pl_evapMassLoss,
                    ref='Estrela et al 2020'),
        PlEstimator(name='Beta_rad', units='dimensionless',
                    descr='radiation pressure/gravity',
                    method=ancestor.pl_beta_rad,
                    ref='Owens et al 2023')
    ]

    return st_estimators, pl_estimators

# ------------- ------------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    '''N. HUBER-FEELY: Tests for empty SV shell'''
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring

def estimateversion():
    '''estimateversion ds'''
    # return dawgie.VERSION(2,0,0)
    return dawgie.VERSION(2,1,0)  # checks for blank values; betaRad included
# ----------------- --------------------------------------------------
# -- ESTIMATOR EVALUATOR ---------------------------------------------
def estimate(fin, out):
    '''estimate ds'''
    st_estimators, pl_estimators = getestimators()

    planets = fin['priors']['planets']

    # get estimates from each stellar estimator
    for est in st_estimators:
        raw_estimate = est.run(fin['priors'], out['data'])
        if raw_estimate is None:  # flag for failed or uncomputed estimator
            continue  # prevent estimator addition
        if isinstance(raw_estimate, dict):
            out['data'][est.name()] = raw_estimate['val']
            if 'uperr' in raw_estimate:
                out['data'][est.name()+'_uperr'] = raw_estimate['uperr']
            if 'lowerr' in raw_estimate:
                out['data'][est.name()+'_lowerr'] = raw_estimate['lowerr']
                pass
            pass
        else:  # default to add
            out['data'][est.name()] = raw_estimate
        if est.descr():
            out['data'][est.name() + '_descr'] = est.descr()
        if est.units():
            out['data'][est.name() + '_units'] = est.units()
        if est.ref(): out['data'][est.name() + '_ref'] = est.ref()
        pass

    # get estimates from each planetary estimator
    out['data']['planets'] = planets
    for pl in planets:
        out['data'][pl] = {}
        for est in pl_estimators:
            raw_estimate = est.run(fin['priors'], out['data'], pl)
            if raw_estimate is None:
                continue  # prevent estimator addition
            if isinstance(raw_estimate, dict):
                out['data'][pl][est.name()] = raw_estimate['val']
                if 'uperr' in raw_estimate:
                    out['data'][pl][est.name()+'_uperr'] = raw_estimate['uperr']
                    pass
                if 'lowerr' in raw_estimate:
                    out['data'][pl][est.name()+'_lowerr'] = raw_estimate['lowerr']
                    pass
                pass
            else:  # default to add
                out['data'][pl][est.name()] = raw_estimate
            if est.descr():
                out['data'][pl][est.name() + '_descr'] = est.descr()
            if est.units():
                out['data'][pl][est.name() + '_units'] = est.units()
            if est.ref():
                out['data'][pl][est.name() + '_ref'] = est.ref()
    out['STATUS'].append(True)  # mark success
    return True
