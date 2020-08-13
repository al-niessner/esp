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
    # defined as function so it can reference functions defined
    # later in file
    st_estimators = [
        ancestor.StellarTypeEstimator(),
        StEstimator(name='luminosity', units='L sun', descr='Stellar luminosity',
                    method=ancestor.st_luminosity)
        ]
    pl_estimators = [
        ancestor.TeqEstimator(),
        PlEstimator(name='density', units='g/cm^3', descr='Density of planet',
                    method=ancestor.pl_density),
        PlEstimator(name='insolation', units='Earth insolation',
                    descr='Incident stellar flux', method=ancestor.pl_insolation,
                    ref='Weiss & Marcy 2014')
        ]
    return st_estimators, pl_estimators

# ------------- ------------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    '''
    N. HUBER-FEELY: Tests for empty SV shell
    '''
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring

def estimateversion():
    return dawgie.VERSION(1,0,1)

# ----------------- --------------------------------------------------
# -- ESTIMATOR EVALUATOR ---------------------------------------------
def estimate(fin, out):
    st_estimators, pl_estimators = getestimators()

    planets = fin['priors']['planets']

    # get estimates from each stellar estimator
    for est in st_estimators:
        raw_estimate = est.run(fin['priors'], out['data'])
        if raw_estimate is None:  # flag for failed or uncomputed estimator
            continue  # prevent estimator addition
        elif isinstance(raw_estimate, dict):
            out['data'][est.name()] = raw_estimate['val']
            if 'uperr' in raw_estimate:
                out['data'][est.name()+'_uperr'] = raw_estimate['uperr']
            if 'lowerr' in raw_estimate:
                out['data'][est.name()+'_lowerr'] = raw_estimate['lowerr']
        else:  # default to add
            out['data'][est.name()] = raw_estimate
        if est.descr():
            out['data'][est.name() + '_descr'] = est.descr()
        if est.units():
            out['data'][est.name() + '_units'] = est.units()
        if est.ref():
            out['data'][est.name() + '_ref'] = est.ref()

    # get estimates from each planetary estimator
    out['data']['planets'] = planets
    for pl in planets:
        out['data'][pl] = {}
        for est in pl_estimators:
            raw_estimate = est.run(fin['priors'], out['data'], pl)
            if raw_estimate is None:
                continue  # prevent estimator addition
            elif isinstance(raw_estimate, dict):
                out['data'][pl][est.name()] = raw_estimate['val']
                if 'uperr' in raw_estimate:
                    out['data'][pl][est.name()+'_uperr'] = raw_estimate['uperr']
                if 'lowerr' in raw_estimate:
                    out['data'][pl][est.name()+'_lowerr'] = raw_estimate['lowerr']
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
