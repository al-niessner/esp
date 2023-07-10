'''ancillary core ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur

# import estimators
from excalibur.ancillary.estimators import StEstimator, PlEstimator
import excalibur.ancillary.estimators as ancestor

import os
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

# ---------------------------- ---------------------------------------
def savesv(aspects, targetlists):
    '''
    save the results as a csv file in /proj/data/spreadsheets
    '''

    svname = 'ancillary.estimate.parameters'

    # RID = int(os.environ.get('RUNID', None))
    RID = os.environ.get('RUNID', None)
    # print('RID',RID)
    if RID:
        RID = f'{int(RID):03}'
    else:
        RID = '666'
    # print('RID',RID)

    # directory where the results are saved
    saveDir = excalibur.context['data_dir'] + '/spreadsheets/RID'+RID + '/'
    #    '/spreadsheets/RID' + f"{RID:03i}" + '/'
    #    '/spreadsheets/RID' + str('%03i' %RID) + '/'
    # print('saveDir:',saveDir)
    if not os.path.exists(saveDir): os.mkdir(saveDir)

    # file name where the results are saved
    # outfileName = svname.replace('.','_') + '_RID' + str('%03i' %RID) + '.csv'
    # outfileName = svname.replace('.','_') + '_RID' + f"{RID:03i}" + '.csv'
    outfileName = svname.replace('.','_') + '_RID'+RID + '.csv'
    # outfile = open(saveDir + outfileName,'w',encoding='ascii')
    with open(saveDir + outfileName,'w',encoding='ascii') as outfile:

        # (the list of extensions is hardcoded at the top of this file)
        exts = SV_EXTS
        # print('extensions:',exts)

        # use 55 Cnc as an example, to make the list of parameters
        ancillary_data = aspects['55 Cnc'][svname]
        st_keys = [key for key in ancillary_data['data'].keys()
                   if ((not key == 'planets') and
                       (key not in ancillary_data['data']['planets']) and
                       (not any(ext in key for ext in SV_EXTS)))]
        # print('st_keys',st_keys)
        pl_keys = [i for i in ancillary_data['data']['e'].keys()
                   if not any(ext in i for ext in SV_EXTS)]
        # print('pl_keys',pl_keys)

        # don't print out the full description of the parameter; bulky and repetitive
        exts.remove('_descr')
        # these uncertainty fields aren't actually calculated
        exts.remove('_lowerr')
        exts.remove('_uperr')

        # write the header row
        outfile.write('star,planet,')
        for key in st_keys:
            # print('key',key)
            outfile.write(key + ',')
            for ext in exts:
                # stellar_type doesn't have units.  maybe add it in estimators.py
                if key+ext != 'stellar_type_units' and \
                   key+ext != 'T_corona_ref' and \
                       key+ext != 'spTyp_units':
                    outfile.write(key+ext + ',')
        for key in pl_keys:
            outfile.write(key + ',')
            for ext in exts:
                outfile.write(key+ext + ',')
        outfile.write('\n')

        # loop through each target, with one row per planet
        for trgt in targetlists['active']:
            ancillary_data = aspects[trgt][svname]

            for planet_letter in ancillary_data['data']['planets']:
                outfile.write(trgt + ',')
                outfile.write(planet_letter + ',')

                for key in st_keys:
                    outfile.write(str(ancillary_data['data'][key]) + ',')
                    for ext in exts:
                        if key+ext != 'stellar_type_units' and \
                           key+ext != 'T_corona_ref' and \
                               key+ext != 'spTyp_units':
                            outfile.write(str(ancillary_data['data'][key+ext]).replace(',',';') + ',')

                for key in pl_keys:
                    outfile.write(str(ancillary_data['data'][planet_letter][key]) + ',')
                    for ext in exts:
                        outfile.write(str(ancillary_data['data'][planet_letter][key+ext]).replace(',',';') + ',')

                outfile.write('\n')
    # outfile.close()

    return
