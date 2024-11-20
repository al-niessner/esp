'''ariel ariel ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import os
import numpy as np

import excalibur

import h5py
from astropy.io.misc.hdf5 import read_table_hdf5

# ---------------------------- ---------------------------------------
def load_ariel_instrument(target, tier):
    '''
    Load in the output from ArielRad - uncertainty as a function of wavelength

    This is different from the previous version above, in that there is a single file now,
    not a separate file for each target.

    H_mag brightness is already taken into account, but
     number of observed transits is not taken into account.
    '''

    noise_model_dir = excalibur.context['data_dir']+'/ariel/'

    # noise_model_filenames = ['arielRad_02aug2023.h5']
    # noise_model_filenames = ['arielRad_07aug2023_mmwFixed.h5']
    # noise_model_filenames = ['arielRad_07aug2023_mmwExcal.h5']
    # noise_model_filenames = ['arielRad_3mar2024_MCStargetsAdded.h5']
    # noise_model_filenames = ['arielRad_14aug2024_mmwFixed.h5']
    # 'mmwFixed' means mmw=2.3
    # 'mmwThorngren' means mmw comes from our mass-metallicity relation
    # noise_model_filenames = ['arielRad_14aug2024_mmwThorngren.h5']

    # uh oh, arielrad is crashing, with too many files open
    #  have to split the file into two parts
    # noise_model_filenames = ['arielRad_29oct2024_mmwFixed-part1.h5',
    #                         'arielRad_29oct2024_mmwFixed-part2.h5']
    noise_model_filenames = ['arielRad_29oct2024_mmwThorngren-part1.h5',
                             'arielRad_29oct2024_mmwThorngren-part2.h5']

    ariel_instrument = None
    for noise_model_filename in noise_model_filenames:
        # if the file is missing, note the error and move on
        if not os.path.isfile(noise_model_dir + noise_model_filename):
            log.warning('--< ARIELSIM ERROR: SNR file missing for %s  %s >--',
                        target,noise_model_filename)
        else:
            with h5py.File(noise_model_dir + noise_model_filename, 'r') as arielRad_results:
                # print('ArielRad keys',arielRad_results.keys())

                targets = arielRad_results['errorbars_evaluated'].keys()

                if target not in targets:
                    # print('NOTE: target not in Ariel SNR file; failing to simulate spectrum',target)
                    # log.warning('--< ARIELSIM: target not in SNR file; failing to simulate spectrum  %s >--',target)
                    #  ariel_instrument = None
                    # print('  not in this part:',noise_model_filename)
                    pass
                else:
                    # print('  found it in this part:',noise_model_filename)

                    # use 'SNR' table to determine the required number of transits
                    # SNR is not an hdf5 table.  just access it like a normal dict
                    SNRtable = arielRad_results['SNR']['SNRTab_to_group']
                    # print('SNR options',SNRtable.keys())

                    if tier==1:
                        nTransitsCH0 = SNRtable['AIRS-CH0-T1-nTransits']['value'][()]
                        nTransitsCH1 = SNRtable['AIRS-CH1-T1-nTransits']['value'][()]
                    elif tier==3:
                        nTransitsCH0 = SNRtable['AIRS-CH0-T3-nTransits']['value'][()]
                        nTransitsCH1 = SNRtable['AIRS-CH1-T3-nTransits']['value'][()]
                    else:
                        if tier!=2: log.warning('--< Unknown Ariel Tier!: %s >--',tier)
                        nTransitsCH0 = SNRtable['AIRS-CH0-T2-nTransits']['value'][()]
                        nTransitsCH1 = SNRtable['AIRS-CH1-T2-nTransits']['value'][()]
                    planetNames = SNRtable['planetName']['value'][()]
                    planetNames = np.array([name.decode('UTF-8') for name in planetNames])

                    thisplanetIndex = np.where(target==planetNames)
                    if len(thisplanetIndex)==0:
                        log.warning('--< ArielRad #-of-visits missing: %s >--',target)
                        nVisits = 666
                    elif len(thisplanetIndex)>1:
                        log.warning('--< ArielRad has multiple target matches?!: %s >--',target)
                        nVisits = 666
                    elif np.isfinite(nTransitsCH0[thisplanetIndex]) and \
                         np.isfinite(nTransitsCH1[thisplanetIndex]):
                        nVisits = np.min([nTransitsCH0[thisplanetIndex],
                                          nTransitsCH1[thisplanetIndex]])
                        nVisits = int(np.ceil(nVisits))
                    else:
                        log.warning('--< ArielRad has non-finite # of visits: %s >--',target)
                        nVisits = 666
                    log.warning('--< ArielRad/Tier-%s requires %s visits for %s >--',
                                str(tier),str(nVisits),target)

                    noiseSpectrum = read_table_hdf5(arielRad_results['errorbars_evaluated'][target],
                                                    path='table')
                    # print('noiseSpectrum options',noiseSpectrum.keys())
                    ariel_instrument = {
                        'nVisits':nVisits,
                        'wavelength':noiseSpectrum['Wavelength'].value,
                        'wavelow':noiseSpectrum['LeftBinEdge'].value,
                        'wavehigh':noiseSpectrum['RightBinEdge'].value,
                        # 'wavebinsize':(noiseSpectrum['RightBinEdge'].value -
                        #                noiseSpectrum['LeftBinEdge'].value),
                        'noise':noiseSpectrum['NoiseOnTransitFloor'].value,
                    }

                    for iwave in range(len(ariel_instrument['wavelow'])-1):
                        # multiply by number slightly above 1 to deal with numerical precision error
                        if ariel_instrument['wavehigh'][iwave] > \
                           ariel_instrument['wavelow'][iwave+1]*1.00001:
                            # print('spectral channels overlap!!',iwave,
                            #       ariel_instrument['wavehigh'][iwave],
                            #       ariel_instrument['wavelow'][iwave+1])
                            # log.warning('--< ARIELSIM adjusting wavelength grid: %s wave=%s >--',
                            #             target,ariel_instrument['wavelength'][iwave])
                            ariel_instrument['wavehigh'][iwave] = ariel_instrument['wavelow'][iwave+1]*0.99999

    if not ariel_instrument:
        log.warning('--< ARIELSIM: target not in SNR files; failing to simulate spectrum  for %s >--',target)

    return ariel_instrument
