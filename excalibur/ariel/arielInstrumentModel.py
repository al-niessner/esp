'''ariel ariel ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import os
import numpy as np

import excalibur

import h5py
from astropy.io.misc.hdf5 import read_table_hdf5

# ---------------------------- ---------------------------------------
def load_ariel_instrument_oldmethod(target):
    '''
    Load in a file that gives SNR as a function of wavelength

    H_mag brightness is already taken into account, but
     number of observed transits is not taken into account.
    Also, need to apply multiplicative factor and noise floor, as desired
    '''

    # there is a separate file for each planet
    # noise_model_filename='/proj/data/ariel/SNRfiles/'+target+'_SNR_ARIEL.txt'
    noise_model_filename=excalibur.context['data_dir'] + '/ariel/SNRfiles/'+target+'_SNR_ARIEL.txt'

    # all targets should have a file, but if not, return None
    if not os.path.isfile(noise_model_filename):
        # print('NOTE: Ariel SNR file missing; failing to simulate spectrum',target)
        log.warning('--< ARIELSIM ERROR: SNR file missing >--')
        ariel_instrument = None
    else:
        with open(noise_model_filename,'r',encoding='ascii') as file:
            wavelength = []
            wavebinsize = []
            SNR = []
            header = []
            for line in file:
                cols = line.split(',')
                if not header:
                    header = cols
                    # print('file header',header)
                else:
                    wavelength.append(float(cols[0]))
                    wavebinsize.append(float(cols[1]))
                    SNR.append(float(cols[2]))

        ariel_instrument = {
            'wavelength':np.array(wavelength),
            'wavebinsize':np.array(wavebinsize),
            'wavelow':np.array(wavelength) - np.array(wavebinsize)/2.,
            'wavehigh':np.array(wavelength) + np.array(wavebinsize)/2.,
            'noise':1./np.array(SNR)
        }

    return ariel_instrument
# ---------------------------- ---------------------------------------
def load_ariel_instrument(target):
    '''
    Load in the output from ArielRad - uncertainty as a function of wavelength

    This is different from the previous version above, in that there is a single file now,
    not a separate file for each target.

    H_mag brightness is already taken into account, but
     number of observed transits is not taken into account.
    '''

    # noise_model_dir = '/proj/data/ariel/'
    noise_model_dir = excalibur.context['data_dir']+'/ariel/'

    # noise_model_filename = 'arielRad_02aug2023.h5'
    # noise_model_filename = 'arielRad_07aug2023_mmwFixed.h5'
    # noise_model_filename = 'arielRad_07aug2023_mmwExcal.h5'
    # noise_model_filename = 'arielRad_3mar2024_MCStargetsAdded.h5'
    noise_model_filename = 'arielRad_14aug2024_mmwFixed.h5'
    # above has mmw=2.3; below uses our mass-metallicity relation
    noise_model_filename = 'arielRad_14aug2024_mmwThorngren.h5'

    # if the file is missing, note the error and move on
    if not os.path.isfile(noise_model_dir + noise_model_filename):
        log.warning('--< ARIELSIM ERROR: SNR file missing for %s >--',target)
        ariel_instrument = None

    else:
        with h5py.File(noise_model_dir + noise_model_filename, 'r') as arielRad_results:

            # print('ArielRad keys',arielRad_results.keys())

            targets = arielRad_results['errorbars_evaluated'].keys()

            if target not in targets:
                # print('NOTE: target not in Ariel SNR file; failing to simulate spectrum',target)
                log.warning('--< ARIELSIM: target not in SNR file; failing to simulate spectrum  %s >--',target)
                ariel_instrument = None

            else:
                noiseSpectrum = read_table_hdf5(arielRad_results['errorbars_evaluated'][target],
                                                path='table')
                # print('noiseSpectrum options',noiseSpectrum.keys())

                # use 'SNR' table to determine the required number of transits
                # SNR is not an hdf5 table.  just access it like a normal dict
                SNRtable = arielRad_results['SNR']['SNRTab_to_group']
                # print('SNR options',SNRtable.keys())

                # use the ** Tier-2 ** number of visits
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
                log.warning('--< ArielRad sets the # of visits (Tier2): %s %s >--',
                            str(nVisits),target)

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
                        log.warning('--< ARIELSIM adjusting wavelength grid: %s wave=%s >--',
                                    target,ariel_instrument['wavelength'][iwave])
                        ariel_instrument['wavehigh'][iwave] = ariel_instrument['wavelow'][iwave+1]*0.99999

    return ariel_instrument
