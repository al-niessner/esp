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
        print('NOTE: Ariel SNR file missing; failing to simulate spectrum',target)
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
    Load in the output from Ariel-Rad - uncertainty as a function of wavelength

    This is different from the previous version above, in that there is a single file now,
    not a separate file for each target.

    H_mag brightness is already taken into account, but
     number of observed transits is not taken into account.
    '''

    # noise_model_dir = '/proj/data/ariel/'
    noise_model_dir = excalibur.context['data_dir']+'/ariel/'

    # noise_model_filename = 'arielRad_02aug2023.h5'
    # noise_model_filename = 'arielRad_07aug2023_mmwFixed.h5'
    noise_model_filename = 'arielRad_07aug2023_mmwExcal.h5'

    # all targets should have a file, but if not, return None
    if not os.path.isfile(noise_model_dir + noise_model_filename):
        print('NOTE: Ariel SNR file missing; failing to simulate spectrum',target)
        ariel_instrument = None

    else:
        with h5py.File(noise_model_dir + noise_model_filename, 'r') as arielRad_results:

            # use the 'SNR' key to determine the required number of transits
            # print('results keys',arielRad_results.keys())

            targets = arielRad_results['errorbars_evaluated'].keys()

            if target not in targets:
                print('NOTE: target not in Ariel SNR file; failing to simulate spectrum',target)
                ariel_instrument = None

            else:
                noiseSpectrum = read_table_hdf5(arielRad_results['errorbars_evaluated'][target],
                                                path='table')

                ariel_instrument = {
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
                        log.warning('--< ADJUSTING ARIEL WAVELENGTH GRID: %s wave=%s >--',
                                    target,ariel_instrument['wavelength'][iwave])
                        ariel_instrument['wavehigh'][iwave] = ariel_instrument['wavelow'][iwave+1]*0.99999

    return ariel_instrument
