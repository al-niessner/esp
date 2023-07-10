'''ariel ariel ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import numpy as np

# ---------------------------- ---------------------------------------
# def load_ariel_instrument(noise_model_filename='/proj/data/ariel/HD_189733_b_SNR_ARIEL.txt'):
def load_ariel_instrument(target):
    '''
    Load in a file that gives SNR as a function of wavelength

    H_mag brightness is already taken into account, but
     number of observed transits is not taken into account.
    Also, need to apply multiplicative factor and noise floor, as desired
    '''

    # there is a separate file for each planet
    noise_model_filename='/proj/data/ariel/SNRfiles/'+target+'_SNR_ARIEL.txt'

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
