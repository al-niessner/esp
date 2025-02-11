'''utilites used by many tasks'''
# -- IMPORTS -- ------------------------------------------------------
import numpy as np
from pytensor import tensor
from pytensor.ifelse import ifelse
# ------------ -------------------------------------------------------
# -- CHEMICAL EQUILIBRIUM -- -----------------------------------------
def crbce(p, temp, C2Or=0., X2Hr=0., N2Or=0.):
    '''
G. ROUDIER: BURROWS AND SHARP 1998 + ANDERS & GREVESSE 1989
    '''
    solar = {
        'nH':9.10e-1, 'nHe':8.87e-2, 'nO':7.76e-4, 'nC':3.29e-4,
        'nNE':1.12e-4, 'nN':1.02e-4, 'nMg':3.49e-5, 'nSi':3.26e-5,
        'nFe':2.94e-5, 'nS':1.68e-5, 'nAr':3.29e-6, 'nAl':2.77e-6,
        'nCa':1.99e-6, 'nNa':1.87e-6, 'nNi':1.61e-6, 'nCr':4.40e-7,
        'nP':3.39e-7, 'nMn':3.11e-7, 'nCl':1.71e-7, 'nK':1.23e-7,
        'nTi':7.83e-8, 'nCo':7.34e-8, 'nF':2.75e-8, 'nV':9.56e-9,
        'nLi':1.86e-9, 'nRb':2.31e-10, 'nCs':1.21e-11
    }
    # print('temp in crbce',temp.eval())
    a1 = 1.106131e6
    b1 = -5.6895e4
    c1 = 62.565
    d1 = -5.81396e-4
    e1 = 2.346515e-8
    RcalpmolpK = 1.9872036  # cal/mol/K
    solCtO = solar['nC']/solar['nO']
    solNtO = solar['nN']/solar['nO']
    metal = solar.copy()
    metal['nH'] = 0.
    metal['nHe'] = 0.

    # print('temp in crbce',temp.eval())
    solvec = np.array([metal[xx] for xx in metal])
    if X2Hr >= np.log10(1./np.sum(solvec)):
        nH = 1e-16
        nH2 = 1e-16
        nHe = 1e-16
        X2Hr = np.log10(1./np.sum(solvec))  # 2.84 MAX
        pass
    else:
        if X2Hr < -10.:
            nH = 1./(1. + solar['nHe']/solar['nH'])
            nH2 = nH/2.
            nHe = nH*solar['nHe']/solar['nH']
            X2Hr = -10.
            pass
        else:
            nH = 1. - (10.**X2Hr)*np.sum(solvec)
            nH2 = nH/2.
            nHe = nH*solar['nHe']/solar['nH']
            pass
        pass
    C2Or = max(C2Or, -10.0)
    C2Or = min(C2Or, 10.0)
    N2Or = max(N2Or, -10.0)
    N2Or = min(N2Or, 10.0)
    pH2 = nH2*p  # array
    # strange. this np.exp call actually works fine it seems. no need to switch to tensor.exp?
    # K1 = np.exp((a1/temp + b1 + c1*temp + d1*temp**2 + e1*temp**3)/(RcalpmolpK*temp))  # tensor
    # print('K1',K1)
    K1 = tensor.exp((a1/temp + b1 + c1*temp + d1*temp**2 + e1*temp**3)/(RcalpmolpK*temp))  # tensor
    # print('K1',K1)
    # print('K1',K1.eval())
    AH2 = (pH2**2.)/(2.*K1)  # array ph2 and tensor K1
    ACpAO = (10.**X2Hr)/nH*solar['nO']*(1. + (10.**C2Or)*solCtO)  # scalar
    ACtAO = (10.**C2Or)*solCtO*(solar['nO']**2)*(((10.**X2Hr)/nH)**2)  # scalar
    BCO = ACpAO + AH2 - np.sqrt((ACpAO+AH2)**2 - 4.*ACtAO)  # AH2 tensor array
    # print('BCO',BCO,BCO.eval())  # bco is a tensor array.  why an array? oh cus AH2 is array
    # print('pH2',pH2)  # array
    # print('p',p)  # array
    # nCO = np.mean(BCO*pH2/p)  # fails
    # nCO = BCO * np.mean(pH2/p)    # pH2 and p are array  BCO is tensor
    nCO = tensor.mean(BCO*pH2/p)
    # print('nCO',nCO.eval())
    #if nCO <= 0: nCO = 1e-16      # tensor.  needs tensor ifthen
    # print('nCO',nCO.eval())
    # nCO = ifelse(nCO <= 0, 1e-16, nCO)
    nCO = tensor.maximum(nCO, 1e-16)
    print('nCO',nCO.eval())
    # BCO tensor; p,pH2 array;  X2Hr,nH scalar
    nCH4 = tensor.mean((2.*(10.**X2Hr)/nH*solar['nC'] - BCO)*pH2/p)
    nH2O = tensor.mean((2.*(10.**X2Hr)/nH*solar['nO'] - BCO)*pH2/p)
    #nCH4 = (2.*(10.**X2Hr)/nH*solar['nC'] - BCO)*np.mean(pH2/p)
    #nH2O = (2.*(10.**X2Hr)/nH*solar['nO'] - BCO)*np.mean(pH2/p)
    # if nCH4 <= 0: nCH4 = 1e-16      # tensor.  needs tensor ifthen
    # if nH2O <= 0: nH2O = 1e-16      # tensor.  needs tensor ifthen
    # nCH4 = ifelse(nCH4 <= 0, 1e-16, nCH4)
    # nCH4 = ifelse(tensor.lte(nCH4, 0), 1e-16, nCH4)  # lte doesnt exist
    # I'm getting these errors with the if-thens
    #  TypeError: The condition argument must be a truthy scalar value
    #  TypeError: 'module' object is not callable
    #    I think that means I can't do the tensor(1.e-16) stuff
    # nCH4 = ifelse(tensor.lt(nCH4, 1e-16), 1e-16, nCH4) 
    # nCH4 = ifelse(tensor.lt(nCH4, 1e-16), tensor(1e-16), nCH4)
    nCH4 = tensor.maximum(nCH4, 1e-16)
    # nH2O = ifelse(nH2O <= 0, tensor(1e-16), nH2O)
    nH2O = tensor.maximum(nH2O, 1e-16)
    print('nH2O',nH2O.eval())
    a2 = 8.16413e5
    b2 = -2.9109e4
    c2 = 58.5878
    d2 = -7.8284e-4
    e2 = 4.729048e-8
    # K2 = np.exp((a2/temp + b2 + c2*temp + d2*temp**2 + e2*temp**3)/(RcalpmolpK*temp))
    K2 = tensor.exp((a2/temp + b2 + c2*temp + d2*temp**2 + e2*temp**3)/(RcalpmolpK*temp))
    # print('K2',K2.eval())
    AN = (10.**X2Hr)*(10.**N2Or)*solNtO*solar['nO']/nH  # solar['nN']/nH
    AH2 = (pH2**2.)/(8.*K2)
    # print('AH2',AH2.eval())
    # another instrument precision problem arising here
    #  (for WASP-74 ariel sims, highmmw cases)
    # take absolute value, just to be sure that there's never a negative value
    # BN2 = AN + AH2 - np.sqrt((AN + AH2)**2. - (AN)**2.)
    BN2 = AN + AH2 - np.sqrt(np.abs((AN + AH2)**2. - (AN)**2.))
    BNH3 = 2.*(AN - BN2)
    # also I changed mean to nanmean here, though this is no longer necessary
    #nN2 = np.nanmean(BN2*pH2/p)
    #nN2 = BN2*np.nanmean(pH2/p)
    nN2 = tensor.mean(BN2*pH2/p)
    # print('nN2',nN2.eval())
    # if nN2 <= 0: nN2 = 1e-16
    # nN2 = ifelse(nN2 <= 0, 1e-16, nN2)
    nN2 = tensor.maximum(nN2, 1e-16)
    print('nN2',nN2.eval())
    # nNH3 = np.nanmean(BNH3*pH2/p)
    # nNH3 = BNH3*np.nanmean(pH2/p)
    nNH3 = tensor.mean(BNH3*pH2/p)
    # print('nNH3',nNH3.eval())
    # if nNH3 <= 0: nNH3 = 1e-16
    # nNH3 = ifelse(nNH3 <= 0, 1e-16, nNH3)
    nNH3 = tensor.maximum(nNH3, 1e-16)
    print('nNH3',nNH3.eval())
    #mixratio = {'H2O':np.log10(nH2O)+6., 'CH4':np.log10(nCH4)+6., 'NH3':np.log10(nNH3)+6.,
    #            'N2':np.log10(nN2)+6., 'CO':np.log10(nCO)+6.}
    mixratio = {'H2O':tensor.log10(nH2O)+6.,
                'CH4':tensor.log10(nCH4)+6., 'NH3':tensor.log10(nNH3)+6.,
                'N2':tensor.log10(nN2)+6., 'CO':tensor.log10(nCO)+6.}
    print('mixratio saved (H2O,CH2,N2,NH3,CO)',
          mixratio['H2O'].eval(),
          mixratio['CH4'].eval(),
          mixratio['N2'].eval(),
          mixratio['NH3'].eval(),
          mixratio['CO'].eval())
    print('nH2,nHe saved',nH2,nHe)
    return mixratio, nH2, nHe
# -------------------- -----------------------------------------------
# -- MEAN MOLECULAR WEIGHT -- ----------------------------------------
def getmmw(mixratio, protosolar=True, fH2=None, fHe=None):
    '''
G. ROUDIER: Mean molecular weight estimate assuming proton mass dominated nucleus
    '''
    molsum = 0.
    mmw = 0.
    weights = {'H2':2., 'He':4., 'CH4':16., 'NH3':17., 'H2O':18.,
               'H2CO':30., 'TIO':64,
               'HCN':27., 'N2':28., 'C2H2':26., 'NO2':46., 'N2O':44.,
               'O3':48., 'HNO3':63., 'O2':32.,
               'CO':28., 'CO2':44., 'NO':30., 'OH':17.}
    # print('mixratio',mixratio)  # (needs eval())
    for elem in mixratio:
        molsum = molsum + 10.**(mixratio[elem]-6.)
        mmw = mmw + 10.**(mixratio[elem]-6.)*weights[elem]
        # print('molsum,mmw',molsum.eval(),mmw.eval())
    # print('molsum,mmw',molsum.eval(),mmw.eval())
    mrH2He = 1. - molsum
    # Lodders 2010
    if protosolar: HEoH2 = 2.*2.343*1e9/(2.431*1e10)
    else: HEoH2 = fHe/fH2
    print('fHe,fH2',fHe,fH2)
    print('HEoH2',HEoH2)
    mrH2 = mrH2He/(1.+HEoH2)
    mrHe = HEoH2*mrH2
    mmw = mrH2*weights['H2'] + mrHe*weights['He'] + mmw
    print('*** mmw ***',mmw.eval())
    return mmw, mrH2, mrHe
