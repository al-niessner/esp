'''system core ds'''
# -- IMPORTS -- ------------------------------------------------------
import os
import numpy as np
import logging; log = logging.getLogger(__name__)

import excalibur

from excalibur.system.autofill import \
    bestValue, calculate_selfConsistency_metric, \
    estimate_mass_from_radius, \
    checkValidData, fixZeroUncertainties, fixPublishedLimits, fillUncertainty, \
    fill_in_some_blank_omegas, \
    derive_RHOstar_from_M_and_R, derive_SMA_from_P_and_Mstar, \
    derive_LOGGstar_from_R_and_M, derive_LOGGplanet_from_R_and_M, \
    derive_Lstar_from_R_and_T, derive_Teqplanet_from_Lstar_and_sma, \
    derive_inclination_from_impactParam, derive_impactParam_from_inclination, \
    derive_sma_from_ars

# ------------- ------------------------------------------------------
# -- SOLAR SYSTEM CONSTANTS -- ---------------------------------------
def ssconstants(mks=False, cgs=False):
    '''
G. ROUDIER: IAU 2012
    '''
    if mks and cgs: ssc = {'Idiot':True}
    if not(mks or cgs):
        ssc = {'Rjup/Rsun':1.0276268506540176e-1,
               'Rsun/AU':4.650467260962158e-3}
        pass
    if mks:
        ssc = {'Rsun':6.957e8,
               'Msun':1.9884158605722263e30,
               'Lsun':3.828e26,
               'Rjup':7.1492e7,
               'Mjup':1.8985233541508517e27,
               'Rearth':6.371e6,
               'Mearth':5.972168e24,
               'AU':1.495978707e11,
               'G':6.67428e-11,
               'c':2.99792e8,
               'Rgas':8.314462e3,
               'Rjup/Rsun':1.0276268506540176e-1,
               'Rsun/AU':4.650467260962158e-3,
               'Tsun': 5772}
        pass
    if cgs:
        ssc = {'Rsun':6.957e10,
               'Msun':1.9884158605722263e33,
               'Lsun':3.828e33,
               'Rjup':7.1492e9,
               'Mjup':1.8985233541508517e30,
               'Rearth':6.371e8,
               'Mearth':5.972168e27,
               'AU':1.495978707e13,
               'G':6.67428e-8,
               'c':2.99792e10,
               'Rgas':8.314462e7,
               'Rjup/Rsun':1.0276268506540176e-1,
               'Rsun/AU':4.650467260962158e-3,
               'Tsun': 5772}
        pass
    ssc['day'] = 24.*60.*60.
    return ssc
# ---------------------------- ---------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    '''
G. ROUDIER: Tests for empty SV shell
    '''
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- BUILD SYSTEM PRIORS -- ------------------------------------------
def buildsp(autofill, out, verbose=False):
    '''
    G. ROUDIER: Surjection from target.autofill.parameters to dictionary output
    '''
    target = list(autofill['starID'].keys())
    target = target[0]
    for p in autofill['starID'][target]['planets']: out['priors'][p] = {}
    out['priors']['planets'] = autofill['starID'][target]['planets'].copy()
    # print('    PLANETS:',out['priors']['planets'])
    out['starkeys'].extend(autofill['starkeys'])
    out['planetkeys'].extend(autofill['planetkeys'])
    out['exts'].extend(autofill['exts'])
    # MANDATORY STELLAR PARAMETERS
    # out['starmdt'].extend(['R*', 'T*', 'FEH*', 'LOGG*', 'M*', 'RHO*', 'Hmag'])
    # out['starmdt'].extend(['R*', 'M*', 'LOGG*', 'RHO*', 'FEH*', 'Hmag', 'T*'])
    # out['starmdt'].extend(['R*', 'M*', 'LOGG*', 'RHO*', 'FEH*', 'Hmag', 'T*', 'L*'])
    out['starmdt'].extend(['R*', 'M*', 'LOGG*', 'T*', 'L*', 'RHO*', 'FEH*', 'Jmag','Hmag','Kmag'])
    # NEW CATEGORY: PARAMS TO PASS THROUGH TO ANCILLARY, BUT THEY"RE NOT MANDATORY
    # out['starnonmdt'].extend(['spTyp', 'L*', 'AGE*'])
    # change L* to mandatory (needed for planet T_eq)
    # out['starnonmdt'].extend(['spTyp', 'AGE*'])
    # 3 additional parameters needed for Ariel-RAD - distance, impact parameter, and transit duration
    # also add in transit depth. may be useful for tracking down depth differences vs taurex
    out['starnonmdt'].extend(['spTyp', 'AGE*', 'dist'])
    # AWKWARD: non-mandatory has to be included in 'starmdt' or it won't display
    out['starmdt'].extend(out['starnonmdt'])
    # MANDATORY PLANET PARAMETERS
    # out['planetmdt'].extend(['inc', 'period', 'ecc', 'rp', 't0', 'sma', 'mass', 'logg'])
    # out['planetmdt'].extend(['rp', 'mass', 'logg', 'sma', 'period', 't0', 'inc', 'ecc', 'omega'])
    out['planetmdt'].extend(['rp', 'mass', 'logg', 'teq', 'sma', 'period', 't0',
                             'inc', 'ecc', 'omega', 'impact'])
    out['planetnonmdt'].extend(['trandur', 'trandepth'])
    out['planetmdt'].extend(out['planetnonmdt'])

    # verify that all needed fields exist in the incoming target state vector
    # (some older crap targets don't have everything, e.g. SWEEPS-11 missing Hmag_uperr)
    valid = checkValidData(autofill['starID'][target],
                           out['starmdt'],
                           out['planetmdt'])
    if not valid: return False

    # some uncertainties are zero, e.g. lots of e=0+-0
    #  remove the zeros (will be filled in below with fillUncertainty())
    autofill['starID'][target] = fixZeroUncertainties(autofill['starID'][target],
                                                      out['starmdt'],
                                                      out['planetmdt'])

    # some values pulled from the Archive are actually upper/lower limits
    #  deal with each parameter on a case-by-case basis
    starLimitReplacements = {'FEH*':'',            # drop this limit on metallicity
                             'AGE*':'keep value',  # keep the age limit?  better than nothing maybe
                             }
    planetLimitReplacements = {'ecc':'0',  # eccentricity upper limits might as well be zero
                               'rp':'',    # (K2-22 b is actually an upper limit on planet radius)
                               'mass':'',  # better to use mass-radius relation than mass upper limit
                               'inc':'keep value',     # lower limits (maybe set to 90?)
                               'impact':'keep value',  # upper limits (maybe set to 0?)
                               'period':'',            # K2-93 has both lower and upper limits. strange
                               }
    # 3 options for inclination/impact parameter   which one???
    #    1) keep the limit as a value
    #    2) use the default values (90, 0)
    #    3) zero it out and hope there's a better one (not good for self-consistency)
    autofill['starID'][target] = fixPublishedLimits(autofill['starID'][target],
                                                    starLimitReplacements, planetLimitReplacements)

    # use stellar mass,radius to fill in blank stellar density
    RHO_derived, RHO_lowerr_derived, RHO_uperr_derived, RHO_ref_derived = \
        derive_RHOstar_from_M_and_R(autofill['starID'][target])
    if autofill['starID'][target]['RHO*'] != RHO_derived:
        # print('RHO before ',autofill['starID'][target]['RHO*'])
        # print('RHO derived',RHO_derived)
        # print('RHO_ref derived',RHO_ref_derived)
        # print('RHO_ref before ',autofill['starID'][target]['RHO*_ref'])
        autofill['starID'][target]['RHO*'] = RHO_derived
        autofill['starID'][target]['RHO*_lowerr'] = RHO_lowerr_derived
        autofill['starID'][target]['RHO*_uperr'] = RHO_uperr_derived
        autofill['starID'][target]['RHO*_ref'] = RHO_ref_derived

    # use a/Rp to fill in semi-major axis
    #  (make sure this comes before sma is derived from period,M*)
    for p in autofill['starID'][target]['planets']:
        sma_derived, sma_lowerr_derived, sma_uperr_derived, sma_ref_derived = \
            derive_sma_from_ars(autofill['starID'][target],p)
        # if autofill['starID'][target][p]['sma'] != sma_derived:
        #    print('sma before ',autofill['starID'][target][p]['sma'])
        #    print('sma derived',sma_derived)
        #    print('sma_ref derived',sma_ref_derived)
        #    print('sma_ref before ',autofill['starID'][target][p]['sma_ref'])
        autofill['starID'][target][p]['sma'] = sma_derived
        autofill['starID'][target][p]['sma_lowerr'] = sma_lowerr_derived
        autofill['starID'][target][p]['sma_uperr'] = sma_uperr_derived
        autofill['starID'][target][p]['sma_ref'] = sma_ref_derived
        autofill['starID'][target][p]['sma_units'] = ['[AU]']*len(sma_derived)

    # use orbital period, stellar mass to fill in blank semi-major axis
    for p in autofill['starID'][target]['planets']:
        sma_derived, sma_lowerr_derived, sma_uperr_derived, sma_ref_derived = \
            derive_SMA_from_P_and_Mstar(autofill['starID'][target],p)
        if autofill['starID'][target][p]['sma'] != sma_derived:
            # print('SMA before ',autofill['starID'][target][p]['sma'])
            # print('SMA derived',sma_derived)
            # print('SMA_ref derived',sma_ref_derived)
            # print('SMA_ref before ',autofill['starID'][target][p]['sma_ref'])
            autofill['starID'][target][p]['sma'] = sma_derived
            autofill['starID'][target][p]['sma_lowerr'] = sma_lowerr_derived
            autofill['starID'][target][p]['sma_uperr'] = sma_uperr_derived
            autofill['starID'][target][p]['sma_ref'] = sma_ref_derived

    # use stellar mass and radius to fill in blank stellar log-g
    logg_derived, logg_lowerr_derived, logg_uperr_derived, logg_ref_derived = \
        derive_LOGGstar_from_R_and_M(autofill['starID'][target])
    if autofill['starID'][target]['LOGG*'] != logg_derived:
        # print('logg* before ',autofill['starID'][target]['LOGG*'])
        # print('logg* derived',logg_derived)
        # print('logg*_ref derived',logg_ref_derived)
        # print('logg*_ref before ',autofill['starID'][target]['LOGG*_ref'])
        autofill['starID'][target]['LOGG*'] = logg_derived
        autofill['starID'][target]['LOGG*_lowerr'] = logg_lowerr_derived
        autofill['starID'][target]['LOGG*_uperr'] = logg_uperr_derived
        autofill['starID'][target]['LOGG*_ref'] = logg_ref_derived

    # use stellar radius and temperature to fill in blank stellar luminosity
    lstar_derived, lstar_lowerr_derived, lstar_uperr_derived, lstar_ref_derived = \
        derive_Lstar_from_R_and_T(autofill['starID'][target])
    if autofill['starID'][target]['L*'] != lstar_derived:
        # print('L* before ',autofill['starID'][target]['L*'])
        # print('L* derived',lstar_derived)
        # print('L*_ref derived',lstar_ref_derived)
        # print('L*_ref before ',autofill['starID'][target]['L*_ref'])
        autofill['starID'][target]['L*'] = lstar_derived
        autofill['starID'][target]['L*_lowerr'] = lstar_lowerr_derived
        autofill['starID'][target]['L*_uperr'] = lstar_uperr_derived
        autofill['starID'][target]['L*_ref'] = lstar_ref_derived

    # use stellar luminosity and planet orbit to derive planet temperature
    for p in autofill['starID'][target]['planets']:
        teq_derived, teq_lowerr_derived, teq_uperr_derived, teq_ref_derived = \
            derive_Teqplanet_from_Lstar_and_sma(autofill['starID'][target],p)
        # print('   planet:',p)
        # print('teq derived',teq_derived, teq_lowerr_derived, teq_uperr_derived, teq_ref_derived)
        if autofill['starID'][target][p]['teq'] != teq_derived:
            # print('teq before ',autofill['starID'][target][p]['teq'])
            # print('teq derived',teq_derived)
            # print('teq_ref derived',teq_ref_derived)
            # print('teq_ref before ',autofill['starID'][target][p]['teq_ref'])
            autofill['starID'][target][p]['teq'] = teq_derived
            autofill['starID'][target][p]['teq_lowerr'] = teq_lowerr_derived
            autofill['starID'][target][p]['teq_uperr'] = teq_uperr_derived
            autofill['starID'][target][p]['teq_ref'] = teq_ref_derived
            autofill['starID'][target][p]['teq_units'] = ['[K]']*len(teq_derived)

    # use planet mass and radius to fill in blank planetary log-g
    for p in autofill['starID'][target]['planets']:
        logg_derived, logg_lowerr_derived, logg_uperr_derived, logg_ref_derived = \
            derive_LOGGplanet_from_R_and_M(autofill['starID'][target],p)
        # this one is different from the previous,
        #  in that the 'logg' field doesn't exist yet
        if 'logg' in autofill['starID'][target][p].keys():
            print('ERROR: logg field shouldnt exist yet')
        if 'mass' not in autofill['starID'][target][p].keys():
            print('ERROR: mass field should exist already')
        # if autofill['starID'][target][p]['logg'] != logg_derived:
        # print('logg before ',autofill['starID'][target][p]['logg'])
        # print('logg derived',logg_derived)
        # print('logg_ref derived',logg_ref_derived)
        # print('logg_ref before ',autofill['starID'][target][p]['logg_ref'])
        autofill['starID'][target][p]['logg'] = logg_derived
        autofill['starID'][target][p]['logg_lowerr'] = logg_lowerr_derived
        autofill['starID'][target][p]['logg_uperr'] = logg_uperr_derived
        autofill['starID'][target][p]['logg_ref'] = logg_ref_derived
        autofill['starID'][target][p]['logg_units'] = ['log10[cm.s-2]']*len(logg_derived)

    # use the impact parameter (and R*, a_p) to fill in blank inclinations
    for p in autofill['starID'][target]['planets']:
        inc_derived, inc_lowerr_derived, inc_uperr_derived, inc_ref_derived = \
                derive_inclination_from_impactParam(autofill['starID'][target],p)
        # if autofill['starID'][target][p]['inc'] != inc_derived:
        #    print('inc before ',autofill['starID'][target][p]['inc'])
        #    print('inc derived',inc_derived)
        #    print('inc_ref derived',inc_ref_derived)
        #    print('inc_ref before ',autofill['starID'][target][p]['inc_ref'])
        autofill['starID'][target][p]['inc'] = inc_derived
        autofill['starID'][target][p]['inc_lowerr'] = inc_lowerr_derived
        autofill['starID'][target][p]['inc_uperr'] = inc_uperr_derived
        autofill['starID'][target][p]['inc_ref'] = inc_ref_derived
        autofill['starID'][target][p]['inc_units'] = ['[degree]']*len(inc_derived)

    # (now the reverse)
    # use the inclination (and R*, a_p) to fill in blank impact parameters
    for p in autofill['starID'][target]['planets']:
        imp_derived, imp_lowerr_derived, imp_uperr_derived, imp_ref_derived = \
                derive_impactParam_from_inclination(autofill['starID'][target],p)
        # if autofill['starID'][target][p]['impact'] != imp_derived:
        #    print('imp before ',autofill['starID'][target][p]['impact'])
        #    print('imp derived',imp_derived)
        #    print('imp_ref derived',imp_ref_derived)
        #    print('imp_ref before ',autofill['starID'][target][p]['impact_ref'])
        autofill['starID'][target][p]['impact'] = imp_derived
        autofill['starID'][target][p]['impact_lowerr'] = imp_lowerr_derived
        autofill['starID'][target][p]['impact_uperr'] = imp_uperr_derived
        autofill['starID'][target][p]['impact_ref'] = imp_ref_derived
        autofill['starID'][target][p]['impact_units'] = ['[R*]']*len(imp_derived)

    # if planet argument of periastron is missing and eccentricity is zero, set to zero
    for p in autofill['starID'][target]['planets']:
        omega_filled, omega_lowerr_filled, omega_uperr_filled, omega_ref_filled = \
            fill_in_some_blank_omegas(autofill['starID'][target],p)
        # if autofill['starID'][target][p]['omega'] != omega_filled:
        #    print('omega before ',autofill['starID'][target][p]['omega'])
        #    print('omega filled',omega_filled)
        #    print('omega_ref filled',omega_ref_filled)
        #    print('omega_ref before ',autofill['starID'][target][p]['omega_ref'])
        autofill['starID'][target][p]['omega'] = omega_filled
        autofill['starID'][target][p]['omega_lowerr'] = omega_lowerr_filled
        autofill['starID'][target][p]['omega_uperr'] = omega_uperr_filled
        autofill['starID'][target][p]['omega_ref'] = omega_ref_filled
        autofill['starID'][target][p]['omega_units'] = ['[degree]']*len(omega_filled)

    # determine the best reference to use, based on self-consistent use of that publication
    _,bestref,bestpubIndices = calculate_selfConsistency_metric(autofill['starID'][target])
    if verbose:
        bestscore,bestref,bestpubIndices = calculate_selfConsistency_metric(autofill['starID'][target])
        print('score/ref',bestscore,bestref)

    # loop over both the mandatory and non-mandatory parameters
    # for lbl in out['starmdt']+out['starnonmdt']:
    for lbl in out['starmdt']:
        # print(' ')
        # print(' ***',lbl,'***')
        #  check for upper/lower limits.  these should have been dealt with above (fixPublishedLimits)
        if lbl+'_lim' in autofill['starID'][target]:
            # print('limit flag',lbl,autofill['starID'][target])
            # print('limit flag',lbl,autofill['starID'][target][lbl+'_lim'])
            for ipub in range(len(autofill['starID'][target][lbl])):
                # print('ipub',ipub,autofill['starID'][target][lbl][ipub])
                val = autofill['starID'][target][lbl][ipub]
                lim = autofill['starID'][target][lbl+'_lim'][ipub]
                ref = autofill['starID'][target][lbl+'_ref'][ipub]
                if val!='' or lim!='':  # skip blank fields
                    if val=='':
                        if lim!='0':
                            print('  huh? a blank field has a limit?!',lbl,val,lim,ref)
                    elif lim=='':  # sometimes limit is missing for derived params
                        if not ref.startswith('derive'):
                            print('  huh? blank limit field?!',lbl,val,lim,ref)
                    elif lbl=='AGE*':
                        # print('ages are sometimes upper or lower limits; pretend itis a value')
                        pass
                    elif lim!='0':  # skip regular measurements (not upper/lower limits)
                        print('  there is a limit flag in the dataset!',lbl,val,lim,ref)
        elif lbl in ['Jmag','Hmag','Kmag','dist','spTyp']:
            pass  # these parameters don't have limit flags
        else:
            print('  ERROR: no limit flag for ',lbl)

        try:
            values = autofill['starID'][target][lbl].copy()
            uperrs = autofill['starID'][target][lbl+'_uperr'].copy()
            lowerrs = autofill['starID'][target][lbl+'_lowerr'].copy()
            refs = autofill['starID'][target][lbl+'_ref'].copy()
            value,uperr,lowerr,ref = bestValue(values,uperrs,lowerrs,refs,lbl,
                                               bestref,bestpubIndices['star'])
        except KeyError:
            value = ''
            uperr = ''
            lowerr = ''
            ref = ''
        if str(value).__len__() > 0:
            if lbl=='spTyp':
                out['priors'][lbl] = value
                out['priors'][lbl+'_uperr'] = ''
                out['priors'][lbl+'_lowerr'] = ''
            else:
                out['priors'][lbl] = float(value)
                fillvalue,autofilled = fillUncertainty(lbl,value,uperr,'uperr')
                if autofilled: out['autofill'].append('errorEstimate:'+lbl+'_uperr')
                out['priors'][lbl+'_uperr'] = fillvalue
                fillvalue,autofilled = fillUncertainty(lbl,value,lowerr,'lowerr')
                if autofilled: out['autofill'].append('errorEstimate:'+lbl+'_lowerr')
                out['priors'][lbl+'_lowerr'] = fillvalue
            strval = autofill['starID'][target][lbl+'_units'].copy()
            out['priors'][lbl+'_units'] = strval[0]
            out['priors'][lbl+'_ref'] = ref
        else:
            out['priors'][lbl] = ''
            for ext in out['exts']: out['priors'][lbl+ext] = ''
            # don't add parameter to the 'missing' list if it is non-mandatory
            if lbl not in out['starnonmdt']:
                out['needed'].append(lbl)
            pass
        pass

    # if stellar metallicity is missing, assume solar
    if 'FEH*' in out['needed']:
        out['priors']['FEH*'] = 0
        out['priors']['FEH*_uperr'] = 0.25
        out['priors']['FEH*_lowerr'] = -0.25
        out['priors']['FEH*_units'] = '[dex]'
        out['priors']['FEH*_ref'] = 'default to solar metallicity'
        out['autofill'].append('default:FEH*')
        out['autofill'].append('default:FEH*_uperr')
        out['autofill'].append('default:FEH*_lowerr')
        index = out['needed'].index('FEH*')
        out['needed'].pop(index)

    # if star luminosity is missing, assume M^4
    #   5/29/23 note that this conditional is never true.  could delete this
    if 'L*' in out['needed']:
        if out['priors']['M*'].__len__() > 0:
            Lstar = float(out['priors']['M*'])**4
            print('Note: setting L* = ',Lstar)
            out['priors']['L*'] = f"{Lstar:6.2f}"
            out['priors']['L*_units'] = '[Lsun]'
            out['priors']['L*_ref'] = 'set to M*^4'
            # (this will set uncertainty to 10%)
            out['priors']['L*_uperr'],_ = fillUncertainty('L*',Lstar,'','uperr')
            out['priors']['L*_lowerr'],_ = fillUncertainty('L*',Lstar,'','lowerr')
            out['autofill'].append('default:L*')
            out['autofill'].append('default:L*_uperr')
            out['autofill'].append('default:L*_lowerr')
            index = out['needed'].index('L*')
            out['needed'].pop(index)

    ssc = ssconstants(cgs=True)
    for p in out['priors']['planets']:
        for lbl in out['planetmdt']:

            #  check for upper/lower limits.  these should have been dealt with above (fixPublishedLimits)
            if lbl+'_lim' in autofill['starID'][target][p]:
                # print('limit flag',lbl,autofill['starID'][target][p][lbl+'_lim'])
                for ipub in range(len(autofill['starID'][target][p][lbl])):
                    val = autofill['starID'][target][p][lbl][ipub]
                    lim = autofill['starID'][target][p][lbl+'_lim'][ipub]
                    ref = autofill['starID'][target][p][lbl+'_ref'][ipub]
                    if val!='' or lim!='':  # skip blank fields
                        if val=='':
                            # ok this is fine (it happens for spectral type)
                            if lim!='0':
                                print('  huh? a blank field has a limit?!',lbl,val,lim,ref)
                        elif lim=='':  # sometimes limit is missing for derived params
                            if not ref.startswith('derive'):
                                print('  huh? blank limit field?!',p,lbl,val,lim,ref)
                        elif lim!='0':  # skip regular measurements (not upper/lower limits)
                            print('  there is a limit flag in the dataset!',p,lbl,val,lim,ref)
                        # else:
                        #    print('  val,lim',p,lbl,val,lim,ref)
            elif lbl=='logg':
                pass  # these parameters don't have limit flags
            else:
                print('  ERROR: no limit flag for ',lbl)

            values = autofill['starID'][target][p][lbl].copy()
            uperrs = autofill['starID'][target][p][lbl+'_uperr'].copy()
            lowerrs = autofill['starID'][target][p][lbl+'_lowerr'].copy()
            refs = autofill['starID'][target][p][lbl+'_ref'].copy()
            value,uperr,lowerr,ref = bestValue(values,uperrs,lowerrs,refs,lbl,
                                               bestref,bestpubIndices[p])

            # if lbl=='teq':
            #     print('best value below',lbl,value,type(value))
            # print('')
            # print(lbl)
            # print(value)
            # print(values)
            if value.__len__() > 0:
                out['priors'][p][lbl] = float(value)
                out['priors'][p][lbl+'_ref'] = ref
                err,autofilled = fillUncertainty(lbl,value,uperr,'uperr')
                if autofilled: out['autofill'].append('errorEstimate:'+p+':'+lbl+'_uperr')
                out['priors'][p][lbl+'_uperr'] = err
                err,autofilled = fillUncertainty(lbl,value,lowerr,'lowerr')
                if autofilled: out['autofill'].append('errorEstimate:'+p+':'+lbl+'_lowerr')
                out['priors'][p][lbl+'_lowerr'] = err
                strval = autofill['starID'][target][p][lbl+'_units'].copy()
                out['priors'][p][lbl+'_units'] = strval[-1]
                pass
            else:
                out['priors'][p][lbl] = ''
                for ext in out['exts']: out['priors'][p][lbl+ext] = ''
                # don't add parameter to the 'missing' list if it is non-mandatory
                if lbl not in out['planetnonmdt']:
                    out['needed'].append(p+':'+lbl)
                pass
            pass

        # if planet mass is missing (and there is a planet radius),
        #  fill in the mass with an assumed mass-radius relation
        if p+':mass' in out['needed'] and p+':rp' not in out['needed']:
            planet_radius = out['priors'][p]['rp']
            assumed_planet_mass = estimate_mass_from_radius(planet_radius)
            out['priors'][p]['mass'] = assumed_planet_mass
            index = out['needed'].index(p+':mass')
            out['needed'].pop(index)
            # uncertainty is pretty large here; let's say 50%
            out['priors'][p]['mass_uperr'] = 0.5 * assumed_planet_mass
            out['priors'][p]['mass_lowerr'] = -0.5 * assumed_planet_mass
            out['priors'][p]['mass_units'] = '[Jupiter mass]'
            out['priors'][p]['mass_ref'] = 'assumed mass/radius relation'
            out['autofill'].append('MRrelation:'+p+':mass')
            out['autofill'].append('MRrelation:'+p+':mass_uperr')
            out['autofill'].append('MRrelation:'+p+':mass_lowerr')

            # careful here - logg also has to be filled in
            if p+':logg' in out['needed']:
                g = ssc['G'] * assumed_planet_mass*ssc['Mjup'] \
                    / (float(planet_radius)*ssc['Rjup'])**2
                logg = np.log10(g)
                out['priors'][p]['logg'] = f'{logg:6.4f}'
                index = out['needed'].index(p+':logg')
                out['needed'].pop(index)
                # uncertainty is pretty large here; let's say 0.2 dex
                out['priors'][p]['logg_uperr'] = 0.2
                out['priors'][p]['logg_lowerr'] = -0.2
                out['priors'][p]['logg_units'] = 'log10[cm.s-2]'
                out['priors'][p]['logg_ref'] = 'assumed mass/radius relation'
                out['autofill'].append('MRrelation:'+p+':logg')
                out['autofill'].append('MRrelation:'+p+':logg_uperr')
                out['autofill'].append('MRrelation:'+p+':logg_lowerr')
            else:
                print('STRANGE: why was there logg but no mass')

        # if planet eccentricity is missing, assume 0
        if p+':ecc' in out['needed']:
            out['priors'][p]['ecc'] = 0
            index = out['needed'].index(p+':ecc')
            out['needed'].pop(index)
            out['priors'][p]['ecc_units'] = ''
            out['priors'][p]['ecc_ref'] = 'default: assumed circular orbit'
            # (this will set uncertainty to the min value of 0.1)
            out['priors'][p]['ecc_uperr'],_ = fillUncertainty('ecc',0,'','uperr')
            out['priors'][p]['ecc_lowerr'],_ = fillUncertainty('ecc',0,'','lowerr')
            out['autofill'].append('default:'+p+':ecc')
            out['autofill'].append('default:'+p+':ecc_uperr')
            out['autofill'].append('default:'+p+':ecc_lowerr')

        # if planet inclination is missing, assume 90 degrees
        if p+':inc' in out['needed']:
            inc = 90e0
            out['priors'][p]['inc'] = inc
            index = out['needed'].index(p+':inc')
            out['needed'].pop(index)
            out['priors'][p]['inc_units'] = '[degree]'
            out['priors'][p]['inc_ref'] = 'default: assumed edge-on orbit'
            out['priors'][p]['inc_uperr'],_ =fillUncertainty('inc',inc,'','uperr')
            out['priors'][p]['inc_lowerr'],_=fillUncertainty('inc',inc,'','lowerr')
            out['autofill'].append('default:'+p+':inc')
            out['autofill'].append('default:'+p+':inc_uperr')
            out['autofill'].append('default:'+p+':inc_lowerr')

        # if argument of periastron is missing, assume 0 degrees
        # this check shouldn't be needed anymore,
        #   but there are some publications that list positive ecc and no omega
        if p+':omega' in out['needed']:
            out['priors'][p]['omega'] = 0
            index = out['needed'].index(p+':omega')
            out['needed'].pop(index)
            out['priors'][p]['omega_units'] = '[degree]'
            out['priors'][p]['omega_ref'] = 'default: assume 0'
            # give it a large uncertainty, say 120 degrees 1-sigma
            out['priors'][p]['omega_uperr'] = 120
            out['priors'][p]['omega_lowerr'] = 120
            out['autofill'].append('default:'+p+':omega')
            out['autofill'].append('default:'+p+':omega_uperr')
            out['autofill'].append('default:'+p+':omega_lowerr')

        # if impact parameter is missing, assume 0
        if p+':impact' in out['needed']:
            out['priors'][p]['impact'] = 0
            index = out['needed'].index(p+':impact')
            out['needed'].pop(index)
            out['priors'][p]['impact_units'] = '[R*]'
            out['priors'][p]['impact_ref'] = 'default: assumed edge-on orbit'
            out['priors'][p]['impact_uperr'] = 0.5
            out['priors'][p]['impact_lowerr'] = 0.5
            out['autofill'].append('default:'+p+':impact')
            out['autofill'].append('default:'+p+':impact_uperr')
            out['autofill'].append('default:'+p+':impact_lowerr')

    for key in out['needed']:
        if ':' in key:
            p = key.split(':')[0]
            if p not in out['ignore']:
                out['ignore'].append(p)
                out['pignore'][p] = out['priors'][p].copy()
                out['priors'].pop(p, None)
                index = out['priors']['planets'].index(p)
                out['priors']['planets'].pop(index)
                pass
            pass
        pass

    # when adding new targets, output star/planet info needed as input for ArielRad
    newtarget = 0
    if newtarget:
        # print(out['priors'].keys())
        # print(out['priors']['b'].keys())
        for planetLetter in out['priors']['planets']:
            print('ArielRad input:   ,',target+','+
                  out['priors']['M*']+','+
                  out['priors']['T*']+','+
                  out['priors']['R*']+','+
                  out['priors']['dist']+','+
                  out['priors']['Kmag']+','+
                  ' '+','+
                  target,planetLetter+','+
                  out['priors'][planetLetter]['period']+','+
                  out['priors'][planetLetter]['teq']+','+
                  out['priors'][planetLetter]['sma'] *ssc['AU']/100+','+
                  out['priors'][planetLetter]['rp'] *ssc['Rjup']/ssc['Rearth']+','+
                  0.3+','+
                  out['priors'][planetLetter]['mass'] *ssc['Mjup']/ssc['Mearth']+','+
                  2.3+','+
                  out['priors'][planetLetter]['trandur']*3600+','+
                  out['priors'][planetLetter]['impact']+','+
                  1)
        # units for planet params:
        # [day] [K] [m]	[Rearth] [] [Mearth] [g/mol] [s] []

    for p in out['ignore']:
        dropIndices = []
        for value in out['needed']:
            index = out['needed'].index(value)
            if p+':' in value:
                dropIndices.append(index)
                out['pneeded'].append(value)
                pass
            pass
        if dropIndices:
            # have to pop off the list in reverse order
            dropIndices.reverse()
            for index in dropIndices:
                out['needed'].pop(index)
        pass
    # starneed = False
    # for p in out['needed']:
    #    if ':' not in p: starneed = True
    #    pass
    # if starneed or (len(out['priors']['planets']) < 1): out['PP'].append(True)
    # log.warning('>-- FORCE PARAMETER: %s', str(out['PP'][-1]))
    # log.warning('>-- MISSING MANDATORY PARAMETERS: %s', str(out['needed']))
    # log.warning('>-- MISSING PLANET PARAMETERS: %s', str(out['pneeded']))
    # log.warning('>-- PLANETS IGNORED: %s', str(out['ignore']))
    # log.warning('>-- AUTOFILL: %s', str(out['autofill']))
    out['STATUS'].append(True)

    return True
# ------------------------- ------------------------------------------
# -- FORCE PRIOR PARAMETERS -- ---------------------------------------
def forcepar(overwrite, out, verbose=False):
    '''
    G. ROUDIER: Completes/Overrides parameters using target/edit.py
    '''

    forced = False  # check if any parameters are overwritten
    # print('starting to force')
    # print('edit.py stuff for this guy:',overwrite)
    # print('out.keys',out.keys())
    for key in overwrite.keys():
        mainkey = key.split(':')[0]
        # print(' key',key)
        if (mainkey not in out.keys()) and (len(mainkey) < 2):
            if mainkey in out['pignore'].keys():
                print('its in pignore',out['pignore'])
                out['priors'][mainkey] = out['pignore'][mainkey].copy()
                pass
            for pkey in overwrite[key].keys():
                if pkey in out['priors'][mainkey].keys() and \
                   out['priors'][mainkey][pkey]!='':
                    if verbose: print('OVERWRITING:',mainkey,pkey,overwrite[mainkey][pkey],
                                      ' current value:',out['priors'][mainkey][pkey])
                else:
                    # print(' adding key:',mainkey,pkey)
                    if verbose: print('OVERWRITING A BLANK:',mainkey,pkey,overwrite[mainkey][pkey])
                out['priors'][mainkey][pkey] = overwrite.copy()[mainkey][pkey]
                if not pkey.endswith('ref') and not pkey.endswith('units'):
                    out['autofill'].append('forcepar:'+mainkey+':'+pkey)
                forced = True
            pass
        else:
            if out['priors'][key]=='':
                if verbose: print('OVERWRITING A BLANK:',key,overwrite[key])
            else:
                if verbose: print('OVERWRITING:',key,overwrite[key],
                                  ' current value:',out['priors'][key])
            out['priors'][key] = overwrite.copy()[key]
            if not key.endswith('ref') and not key.endswith('units'):
                out['autofill'].append('forcepar:'+key)
            forced = True
    # if something has been overwritten, save forcepar as true (shows at top of system.finalize)
    out['PP'].append(forced)

    for n in out['needed'].copy():
        if ':' not in n:
            try:
                float(out['priors'][n])
                out['needed'].pop(out['needed'].index(n))
                pass
            except (ValueError, KeyError): forced = False
            pass
        else:
            try:
                pnet = n.split(':')[0]
                pkey = n.split(':')[1]
                float(out['priors'][pnet][pkey])
                out['needed'].pop(out['needed'].index(n))
                pass
            except (ValueError, KeyError):
                if ('mass' not in n) and ('rho' not in n): forced = False
                pass
            pass
        pass
    ptry = []
    for p in out['pneeded'].copy():
        pnet = p.split(':')[0]
        pkey = p.split(':')[1]
        ptry.append(pnet)
        try:
            float(out['priors'][pnet][pkey])
            out['pneeded'].pop(out['pneeded'].index(p))
            pass
        except (ValueError, KeyError):
            if ('mass' not in p) and ('rho' not in p): forced = False
            pass
        pass
    for p in set(ptry):
        addback = True
        planetchecklist = [k for k in out['pneeded']
                           if k.split(':')[0] not in out['ignore']]
        for pkey in planetchecklist:
            if pkey in [p+':mass', p+':rho']:
                if 'logg' not in out['priors'][p].keys(): addback = False
                else: out['pneeded'].pop(out['pneeded'].index(pkey))
            else:
                if p in pkey: addback = False
            pass
        if addback: out['priors']['planets'].append(p)
    starneed = False
    for p in out['needed']:
        if ':' not in p: starneed = True

    # these 5 print statements moved from above, so they're after overwriting
    # actually move it out to after call; this print only done if params are forced
    # log.warning('>-- FORCE PARAMETER: %s', str(out['PP'][-1]))
    # log.warning('>-- MISSING MANDATORY PARAMETERS: %s', str(out['needed']))
    # log.warning('>-- MISSING PLANET PARAMETERS: %s', str(out['pneeded']))
    # log.warning('>-- PLANETS IGNORED: %s', str(out['ignore']))
    # log.warning('>-- AUTOFILL: %s', str(out['autofill']))
    if starneed or (len(out['priors']['planets']) < 1):
        success = False
        # log.warning('>-- MISSING MANDATORY PARAMETERS')
        # log.warning('>-- ADD THEM TO SYSTEM/OVERWRITER')
        log.warning('>-- PARAMETER STILL MISSING; ADD TO SYSTEM/OVERWRITER')
    else:
        success = True
        # log.warning('>-- PRIORITY PARAMETERS SUCCESSFUL')
        log.warning('>-- PARAMETER FORCING SUCCESSFUL')
    return success
# ---------------------------- ---------------------------------------
def savesv(aspects, targetlists):
    '''
    save the results as a csv file in /proj/data/spreadsheets
    '''
    svname = 'system.finalize.parameters'

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
    # saveDir = excalibur.context['data_dir'] + \
    #    '/spreadsheets/RID' + f"{RID:03i}" + '/'
    #   '/spreadsheets/RID' + str('%03i' %RID) + '/'
    # print('saveDir:',saveDir)
    if not os.path.exists(saveDir): os.mkdir(saveDir)

    # file name where the results are saved
    # outfileName = svname.replace('.','_') + '_RID' + str('%03i' %RID) + '.csv'
    # outfileName = svname.replace('.','_') + '_RID' + f"{RID:03i}" + '.csv'
    outfileName = svname.replace('.','_') + '_RID'+RID + '.csv'
    # outfile = open(saveDir + outfileName,'w',encoding='ascii')
    with open(saveDir + outfileName,'w',encoding='ascii') as outfile:

        # 55 Cnc is used as an example, to get the default header
        system_data = aspects['55 Cnc'][svname]
        st_keys = system_data['starmdt']           # start with mandatory params
        # also include non-mandatory params?
        #  no actually they've already been added in. (see 'awkward' above)
        # st_keys.extend(system_data['starnonmdt'])
        # print('st_keys',st_keys)
        pl_keys = system_data['planetmdt']
        # print('pl_keys',pl_keys)

        # this extension info is not needed for value histograms, but is required for full data dump
        exts = system_data['exts']
        # print('extensions:',exts)

        # write the header row
        outfile.write('star,planet,')
        for key in st_keys:
            outfile.write(key + ',')
            for ext in exts:
                outfile.write(key+ext + ',')
        for key in pl_keys:
            outfile.write(key + ',')
            for ext in exts:
                outfile.write(key+ext + ',')
        outfile.write('\n')

        # loop through each target, with one row per planet
        for trgt in targetlists['active']:
            system_data = aspects[trgt][svname]

            for planet_letter in system_data['priors']['planets']:
                outfile.write(trgt + ',')
                outfile.write(planet_letter + ',')

                for key in st_keys:
                    outfile.write(str(system_data['priors'][key]) + ',')
                    for ext in exts:
                        outfile.write(str(system_data['priors'][key+ext]).replace(',',';') + ',')

                for key in pl_keys:
                    outfile.write(str(system_data['priors'][planet_letter][key]) + ',')
                    for ext in exts:
                        outfile.write(str(system_data['priors'][planet_letter][key+ext]).replace(',',';') + ',')

                outfile.write('\n')
    # jenkins requires with-open, so close isn't needed anymore
    # instead there's a bunch of dumbass indentation. thanks so much jenkins!
    # outfile.close()

    return
