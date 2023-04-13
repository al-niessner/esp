'''system core ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

# import excalibur.target.edit as trgedit
# overwrite = trgedit.ppar()

from excalibur.system.autofill import \
    bestValue, fillUncertainty, \
    derive_RHOstar_from_M_and_R, derive_SMA_from_P_and_Mstar, \
    derive_LOGGstar_from_R_and_M, derive_LOGGplanet_from_R_and_M

import numpy as np
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
def buildsp(autofill, out):
    '''
    G. ROUDIER: Surjection from target.autofill.parameters to dictionary output
    '''
    target = list(autofill['starID'].keys())
    target = target[0]
    for p in autofill['starID'][target]['planets']: out['priors'][p] = {}
    out['priors']['planets'] = autofill['starID'][target]['planets'].copy()
    out['starkeys'].extend(autofill['starkeys'])
    out['planetkeys'].extend(autofill['planetkeys'])
    out['exts'].extend(autofill['exts'])
    # REMOVED FROM STELLAR MANDATORY LIST:  L*, AGE*
    #  we could add L* back into the list, if desired (can derive it from R*,T*)
    out['starmdt'].extend(['R*', 'T*', 'FEH*', 'LOGG*', 'M*', 'RHO*', 'Hmag'])
    # ADDED TO THE PLANET MANDATORY LIST: logg
    # out['planetmdt'].extend(['inc', 'period', 'ecc', 'rp', 't0', 'sma', 'mass'])
    out['planetmdt'].extend(['inc', 'period', 'ecc', 'rp', 't0', 'sma', 'mass', 'logg'])
    # NOTE: this list of mandatory extensions is not actually used anymore
    out['extsmdt'].extend(['_lowerr', '_uperr'])

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
            # exit('test2')

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
        # exit('test3')

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
        autofill['starID'][target][p]['logg_ref'] = logg_ref_derived
        autofill['starID'][target][p]['logg_units'] = ['log10[cm.s-2]']*len(logg_derived)
        # exit('test4')

    for lbl in out['starmdt']:
        # Retrocompatibility
        try:
            values = autofill['starID'][target][lbl].copy()
            uperrs = autofill['starID'][target][lbl+'_uperr'].copy()
            lowerrs = autofill['starID'][target][lbl+'_lowerr'].copy()
            refs = autofill['starID'][target][lbl+'_ref'].copy()
            value,uperr,lowerr,ref = bestValue(values,uperrs,lowerrs,refs)
        except KeyError:
            value = ''
            uperr = ''
            lowerr = ''
            ref = ''
        if value.__len__() > 0:
            if lbl=='spTyp':
                out['priors'][lbl] = value
            else:
                out['priors'][lbl] = float(value)
            out['priors'][lbl+'_ref'] = ref
            fillvalue,autofilled = fillUncertainty(lbl,value,uperr,'uperr')
            if autofilled: out['autofill'].append(lbl+'_uperr')
            out['priors'][lbl+'_uperr'] = fillvalue
            fillvalue,autofilled = fillUncertainty(lbl,value,lowerr,'lowerr')
            if autofilled: out['autofill'].append(lbl+'_lowerr')
            out['priors'][lbl+'_lowerr'] = fillvalue
            strval = autofill['starID'][target][lbl+'_units'].copy()
            out['priors'][lbl+'_units'] = strval[0]
            pass
        else:
            out['priors'][lbl] = ''
            for ext in out['exts']: out['priors'][lbl+ext] = ''
            out['needed'].append(lbl)
            pass
        pass
    ssc = ssconstants(cgs=True)
    if ('LOGG*' in out['needed']) and ('R*' not in out['needed']):
        radstar = (out['priors']['R*'])*(ssc['Rsun'])
        values = autofill['starID'][target]['RHO*'].copy()
        uperrs = autofill['starID'][target]['RHO*_uperr'].copy()
        lowerrs = autofill['starID'][target]['RHO*_lowerr'].copy()
        refs = autofill['starID'][target]['RHO*_ref'].copy()
        value,uperr,lowerr,ref = bestValue(values,uperrs,lowerrs,refs)
        if value.__len__() > 0:
            # exit('THIS SHOULDNT BE NEEDED ANYMORE1')
            print('THIS SHOULDNT BE NEEDED ANYMORE1')
            rho = float(value)
            g = 4e0*np.pi*(ssc['G'])*rho*radstar/3e0
            out['priors']['LOGG*'] = np.log10(g)
            out['autofill'].append('LOGG*')
            index = out['needed'].index('LOGG*')
            out['needed'].pop(index)
            # note that the RHO* uncertainty is not used (nor R* uncertainty)
            fillvalue,autofilled = fillUncertainty('LOGG*',np.log10(g),'','uperr')
            out['priors']['LOGG*_uperr'] = fillvalue
            if autofilled: out['autofill'].append('LOGG*_uperr')
            fillvalue,autofilled = fillUncertainty('LOGG*',np.log10(g),'','lowerr')
            out['priors']['LOGG*_lowerr'] = fillvalue
            if autofilled: out['autofill'].append('LOGG*_lowerr')

            strval = autofill['starID'][target]['LOGG*_units'].copy()
            out['priors']['LOGG*_units'] = strval[-1]
            out['priors']['LOGG*_ref'] = 'System Prior Auto Fill'
            pass
        values = autofill['starID'][target]['M*'].copy()
        uperrs = autofill['starID'][target]['M*_uperr'].copy()
        lowerrs = autofill['starID'][target]['M*_lowerr'].copy()
        refs = autofill['starID'][target]['M*_ref'].copy()
        value,uperr,lowerr,ref = bestValue(values,uperrs,lowerrs,refs)
        if (value.__len__() > 0) and ('LOGG*' in out['needed']):
            # exit('THIS SHOULDNT BE NEEDED ANYMORE2')
            print('THIS SHOULDNT BE NEEDED ANYMORE2')
            mass = float(value)*ssc['Msun']
            g = (ssc['G'])*mass/(radstar**2)
            out['priors']['LOGG*'] = np.log10(g)
            out['autofill'].append('LOGG*')
            index = out['needed'].index('LOGG*')
            out['needed'].pop(index)
            # note that the M* uncertainty is not used (nor R* uncertainty)
            fillvalue,autofilled = fillUncertainty('LOGG*',np.log10(g),'','uperr')
            out['priors']['LOGG*_uperr'] = fillvalue
            if autofilled: out['autofill'].append('LOGG*_uperr')
            fillvalue,autofilled = fillUncertainty('LOGG*',np.log10(g),'','lowerr')
            out['priors']['LOGG*_lowerr'] = fillvalue
            if autofilled: out['autofill'].append('LOGG*_lowerr')

            strval = autofill['starID'][target]['LOGG*_units'].copy()
            out['priors']['LOGG*_units'] = strval[-1]
            out['priors']['LOGG*_ref'] = 'System Prior Auto Fill'
            pass
        pass
    for p in out['priors']['planets']:
        for lbl in out['planetmdt']:
            values = autofill['starID'][target][p][lbl].copy()
            uperrs = autofill['starID'][target][p][lbl+'_uperr'].copy()
            lowerrs = autofill['starID'][target][p][lbl+'_lowerr'].copy()
            refs = autofill['starID'][target][p][lbl+'_ref'].copy()
            value,uperr,lowerr,ref = bestValue(values,uperrs,lowerrs,refs)
            # print('')
            # print(lbl)
            # print(value)
            # print(values)
            if value.__len__() > 0:
                out['priors'][p][lbl] = float(value)
                out['priors'][p][lbl+'_ref'] = ref
                err,autofilled = fillUncertainty(lbl,value,uperr,'uperr')
                if autofilled: out['autofill'].append(p+':'+lbl+'_uperr')
                out['priors'][p][lbl+'_uperr'] = err
                err,autofilled = fillUncertainty(lbl,value,lowerr,'lowerr')
                if autofilled: out['autofill'].append(p+':'+lbl+'_lowerr')
                out['priors'][p][lbl+'_lowerr'] = err
                strval = autofill['starID'][target][p][lbl+'_units'].copy()
                out['priors'][p][lbl+'_units'] = strval[-1]
                pass
            else:
                out['priors'][p][lbl] = ''
                for ext in out['exts']: out['priors'][p][lbl+ext] = ''
                out['needed'].append(p+':'+lbl)
                pass
            pass
        if p+':ecc' in out['needed']:
            out['priors'][p]['ecc'] = 0e0
            out['autofill'].append(p+':ecc')
            index = out['needed'].index(p+':ecc')
            out['needed'].pop(index)
            strval = autofill['starID'][target][p]['ecc_units'].copy()
            out['priors'][p]['ecc_units'] = strval[-1]
            out['priors'][p]['ecc_ref'] = 'System Prior Auto Fill'
            err,autofilled = fillUncertainty('ecc',0,'','uperr')
            out['priors'][p]['ecc_uperr'] = err
            if autofilled: out['autofill'].append(p+':ecc_uperr')
            err,autofilled = fillUncertainty('ecc',0,'','lowerr')
            out['priors'][p]['ecc_lowerr'] = err
            if autofilled: out['autofill'].append(p+':ecc_lowerr')
        pincnd = (p+':inc') in out['needed']
        psmain = (p+':sma') not in out['needed']
        rstarin = 'R*' not in out['needed']
        if pincnd and psmain and rstarin:
            values = autofill['starID'][target][p]['impact'].copy()
            uperrs = autofill['starID'][target][p]['impact_uperr'].copy()
            lowerrs = autofill['starID'][target][p]['impact_lowerr'].copy()
            refs = autofill['starID'][target][p]['impact_ref'].copy()
            value,uperr,lowerr,ref = bestValue(values,uperrs,lowerrs,refs)
            if value.__len__() > 0:
                # exit('THIS SHOULDNT BE NEEDED ANYMORE3')
                print('THIS SHOULDNT BE NEEDED ANYMORE3')
                sininc = float(value)*(out['priors']['R*'])*ssc['Rsun/AU']/(out['priors'][p]['sma'])
                inc = 9e1 - np.arcsin(sininc)*18e1/np.pi
                out['priors'][p]['inc'] = inc
                out['autofill'].append(p+':inc')
                index = out['needed'].index(p+':inc')
                out['needed'].pop(index)
                strval = autofill['starID'][target][p]['inc_units'].copy()
                out['priors'][p]['inc_units'] = strval[-1]
                out['priors'][p]['inc_ref'] = 'System Prior Auto Fill'
                fillValue,autofilled = fillUncertainty('inc',inc,'','uperr')
                out['priors'][p]['inc_uperr'] = fillValue
                if autofilled: out['autofill'].append(p+':inc_uperr')
                fillValue,autofilled = fillUncertainty('inc',inc,'','lowerr')
                out['priors'][p]['inc_lowerr'] = fillValue
                if autofilled: out['autofill'].append(p+':inc_lowerr')
                pass
            pass
        if p+':inc' in out['needed']:
            inc = 90e0
            out['priors'][p]['inc'] = inc
            out['autofill'].append(p+':inc')
            index = out['needed'].index(p+':inc')
            out['needed'].pop(index)
            out['priors'][p]['inc_units'] = '[degree]'
            out['priors'][p]['inc_ref'] = 'System Prior Auto Fill'
            fillValue,autofilled = fillUncertainty('inc',inc,'','uperr')
            out['priors'][p]['inc_uperr'] = fillValue
            if autofilled: out['autofill'].append(p+':inc_uperr')
            fillValue,autofilled = fillUncertainty('inc',inc,'','lowerr')
            out['priors'][p]['inc_lowerr'] = fillValue
            if autofilled: out['autofill'].append(p+':inc_lowerr')
        prpin = (p+':rp') not in out['needed']
        pmassin = (p+':mass') not in out['needed']
        if prpin and pmassin:
            # (setting of planet logg is no longer needed here; already done above)
            pass
        else: out['needed'].append(p+':logg')
        pass
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
    # print('needed,pneeded',out['needed'],out['pneeded'])
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
                # print('index',index)
                out['needed'].pop(index)
        pass
    # print('needed,pneeded',out['needed'],out['pneeded'])
    starneed = False
    for p in out['needed']:
        if ':' not in p: starneed = True
        pass
    if starneed or (len(out['priors']['planets']) < 1): out['PP'].append(True)
    log.warning('>-- FORCE PARAMETER: %s', str(out['PP'][-1]))
    log.warning('>-- MISSING MANDATORY PARAMETERS: %s', str(out['needed']))
    log.warning('>-- MISSING PLANET PARAMETERS: %s', str(out['pneeded']))
    log.warning('>-- PLANETS IGNORED: %s', str(out['ignore']))
    log.warning('>-- AUTOFILL: %s', str(out['autofill']))
    out['STATUS'].append(True)
    return True
# ------------------------- ------------------------------------------
# -- FORCE PRIOR PARAMETERS -- ---------------------------------------
def forcepar(overwrite, out):
    '''
G. ROUDIER: Completes/Overrides parameters using target/edit.py
    '''
    forced = True
    for key in overwrite.keys():
        mainkey = key.split(':')[0]
        if (mainkey not in out.keys()) and (len(mainkey) < 2):
            if mainkey in out['pignore'].keys():
                out['priors'][mainkey] = out['pignore'][mainkey].copy()
                pass
            for pkey in overwrite[key].keys():
                out['priors'][mainkey][pkey] = overwrite.copy()[mainkey][pkey]
                pass
            pass
        else: out['priors'][key] = overwrite.copy()[key]
        pass
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
                pass
            else:
                if p in pkey: addback = False
                pass
            pass
        if addback: out['priors']['planets'].append(p)
        pass
    starneed = False
    for p in out['needed']:
        if ':' not in p: starneed = True
        pass
    if starneed or (len(out['priors']['planets']) < 1):
        forced = False
        log.warning('>-- MISSING MANDATORY PARAMETERS')
        log.warning('>-- ADD THEM TO TARGET/EDIT.PY PPAR()')
        pass
    else:
        forced = True
        log.warning('>-- PRIORITY PARAMETERS SUCCESSFUL')
        pass
    return forced
# ---------------------------- ---------------------------------------
