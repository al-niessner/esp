# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import numpy as np
# ------------- ------------------------------------------------------
# -- SOLAR SYSTEM CONSTANTS -- ---------------------------------------
def ssconstants(mks=False, cgs=False):
    '''
IAU 2012
    '''
    if mks and cgs: ssc = {'Idiot':True}
    if not(mks or cgs):
        ssc = {'Rjup/Rsun':1.0276268506540176e-1,
               'Rsun/AU':4.650467260962158e-3}
        pass
    if mks:
        ssc = {'Rsun':6.957e8,
               'Msun':1.9884158605722263e32,
               'Rjup':7.1492e7,
               'Mjup':1.8985233541508517e27,
               'AU':1.495978707e11,
               'G':6.67428e-11,
               'Rjup/Rsun':1.0276268506540176e-1,
               'Rsun/AU':4.650467260962158e-3}
        pass
    if cgs:
        ssc = {'Rsun':6.957e10,
               'Msun':1.9884158605722263e35,
               'Rjup':7.1492e9,
               'Mjup':1.8985233541508517e30,
               'AU':1.495978707e13,
               'G':6.67428e-8,
               'Rjup/Rsun':1.0276268506540176e-1,
               'Rsun/AU':4.650467260962158e-3}
        pass
    return ssc
# ---------------------------- ---------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- BUILD SYSTEM PRIORS -- ------------------------------------------
def buildsp(autofill, out):
    target = [t for t in autofill['starID'].keys()]
    target = target[0]
    for p in autofill['starID'][target]['planets']: out['priors'][p] = {}
    out['priors']['planets'] = autofill['starID'][target]['planets'].copy()
    out['starkeys'].extend(autofill['starkeys'])
    out['planetkeys'].extend(autofill['planetkeys'])
    out['exts'].extend(autofill['exts'])
    out['starmdt'].extend(['R*', 'T*', 'FEH*', 'LOGG*'])
    out['planetmdt'].extend(['inc', 'period', 'ecc', 'rp', 't0', 'sma', 'mass'])
    out['extsmdt'].extend(['_lowerr', '_uperr'])
    for lbl in out['starmdt']:
        value = autofill['starID'][target][lbl].copy()
        value = value[-1]
        if value.__len__() > 0:
            out['priors'][lbl] = float(value)
            for ext in out['extsmdt']:
                value = autofill['starID'][target][lbl+ext].copy()
                value = value[-1]
                if value.__len__() > 0:
                    fillvalue = float(value)
                    if ('lowerr' in ext) and (fillvalue == 0):
                        fillvalue = out['priors'][lbl]*(-1e-1)
                        pass
                    if ('uperr' in ext) and (fillvalue == 0):
                        fillvalue = out['priors'][lbl]*(1e-1)
                        pass
                    out['priors'][lbl+ext] = fillvalue
                    pass
                else:
                    out['priors'][lbl+ext] = out['priors'][lbl]/1e1
                    out['autofill'].append(lbl+ext)
                    pass
                pass
            strval = autofill['starID'][target][lbl+'_units'].copy()
            strval = strval[-1]
            out['priors'][lbl+'_units'] = strval
            strval = autofill['starID'][target][lbl+'_ref'].copy()
            strval = strval[-1]
            out['priors'][lbl+'_ref'] = strval
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
        value = autofill['starID'][target]['RHO*'].copy()
        value = value[-1]
        if value.__len__() > 0:
            rho = float(value)
            g = 4e0*np.pi*(ssc['G'])*rho*radstar/3e0
            out['priors']['LOGG*'] = np.log10(g)
            out['autofill'].append('LOGG*')
            index = out['needed'].index('LOGG*')
            out['needed'].pop(index)
            for ext in out['extsmdt']:
                out['priors']['LOGG*'+ext] = np.log10(g)/1e1
                out['autofill'].append('LOGG*'+ext)
                pass
            strval = autofill['starID'][target]['LOGG*_units'].copy()
            strval = strval[-1]
            out['priors']['LOGG*_units'] = strval
            out['priors']['LOGG*_ref'] = 'System Prior Auto Fill'
            pass
        value = autofill['starID'][target]['M*'].copy()
        value = value[-1]
        if (value.__len__() > 0) and ('LOGG*' in out['needed']):
            mass = float(value)*ssc['Msun']
            g = (ssc['G'])*mass/(radstar**2)
            out['priors']['LOGG*'] = np.log10(g)
            out['autofill'].append('LOGG*')
            index = out['needed'].index('LOGG*')
            out['needed'].pop(index)
            for ext in out['extsmdt']:
                out['priors']['LOGG*'+ext] = np.log10(g)/1e1
                out['autofill'].append('LOGG*'+ext)
                pass
            strval = autofill['starID'][target]['LOGG*_units'].copy()
            strval = strval[-1]
            out['priors']['LOGG*_units'] = strval
            out['priors']['LOGG*_ref'] = 'System Prior Auto Fill'
            pass
        pass
    for p in out['priors']['planets']:
        for lbl in out['planetmdt']:
            value = autofill['starID'][target][p][lbl].copy()
            value = value[-1]
            if value.__len__() > 0:
                out['priors'][p][lbl] = float(value)
                for ext in out['extsmdt']:
                    value = autofill['starID'][target][p][lbl+ext].copy()
                    value = value[-1]
                    if value.__len__() > 0: out['priors'][p][lbl+ext] = float(value)
                    else:
                        err = out['priors'][p][lbl]/1e1
                        out['priors'][p][lbl+ext] = err
                        out['autofill'].append(p+':'+lbl+ext)
                        pass
                    pass
                strval = autofill['starID'][target][p][lbl+'_units'].copy()
                strval = strval[-1]
                out['priors'][p][lbl+'_units'] = strval
                strval = autofill['starID'][target][p][lbl+'_ref'].copy()
                strval = strval[-1]
                out['priors'][p][lbl+'_ref'] = strval
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
            strval = strval[-1]
            out['priors'][p]['ecc_units'] = strval
            out['priors'][p]['ecc_ref'] = 'System Prior Auto Fill'
            for ext in out['extsmdt']:
                err = 1e-10
                out['priors'][p]['ecc'+ext] = err
                out['autofill'].append(p+':ecc'+ext)
                pass
            pass
        pincnd = (p+':inc') in out['needed']
        psmain = (p+':sma') not in out['needed']
        rstarin = 'R*' not in out['needed']
        if pincnd and psmain and rstarin:
            value = autofill['starID'][target][p]['impact'].copy()
            value = value[-1]
            if value.__len__() > 0:
                sininc = float(value)*(out['priors']['R*'])*ssc['Rsun/AU']/(out['priors'][p]['sma'])
                inc = 9e1 - np.arcsin(sininc)*18e1/np.pi
                out['priors'][p]['inc'] = inc
                out['autofill'].append(p+':inc')
                index = out['needed'].index(p+':inc')
                out['needed'].pop(index)
                strval = autofill['starID'][target][p]['inc_units'].copy()
                strval = strval[-1]
                out['priors'][p]['inc_units'] = strval
                out['priors'][p]['inc_ref'] = 'System Prior Auto Fill'
                for ext in out['extsmdt']:
                    err = inc/1e1
                    if inc + err > 9e1: err = (9e1 - inc)
                    out['priors'][p]['inc'+ext] = err
                    out['autofill'].append(p+':inc'+ext)
                    pass
                pass
            pass
        if p+':inc' in out['needed']:
            inc = 90e0
            out['priors'][p]['inc'] = inc
            out['autofill'].append(p+':inc')
            index = out['needed'].index(p+':inc')
            out['needed'].pop(index)
            out['priors'][p]['inc_units'] = ['[degree]']
            out['priors'][p]['inc_ref'] = 'System Prior Auto Fill'
            for ext in out['extsmdt']:
                err = inc/1e1
                if inc + err > 9e1: err = (9e1 - inc)
                out['priors'][p]['inc'+ext] = err
                out['autofill'].append(p+':inc'+ext)
                pass
            pass
        prpin = (p+':rp') not in out['needed']
        pmassin = (p+':mass') not in out['needed']
        if prpin and pmassin:
            radplanet = out['priors'][p]['rp']*ssc['Rjup']
            mass = out['priors'][p]['mass']*ssc['Mjup']
            g = (ssc['G'])*mass/(radplanet**2)
            out['priors'][p]['logg'] = np.log10(g)
            out['autofill'].append(p+':logg')
            for ext in out['extsmdt']:
                out['priors'][p]['logg'+ext] = np.log10(g)/1e1
                out['autofill'].append(p+':logg'+ext)
                pass
            out['priors'][p]['logg_units'] = 'log10[cm.s-2]'
            out['priors'][p]['logg_ref'] = 'System Prior Auto Fill'
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
    for p in out['ignore']:
        for value in out['needed']:
            index = out['needed'].index(value)
            if p+':' in value:
                out['needed'].pop(index)
                out['pneeded'].append(value)
                pass
            pass
        pass
    starneed = False
    for p in out['needed']:
        if ':' not in p: starneed = True
        pass
    if starneed or (len(out['priors']['planets']) < 1): out['PP'].append(True)
    log.log(31, '>-- FORCE PARAMETER: %s', str(out['PP'][-1]))
    log.log(31, '>-- MISSING MANDATORY PARAMETERS: %s', str(out['needed']))
    log.log(31, '>-- MISSING PLANET PARAMETERS: %s', str(out['pneeded']))
    log.log(31, '>-- PLANETS IGNORED: %s', str(out['ignore']))
    log.log(31, '>-- AUTOFILL: %s', str(out['autofill']))
    out['STATUS'].append(True)
    return True
# ------------------------- ------------------------------------------
# -- FORCE PRIOR PARAMETERS -- ---------------------------------------
def forcepar(overwrite, out):
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
        for pkey in out['pneeded']:
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
        log.log(31, '>-- MISSING MANDATORY PARAMETERS')
        log.log(31, '>-- ADD THEM TO TARGET/EDIT.PY PPAR()')
        pass
    else:
        forced = True
        log.log(31, '>-- PRIORITY PARAMETERS SUCCESSFUL')
        pass
    return forced
# ---------------------------- ---------------------------------------
