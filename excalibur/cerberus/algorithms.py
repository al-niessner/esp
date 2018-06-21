# -- IMPORTS -- ------------------------------------------------------
import os
import sys
import pdb

import dawgie
import dawgie.context

import exo.spec.ae.system as sys
import exo.spec.ae.system.algorithms as sysalg
#import exo.spec.ae.extrasolar as makepriors
#import exo.spec.ae.extrasolar.algorithms as mkpalg

import exo.spec.ae.cerberus as cerberus
import exo.spec.ae.cerberus.states as crbstates
import exo.spec.ae.cerberus.core as crb
# ------------- ------------------------------------------------------
# -- ALGOS RUN OPTIONS -- --------------------------------------------
# PLOTS
verbose = False
debug = False
# MCMC
resolution = 'high'
mclen = int(4e4) # 4e4
eclipse = False
# FINESSE
finesse = False
rfnslist = ['R10', 'R20', 'R30', 'R40','R50',
            'R60', 'R70', 'R80', 'R90', 'R100']
# VALIDATION
atmosmc = False
lbroadening = False
lshifting = False
# DEACTIVATE TASKS ENABLE VIEW
dtev = False
atmosdtev = False
# ----------------------- --------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
# -- CROSS SECTION LIBRARY
class xslib(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,5,1)
        self.__mcmc = sysalg.force()
        self.__priors = sysalg.prior()
        self.__xslib = crbstates.xslibSV()
        self.__fxslib = crbstates.xslibSV('FINESSE')
        self.__jwsxslib = crbstates.xslibSV('JWST_sims')
        return
    
    def name(self):
        return 'xslib'
    def previous(self):
        return [dawgie.ALG_REF(sys.factory, self.__mcmc),
                dawgie.ALG_REF(sys.factory, self.__priors)]
    def state_vectors(self):
        return [self.__xslib, self.__fxslib, self.__jwsxslib]
    def run(self, ds, ps):
        if False:
            priors = self.__priors.sv_as_dict()['parameters']
            if eclipse:
                mcmc = self.__mcmc.sv_as_dict()['ecl_ima']
                jwsmcmc = self.__mcmc.sv_as_dict()['ecl_JWST_sims']
                pass
            else:
                mcmc = self.__mcmc.sv_as_dict()['ima']
                jwsmcmc = self.__mcmc.sv_as_dict()['JWST_sims']
                pass
            valid, errcode = crb.checksvin(mcmc)
            jwsvalid, jwserrcode = crb.checksvin(jwsmcmc)        
            if dtev:
                valid = False
                jwsvalid = False
                pass
            # WFC3 SCAN
            update = False
            if valid:
                out = self.__xslib
                update = self._xslib(mcmc, priors, out)
                pass
            else: self._failure(errcode)
            if update: ds.update()
            # JWST SIM
            update = False
            if jwsvalid:
                out = self.__jwsxslib
                update = self._xslib(jwsmcmc, priors, out)
                pass
            else: self._failure(errcode)
            if update: ds.update()
            pass
        return
    
    def _failure(self, errcode, verbose=verbose):
        errstr = crb.errcode2errstr(errcode)
        errmess = ' '.join(('CERBERUS.xslib:', errstr))
        if verbose: print(errmess)
        return
    def _xslib(self, mcmc, priors, out):
        resollist = [res for res in mcmc.keys() if (len(mcmc[res]) > 0)]
        for myresol in resollist:
            crb.xsecs(mcmc, priors, out,
                      res=myresol, finesse=False,
                      verbose=verbose, debug=debug)
            pass
        if finesse:
            for myresol in rfnslist:
                crb.xsecs(mcmc, priors, self.__fxslib,
                          res=myresol, finesse=True,
                          verbose=verbose, debug=debug)
                pass
            pass
        return True
    pass

# -- HAZE DENSITY PROFILE LIBRARY
class hazelib(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,2,1)
        self.__mcmc = lzzalg.mcmc()
        self.__priors = mkpalg.NBB()
        self.__hzlib = crbstates.hazelibSV()
        self.__jwshzlib = crbstates.hazelibSV('JWST_sims')
        return
    
    def name(self):
        return 'hazelib'
    def previous(self):
        return [dawgie.ALG_REF(mcmclzz.factory, self.__mcmc),
                dawgie.ALG_REF(makepriors.factory, self.__priors)]
    def state_vectors(self):
        return [self.__hzlib, self.__jwshzlib]
    def run(self, ds, ps):
        priors = self.__priors.sv_as_dict()['parameters']
        mcmc = self.__mcmc.sv_as_dict()['ima']
        jwsmcmc = self.__mcmc.sv_as_dict()['JWST_sims']
        valid, errcode = crb.checksvin(mcmc)
        jwsvalid, jwserrcode = crb.checksvin(jwsmcmc)
        if dtev:
            valid = False
            jwsvalid = False
            pass
        update = False
        if valid:
            out = self.__hzlib
            update = self._hazelib(mcmc, priors, out)
            pass
        else: self._failure(errcode)
        if update: ds.update()
        update = False
        if jwsvalid:
            out = self.__jwshzlib
            update = self._hazelib(jwsmcmc, priors, out)
        else: self._failure(errcode)
        if update: ds.update()
        return
    
    def _failure(self, errcode):
        errstr = crb.errcode2errstr(errcode)
        errmess = ' '.join(('CERBERUS.hazelib:', errstr))
        if verbose: print(errmess)
        return
    def _hazelib(self, mcmc, priors, out):
        crb.hzlib(mcmc, priors, out,
                  verbose=verbose, debug=debug)
        return True
    pass

# -- ABUNDANCES RETRIEVAL
class atmos(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,5,1)
        self.__mcmc = lzzalg.mcmc()
        self.__priors = mkpalg.NBB()
        self.__xslib = xslib()
        self.__hzlib = hazelib()
        self.__atmos = crbstates.atmosSV()
        self.__fatmos = crbstates.atmosSV('FINESSE')
        self.__jwsatmos = crbstates.atmosSV('JWST_sims')
        return
    
    def name(self):
        return 'atmos'
    def previous(self):
        return [dawgie.ALG_REF(mcmclzz.factory, self.__mcmc),
                dawgie.ALG_REF(makepriors.factory, self.__priors),
                dawgie.ALG_REF(cerberus.factory, self.__xslib),
                dawgie.ALG_REF(cerberus.factory, self.__hzlib)]
    def state_vectors(self):
        return [self.__atmos, self.__fatmos, self.__jwsatmos]
    def run(self, ds, ps):
        priors = self.__priors.sv_as_dict()['parameters']
        fxslib = self.__xslib.sv_as_dict()['FINESSE']
        mcmc = self.__mcmc.sv_as_dict()['ima']
        xslib = self.__xslib.sv_as_dict()['lines']
        hzlib = self.__hzlib.sv_as_dict()['vdensity']
        jwsmcmc = self.__mcmc.sv_as_dict()['JWST_sims']        
        jwsxslib = self.__xslib.sv_as_dict()['JWST_sims']
        jwshzlib = self.__hzlib.sv_as_dict()['JWST_sims']
        valid, errcode = crb.checksvin(mcmc)
        jwsvalid, jwserrcode = crb.checksvin(jwsmcmc)
        if dtev or atmosdtev:
            valid = False
            jwsvalid = False
            pass
        update = False
        if valid:
            out = self.__atmos
            update = self._atmos(mcmc, priors,
                                 xslib, hzlib, fxslib, out)
            pass
        else: self._failure(errcode)
        if update: ds.update()
        update = False
        if jwsvalid:
            out = self.__jwsatmos
            update = self._atmos(jwsmcmc, priors,
                                 jwsxslib, jwshzlib, fxslib, out)
            pass
        else: self._failure(errcode)
        if update: ds.update()
        return
    
    def _failure(self, errcode, verbose=verbose):
        errstr = crb.errcode2errstr(errcode)
        errmess = ' '.join(('CERBERUS.atmos:', errstr))
        if verbose: print(errmess)
        return
    def _atmos(self, mcmc, priors, xslib, hzlib, fxslib, out):
        resollist = xslib['RES']
        xsecs = xslib['XSECS'][resollist.index(resolution)]
        qtgrid = xslib['QTGRID'][resollist.index(resolution)]
        crb.atmos(mcmc, priors, xsecs, qtgrid, hzlib, out,
                  lbroadening=lbroadening, lshifting=lshifting,
                  res=resolution, mclen=mclen, atmosmc=atmosmc,
                  verbose=verbose, debug=debug)
        return True
    pass
# ---------------- ---------------------------------------------------
