# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import excalibur.cerberus as crb
import excalibur.cerberus.core as crbcore
import excalibur.cerberus.states as crbstates

import excalibur.system as sys
import excalibur.system.algorithms as sysalg
import excalibur.transit as trn
import excalibur.transit.algorithms as trnalg
import excalibur.target.edit as trgedit
# ------------- ------------------------------------------------------
# -- ALGOS RUN OPTIONS -- --------------------------------------------
# VERBOSE AND DEBUG
verbose = False
debug = False
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if len(t.replace(' ', '')) > 0]
# ----------------------- --------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class xslib(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__verbose = verbose
        self.__spc = trnalg.spectrum()
        self.__fin = sysalg.finalize()
        self.__out = [crbstates.xslibSV(ext) for ext in fltrs]        
        return

    def name(self):
        return 'xslib'
    
    def previous(self):
        return [dawgie.ALG_REF(trn.factory, self.__spc),
                dawgie.ALG_REF(sys.factory, self.__fin)]

    def state_vectors(self):
        return self.__out
    
    def run(self, ds, ps):
        svupdate = []
        vfin, sfin = crbcore.checksv(self.__fin.sv_as_dict()['parameters'])
        for ext in fltrs:
            update = False
            vspc, sspc = crbcore.checksv(self.__spc.sv_as_dict()[ext])
            if vspc and vfin:
                update = self._xslib(self.__spc.sv_as_dict()[ext],
                                     self.__fin.sv_as_dict()['parameters'],
                                     fltrs.index(ext))
                pass
            else:
                errstr = [m for m in [sspc, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[fltrs.index(ext)])
            pass
        return

    def _xslib(self, spc, fin, index):
        cs = crbcore.xsecs(spc, fin, self.__out[index],
                           verbose=self.__verbose, debug=debug)
        return cs

    def _failure(self, errstr):
        errmess = '--< CERBERUS XSLIB: ' + errstr + ' >--'
        if self.__verbose: print(errmess)
        return
    pass

# -- HAZE DENSITY PROFILE LIBRARY
class hazelib(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,2,1)
        self.__mcmc = trnalg.whitelight()
        self.__priors = sysalg.finalize()
        self.__hzlib = crbstates.hazelibSV()
        self.__jwshzlib = crbstates.hazelibSV('JWST_sims')
        return

    def name(self):
        return 'hazelib'
    def previous(self):
        return [dawgie.ALG_REF(trn.factory, self.__mcmc),
                dawgie.ALG_REF(sys.factory, self.__priors)]
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
        self.__mcmc = trnalg.whitelight()
        self.__priors = sysalg.finalize()
        self.__xslib = xslib()
        self.__hzlib = hazelib()
        self.__atmos = crbstates.atmosSV()
        self.__fatmos = crbstates.atmosSV('FINESSE')
        self.__jwsatmos = crbstates.atmosSV('JWST_sims')
        return

    def name(self):
        return 'atmos'
    def previous(self):
        return [dawgie.ALG_REF(trn.factory, self.__mcmc),
                dawgie.ALG_REF(sys.factory, self.__priors),
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
