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
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
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

class atmos(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__verbose = verbose
        self.__spc = trnalg.spectrum()
        self.__fin = sysalg.finalize()
        self.__xsl = xslib()
        self.__out = [crbstates.atmosSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'atmos'

    def previous(self):
        return [dawgie.ALG_REF(trn.factory, self.__spc),
                dawgie.ALG_REF(sys.factory, self.__fin),
                dawgie.ALG_REF(crb.factory, self.__xsl)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        vfin, sfin = crbcore.checksv(self.__fin.sv_as_dict()['parameters'])
        for ext in fltrs:
            update = False
            vxsl, sxsl = crbcore.checksv(self.__xsl.sv_as_dict()[ext])
            vspc, sspc = crbcore.checksv(self.__spc.sv_as_dict()[ext])
            if vfin and vxsl and vspc:
                update = self._atmos(self.__fin.sv_as_dict()['parameters'],
                                     self.__xsl.sv_as_dict()[ext],
                                     self.__spc.sv_as_dict()[ext],
                                     fltrs.index(ext))
                pass
            else:
                errstr = [m for m in [sfin, sxsl, sspc] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[fltrs.index(ext)])
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        return

    def _atmos(self, fin, xsl, spc, index):
        am = crb.atmos(fin, xsl, spc, self.__out[index],
                       verbose=self.__verbose, debug=debug)
        return am

    def _failure(self, errstr):
        errmess = '--< CERBERUS ATMOS: ' + errstr + ' >--'
        if self.__verbose: print(errmess)
        return
    pass
# ---------------- ---------------------------------------------------
