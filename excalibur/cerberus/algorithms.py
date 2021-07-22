# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import logging; log = logging.getLogger(__name__)

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
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
fltrs.append('STIS-WFC3')
fltrs = [f for f in fltrs if 'Spitzer' not in f]
# ----------------------- --------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class xslib(dawgie.Algorithm):
    def __init__(self):
        self._version_ = crbcore.myxsecsversion()
        self.__spc = trnalg.spectrum()
        self.__out = [crbstates.xslibSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'xslib'

    def previous(self):
        return [dawgie.ALG_REF(trn.task, self.__spc)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        for ext in fltrs:
            update = False
            vspc, sspc = crbcore.checksv(self.__spc.sv_as_dict()[ext])
            if vspc:
                log.warning('--< CERBERUS XSLIB: %s >--', ext)
                update = self._xslib(self.__spc.sv_as_dict()[ext], fltrs.index(ext))
                pass
            else:
                errstr = [m for m in [sspc] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[fltrs.index(ext)])
            pass
        self.__out = svupdate
        if self.__out: ds.update()
        return

    def _xslib(self, spc, index):
        cs = crbcore.myxsecs(spc, self.__out[index], verbose=False)
        return cs

    @staticmethod
    def _failure(errstr):
        log.warning('--< CERBERUS XSLIB: %s >--', errstr)
        return
    pass

class atmos(dawgie.Algorithm):
    def __init__(self):
        self._version_ = crbcore.atmosversion()
        self.__spc = trnalg.spectrum()
        self.__fin = sysalg.finalize()
        self.__xsl = xslib()
        self.__out = [crbstates.atmosSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'atmos'

    def previous(self):
        return [dawgie.ALG_REF(trn.task, self.__spc),
                dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(crb.task, self.__xsl)]

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
                log.warning('--< CERBERUS ATMOS: %s >--', ext)
                update = self._atmos(self.__fin.sv_as_dict()['parameters'],
                                     self.__xsl.sv_as_dict()[ext],
                                     self.__spc.sv_as_dict()[ext],
                                     fltrs.index(ext), ext)
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

    def _atmos(self, fin, xsl, spc, index, ext):
        am = crbcore.atmos(fin, xsl, spc, self.__out[index], ext,
                           mclen=int(15e3),
                           sphshell=True, verbose=False)  # singlemod='TEC' after mclen
        return am

    @staticmethod
    def _failure(errstr):
        log.warning('--< CERBERUS ATMOS: %s >--', errstr)
        return
    pass
# ---------------- ---------------------------------------------------
