'''cerberus algorithms ds'''
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
# fltrs = [f for f in fltrs if 'STIS-WFC3' in f]
fltrs = [f for f in fltrs if 'Spitzer' not in f]
# ----------------------- --------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class xslib(dawgie.Algorithm):
    '''Cross Section Library'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = crbcore.myxsecsversion()
        self.__spc = trnalg.spectrum()
        self.__out = [crbstates.xslibSV(ext) for ext in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'xslib'

    def previous(self):
        '''Input State Vectors: transit.spectrum'''
        return [dawgie.ALG_REF(trn.task, self.__spc)]

    def state_vectors(self):
        '''Output State Vectors: cerberus.xslib'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''
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
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for CERBERUS.{self.name()}')
        return

    def _xslib(self, spc, index):
        '''Core code call'''
        cs = crbcore.myxsecs(spc, self.__out[index], verbose=False)
        return cs

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< CERBERUS XSLIB: %s >--', errstr)
        return
    pass

class atmos(dawgie.Algorithm):
    '''Atmospheric retrievial'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = crbcore.atmosversion()
        self.__spc = trnalg.spectrum()
        self.__fin = sysalg.finalize()
        self.__xsl = xslib()
        self.__out = [crbstates.atmosSV(ext) for ext in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'atmos'

    def previous(self):
        '''Input State Vectors: transit.spectrum, system.finalize, cerberus.xslib'''
        return [dawgie.ALG_REF(trn.task, self.__spc),
                dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(crb.task, self.__xsl)]

    def state_vectors(self):
        '''Output State Vectors: cerberus.atmos'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''
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
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for CERBERUS.{self.name()}')
        return

    def _atmos(self, fin, xsl, spc, index, ext):
        '''Core code call'''
        am = crbcore.atmos(fin, xsl, spc, self.__out[index], ext,
                           mclen=int(15e3),
                           sphshell=True, verbose=False)  # singlemod='TEC' after mclen
        return am

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< CERBERUS ATMOS: %s >--', errstr)
        return
    pass

class release(dawgie.Algorithm):
    '''Format release products Roudier et al. 2021'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = crbcore.rlsversion()
        self.__fin = sysalg.finalize()
        self.__atmos = atmos()
        self.__out = [crbstates.rlsSV(ext) for ext in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'release'

    def previous(self):
        '''Input State Vectors: cerberus.atmos'''
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(crb.task, self.__atmos)]

    def state_vectors(self):
        '''Output State Vectors: cerberus.release'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''
        svupdate = []
        vfin, sfin = crbcore.checksv(self.__fin.sv_as_dict()['parameters'])
        ext = 'HST-WFC3-IR-G141-SCAN'
        update = False
        if vfin:
            log.warning('--< CERBERUS RELEASE: %s >--', ext)
            # pylint: disable=protected-access
            update = self._release(ds._tn(),
                                   self.__fin.sv_as_dict()['parameters'],
                                   fltrs.index(ext))
            pass
        else:
            errstr = [m for m in [sfin] if m is not None]
            self._failure(errstr[0])
            pass
        if update: svupdate.append(self.__out[fltrs.index(ext)])
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for CERBERUS.{self.name()}')
        return

    def _release(self, trgt, fin, index):
        '''Core code call'''
        rlsout = crbcore.release(trgt, fin, self.__out[index], verbose=False)
        return rlsout

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< CERBERUS RELEASE: %s >--', errstr)
        return
    pass
# ---------------- ---------------------------------------------------
