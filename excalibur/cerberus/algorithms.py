'''cerberus algorithms ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import logging; log = logging.getLogger(__name__)

import excalibur.system as sys
import excalibur.system.algorithms as sysalg
import excalibur.taurex as tau
import excalibur.taurex.algorithms as taualg
import excalibur.transit as trn
import excalibur.transit.algorithms as trnalg
import excalibur.target.edit as trgedit
from excalibur import ariel
import excalibur.ariel.algorithms as arielalg
import excalibur.cerberus as crb
import excalibur.cerberus.core as crbcore
import excalibur.cerberus.states as crbstates

# ------------- ------------------------------------------------------
# -- ALGOS RUN OPTIONS -- --------------------------------------------
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
fltrs.append('Ariel-sim')

# ----------------------- --------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class xslib(dawgie.Algorithm):
    '''Cross Section Library'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = crbcore.myxsecsversion()
        self.__spc = trnalg.spectrum()
        self.__tau = taualg.TransitSpectrumInjection()
        self.__arielsim = arielalg.sim_spectrum()
        self.__out = [crbstates.xslibSV(ext) for ext in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'xslib'

    def previous(self):
        '''Input State Vectors: transit.spectrum'''
        return [dawgie.ALG_REF(trn.task, self.__spc),
                dawgie.ALG_REF(tau.task, self.__tau),
                dawgie.ALG_REF(ariel.task, self.__arielsim)]

    def state_vectors(self):
        '''Output State Vectors: cerberus.xslib'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''
        svupdate = []
        for ext in fltrs:
            update = False

            if ext in self.__tau.sv_as_dict():
                sv = self.__tau.sv_as_dict()[ext]
                vspc, sspc = crbcore.checksv(sv)
            else:
                vspc = False

            if not vspc:
                if ext=='Ariel-sim':
                    sv = self.__arielsim.sv_as_dict()['parameters']
                    vspc, sspc = crbcore.checksv(sv)
                else:
                    if ext in self.__spc.sv_as_dict().keys():
                        sv = self.__spc.sv_as_dict()[ext]
                        vspc, sspc = crbcore.checksv(sv)
                    else:
                        vspc = False
                        sspc = 'This filter doesnt have a spectrum (no data?): ' + ext

            # pylint: disable=protected-access
            prcd = trgedit.proceed(ds._tn(), ext, verbose=False)
            if vspc and prcd:
                log.warning('--< CERBERUS XSLIB: %s >--', ext)
                update = self._xslib(sv, fltrs.index(ext))
            else:
                errstr = [m for m in [sspc] if m is not None]
                if not prcd: errstr = [ext + ' Kicked by edit.processme()']
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[fltrs.index(ext)])
            pass
        self.__out = svupdate

        # clear out the False STATUS flags
        #  this helped during debugging; not sure it is still necessary
        # for i in range(len(self.__out)):
        #     self.__out[i]['STATUS'].remove(False)

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
        self.__tau = taualg.TransitSpectrumInjection()
        self.__arielsim = arielalg.sim_spectrum()
        self.__out = [crbstates.atmosSV(ext) for ext in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'atmos'

    def previous(self):
        '''Input State Vectors: transit.spectrum, system.finalize, cerberus.xslib'''
        return [dawgie.ALG_REF(trn.task, self.__spc),
                dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(crb.task, self.__xsl),
                dawgie.ALG_REF(tau.task, self.__tau),
                dawgie.ALG_REF(ariel.task, self.__arielsim)]

    def state_vectors(self):
        '''Output State Vectors: cerberus.atmos'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''
        svupdate = []
        vfin, sfin = crbcore.checksv(self.__fin.sv_as_dict()['parameters'])
        if sfin: sfin = 'Missing system params!'

        for ext in fltrs:
            update = False
            vxsl, sxsl = crbcore.checksv(self.__xsl.sv_as_dict()[ext])
            if sxsl: sxsl = ext + ' missing XSL'

            if ext in self.__tau.sv_as_dict():
                sv = self.__tau.sv_as_dict()[ext]
                vspc, sspc = crbcore.checksv(sv)
            else: vspc = False

            if not vspc:
                if ext=='Ariel-sim':
                    sv = self.__arielsim.sv_as_dict()['parameters']
                    vspc, sspc = crbcore.checksv(sv)
                else:
                    if ext in self.__spc.sv_as_dict().keys():
                        sv = self.__spc.sv_as_dict()[ext]
                        vspc, sspc = crbcore.checksv(sv)
                    else:
                        vspc = False
                        sspc = 'This filter doesnt seem to exist: ' + ext

            # pylint: disable=protected-access
            prcd = trgedit.proceed(ds._tn(), ext, verbose=False)
            if vfin and vxsl and vspc and prcd:
                log.warning('--< CERBERUS ATMOS: %s >--', ext)
                update = self._atmos(self.__fin.sv_as_dict()['parameters'],
                                     self.__xsl.sv_as_dict()[ext],
                                     sv, fltrs.index(ext), ext)
            else:
                if not (vfin and vxsl and vspc):
                    errstr = [m for m in [sfin, sspc, sxsl] if m is not None]
                elif not prcd:
                    errstr = [ext + ' Kicked by edit.processme()']
                else:
                    errstr = ['HUH?! BAD LOGIC HERE']
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
        if ext=='Ariel-sim':
            MCMC_chain_length = 1000
            MCMC_chain_length = 5000
            MCMC_chain_length = 2000
            # MCMC_chain_length = 10
        else:
            MCMC_chain_length = 15000
            MCMC_chain_length = 200
        print(' calling atmos from cerb-alg-atmos  chain len=',MCMC_chain_length)
        import os  # because this is for testing only, pylint: disable=import-outside-toplevel
        import time  # because this is for testing only, pylint: disable=import-outside-toplevel
        os.environ['NUMEXPR_NUM_THREADS'] = "1"
        for MCMC_chain_length in [200, 400, 800]:
            t0 = time.time()
            am = crbcore.atmos(fin, xsl, spc, self.__out[index], ext,
                               mclen=MCMC_chain_length,
                               sphshell=True, verbose=False)  # singlemod='TEC' after mclen
            log.info('It took %f8.2 seconds for %d chain length', time.time() - t0, MCMC_chain_length)
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

class results(dawgie.Algorithm):
    '''
    Plot the best-fit spectrum, to see how well it fits the data
    Plot the corner plot, to see how well each parameter is constrained
    '''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = crbcore.resultsversion()
        self.__fin = sysalg.finalize()
        self.__xsl = xslib()
        self.__atm = atmos()
        self.__out = [crbstates.resSV(filt) for filt in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'results'

    def previous(self):
        '''Input State Vectors: cerberus.atmos'''
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(crb.task, self.__xsl),
                dawgie.ALG_REF(crb.task, self.__atm)]

    def state_vectors(self):
        '''Output State Vectors: cerberus.results'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''

        svupdate = []
        vfin, sfin = crbcore.checksv(self.__fin.sv_as_dict()['parameters'])

        # filts = ['HST-WFC3-IR-G141-SCAN']
        # filts = ['Ariel-sim']
        # filts = ['HST-WFC3-IR-G141-SCAN', 'Ariel-sim']

        update = False
        if vfin:
            # available_filters = self.__xsl.sv_as_dict().keys()
            # print('available_filters',available_filters)
            available_filters = self.__atm.sv_as_dict().keys()
            # print('available_filters',available_filters)

            # for filt in filts:
            for filt in available_filters:
                log.warning('--< CERBERUS RESULTS: %s >--', filt)
                # pylint: disable=protected-access
                vxsl, _ = crbcore.checksv(self.__xsl.sv_as_dict()[filt])
                vatm, satm = crbcore.checksv(self.__atm.sv_as_dict()[filt])
                if vxsl and vatm:
                    update = self._results(ds._tn(), filt,
                                           self.__fin.sv_as_dict()['parameters'],
                                           self.__xsl.sv_as_dict()[filt]['data'],
                                           self.__atm.sv_as_dict()[filt]['data'],
                                           fltrs.index(filt))
                    if update: svupdate.append(self.__out[fltrs.index(filt)])
                else:
                    errstr = [m for m in [satm] if m is not None]
                    self._failure(errstr[0])
        else:
            errstr = [m for m in [sfin] if m is not None]
            self._failure(errstr[0])

        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for CERBERUS.{self.name()}')
        return

    def _results(self, trgt, filt, fin, xsl, atm, index):
        '''Core code call'''
        resout = crbcore.results(trgt, filt, fin, xsl, atm,
                                 self.__out[index], verbose=False)
        return resout

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< CERBERUS RESULTS: %s >--', errstr)
        return
    pass
# ---------------- ---------------------------------------------------

class analysis(dawgie.Analyzer):
    '''analysis ds'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = crbcore.resultsversion()  # same version number as results
        # self.__fin = sysalg.finalize()
        # self.__xsl = xslib()
        # self.__atm = atmos()
        # self.__out = crbstates.analysisSV('retrievalCheck')
        self.__out = [crbstates.analysisSV(filt) for filt in fltrs]
        return

    # def previous(self):
    #    '''Input State Vectors: cerberus.atmos'''
    #    return [dawgie.ALG_REF(sys.task, self.__fin),
    #            dawgie.ALG_REF(crb.task, self.__xsl),
    #            dawgie.ALG_REF(crb.task, self.__atm)]

    def feedback(self):
        '''feedback ds'''
        return []

    def name(self):
        '''Database name for subtask extension'''
        return 'analysis'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        '''traits ds'''
        return [dawgie.SV_REF(crb.task, atmos(),
                              atmos().state_vectors()[0])]

    def state_vectors(self):
        '''Output State Vectors: cerberus.analysis'''
        return self.__out

    def run(self, aspects:dawgie.Aspect):
        '''Top level algorithm call'''
        filts = ['Ariel-sim']
        svupdate = []
        for filt in filts:
            update = self._analysis(aspects, fltrs.index(filt))
            if update: svupdate.append(self.__out[fltrs.index(filt)])
        self.__out = svupdate
        if self.__out.__len__() > 0: aspects.ds().update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for CERBERUS.{self.name()}')
        return

    def _analysis(self, aspects, index):
        '''Core code call'''
        analysisout = crbcore.analysis(aspects,
                                       self.__out[index], verbose=False)
        return analysisout

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< CERBERUS ANALYSIS: %s >--', errstr)
        return

# -------------------------------------------------------------------
