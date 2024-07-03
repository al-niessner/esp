'''cerberus algorithms ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import numexpr; numexpr.ncores = 1  # this is actually a performance enhancer!

from collections import namedtuple
import logging; log = logging.getLogger(__name__)

import excalibur.system as sys
import excalibur.system.algorithms as sysalg
import excalibur.ancillary as anc
import excalibur.ancillary.algorithms as ancillaryalg
import excalibur.runtime as rtime
import excalibur.runtime.algorithms as rtalg
import excalibur.runtime.binding as rtbind
import excalibur.taurex as tau
import excalibur.taurex.algorithms as taualg
import excalibur.transit as trn
import excalibur.transit.algorithms as trnalg
from excalibur import ariel
import excalibur.ariel.algorithms as arielalg
import excalibur.cerberus as crb
import excalibur.cerberus.core as crbcore
import excalibur.cerberus.states as crbstates

fltrs = [str(fn) for fn in rtbind.filter_names.values()]

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
        self.__rt = rtalg.autofill()
        self.__out = [crbstates.xslibSV(fltr) for fltr in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'xslib'

    def previous(self):
        '''Input State Vectors: transit.spectrum'''
        return [dawgie.ALG_REF(trn.task, self.__spc),
                dawgie.ALG_REF(tau.task, self.__tau),
                dawgie.ALG_REF(ariel.task, self.__arielsim)] + \
                self.__rt.refs_for_proceed()

    def state_vectors(self):
        '''Output State Vectors: cerberus.xslib'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''

        svupdate = []
        for fltr in self.__rt.sv_as_dict()['status']['allowed_filter_names']:
            # stop here if it is not a runtime target
            self.__rt.proceed(fltr)

            update = False

            if fltr in self.__tau.sv_as_dict():
                sv = self.__tau.sv_as_dict()[fltr]
                vspc, sspc = crbcore.checksv(sv)
            else:
                vspc = False

            if not vspc:
                if fltr=='Ariel-sim':
                    sv = self.__arielsim.sv_as_dict()['parameters']
                    vspc, sspc = crbcore.checksv(sv)
                    sspc = 'Ariel-sim spectrum not found'
                else:
                    if fltr in self.__spc.sv_as_dict().keys():
                        sv = self.__spc.sv_as_dict()[fltr]
                        vspc, sspc = crbcore.checksv(sv)
                    else:
                        vspc = False
                        sspc = 'This filter doesnt have a spectrum: ' + fltr

            if vspc:
                log.warning('--< CERBERUS XSLIB: %s >--', fltr)
                update = self._xslib(sv, fltrs.index(fltr))
            else:
                errstr = [m for m in [sspc] if m is not None]
                self._failure(errstr[0])

            if update: svupdate.append(self.__out[fltrs.index(fltr)])
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
    # need lots of state info; pylint: disable=too-many-instance-attributes
    def __init__(self):
        '''__init__ ds'''
        self._version_ = crbcore.atmosversion()
        self.__spc = trnalg.spectrum()
        self.__fin = sysalg.finalize()
        self.__xsl = xslib()
        self.__tau = taualg.TransitSpectrumInjection()
        self.__arielsim = arielalg.sim_spectrum()
        self.__rt = rtalg.autofill()
        self.__out = [crbstates.atmosSV(fltr) for fltr in fltrs]
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
                dawgie.ALG_REF(ariel.task, self.__arielsim),
                dawgie.V_REF(rtime.task, self.__rt, self.__rt.sv_as_dict()['status'],
                             'cerberus_steps'),
                dawgie.V_REF(rtime.task, self.__rt, self.__rt.sv_as_dict()['status'],
                             'cerberus_atmos_fitCloudParameters'),
                dawgie.V_REF(rtime.task, self.__rt, self.__rt.sv_as_dict()['status'],
                             'cerberus_atmos_fitT'),
                dawgie.V_REF(rtime.task, self.__rt, self.__rt.sv_as_dict()['status'],
                             'cerberus_atmos_fitCtoO'),
                dawgie.V_REF(rtime.task, self.__rt, self.__rt.sv_as_dict()['status'],
                             'cerberus_atmos_fitNtoO')] + \
                self.__rt.refs_for_proceed()

    def state_vectors(self):
        '''Output State Vectors: cerberus.atmos'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''

        vfin, sfin = crbcore.checksv(self.__fin.sv_as_dict()['parameters'])
        if sfin: sfin = 'Missing system params!'

        svupdate = []
        for fltr in self.__rt.sv_as_dict()['status']['allowed_filter_names']:
            # stop here if it is not a runtime target
            self.__rt.proceed(fltr)

            update = False
            if fltr in self.__xsl.sv_as_dict():
                vxsl, sxsl = crbcore.checksv(self.__xsl.sv_as_dict()[fltr])
                if sxsl: sxsl = fltr + ' missing XSL'
            else:
                vxsl, sxsl = (False, fltr + ' missing XSL')

            if fltr in self.__tau.sv_as_dict():
                sv = self.__tau.sv_as_dict()[fltr]
                vspc, sspc = crbcore.checksv(sv)
                if sspc: sspc = fltr + ' missing taurex'
            else:
                vspc, sspc = (False, fltr + ' missing taurex')

            if not vspc:
                if fltr=='Ariel-sim':
                    sv = self.__arielsim.sv_as_dict()['parameters']
                    vspc, sspc = crbcore.checksv(sv)
                    sspc = 'Ariel-sim spectrum not found'
                else:
                    if fltr in self.__spc.sv_as_dict().keys():
                        sv = self.__spc.sv_as_dict()[fltr]
                        vspc, sspc = crbcore.checksv(sv)
                    else:
                        vspc = False
                        sspc = 'This filter doesnt have a spectrum: ' + fltr

            if vfin and vxsl and vspc:
                log.warning('--< CERBERUS ATMOS: %s >--', fltr)
                runtime = self.__rt.sv_as_dict()['status']

                # runtime_params = {
                #    'MCMC_chain_length':runtime['cerberus_steps'],
                #    'fitCloudParameters':runtime['cerberus_atmos_fitCloudParameters'],
                #    'fitT':runtime['cerberus_atmos_fitT'],
                #    'fitCtoO':runtime['cerberus_atmos_fitCtoO'],
                #    'fitNtoO':runtime['cerberus_atmos_fitNtoO']}
                cerb_params = namedtuple('cerberus_params_from_runtime',[
                    'MCMC_chain_length',
                    'fitCloudParameters',
                    'fitT',
                    'fitCtoO',
                    'fitNtoO'])
                runtime_params = cerb_params(
                    MCMC_chain_length=runtime['cerberus_steps'],
                    fitCloudParameters=runtime['cerberus_atmos_fitCloudParameters'],
                    fitT=runtime['cerberus_atmos_fitT'],
                    fitCtoO=runtime['cerberus_atmos_fitCtoO'],
                    fitNtoO=runtime['cerberus_atmos_fitNtoO'])
                # print('runtime params',runtime_params)

                update = self._atmos(self.__fin.sv_as_dict()['parameters'],
                                     self.__xsl.sv_as_dict()[fltr],
                                     sv, runtime_params, fltrs.index(fltr), fltr)
            else:
                errstr = [m for m in [sfin, sspc, sxsl] if m is not None]
                self._failure(errstr[0])
            if update: svupdate.append(self.__out[fltrs.index(fltr)])
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for CERBERUS.{self.name()}')
        return

    def _atmos(self, fin, xsl, spc, runtime_params, index, fltr):
        '''Core code call'''

        MCMC_chain_length = runtime_params.MCMC_chain_length.value()
        # MCMC_chain_length = 30
        # print('MCMC_chain_length',MCMC_chain_length)

        log.info(' calling atmos from cerb-alg-atmos  chain len=%d',MCMC_chain_length)
        am = crbcore.atmos(fin, xsl, spc, runtime_params, self.__out[index], fltr,
                           mclen=MCMC_chain_length,
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
        self.__out = [crbstates.rlsSV(fltr) for fltr in fltrs]
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
        fltr = 'HST-WFC3-IR-G141-SCAN'
        update = False
        if vfin:
            log.warning('--< CERBERUS RELEASE: %s >--', fltr)
            # pylint: disable=protected-access
            update = self._release(ds._tn(),
                                   self.__fin.sv_as_dict()['parameters'],
                                   fltrs.index(fltr))
            pass
        else:
            errstr = [m for m in [sfin] if m is not None]
            self._failure(errstr[0])
            pass
        if update: svupdate.append(self.__out[fltrs.index(fltr)])
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
        self.__anc = ancillaryalg.estimate()
        self.__xsl = xslib()
        self.__atm = atmos()
        self.__rt = rtalg.autofill()
        self.__out = [crbstates.resSV(fltr) for fltr in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'results'

    def previous(self):
        '''Input State Vectors: cerberus.atmos'''
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(anc.task, self.__anc),
                dawgie.ALG_REF(crb.task, self.__xsl),
                dawgie.ALG_REF(crb.task, self.__atm)] + \
                self.__rt.refs_for_proceed()

    def state_vectors(self):
        '''Output State Vectors: cerberus.results'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''

        svupdate = []
        vfin, sfin = crbcore.checksv(self.__fin.sv_as_dict()['parameters'])
        vanc, sanc = crbcore.checksv(self.__anc.sv_as_dict()['parameters'])

        update = False
        if vfin and vanc:
            # available_filters = self.__xsl.sv_as_dict().keys()
            # available_filters = self.__atm.sv_as_dict().keys()
            # print('available_filters',available_filters)
            # allowed_filters = self.__rt.sv_as_dict()['status']['allowed_filter_names']
            # print('allowed filters in cerb.results',allowed_filters)

            for fltr in self.__rt.sv_as_dict()['status']['allowed_filter_names']:
                # stop here if it is not a runtime target
                self.__rt.proceed(fltr)

                vxsl, sxsl = crbcore.checksv(self.__xsl.sv_as_dict()[fltr])
                vatm, satm = crbcore.checksv(self.__atm.sv_as_dict()[fltr])
                if vxsl and vatm:
                    log.warning('--< CERBERUS RESULTS: %s >--', fltr)
                    update = self._results(ds._tn(),  # pylint: disable=protected-access
                                           fltr,
                                           self.__fin.sv_as_dict()['parameters'],
                                           self.__anc.sv_as_dict()['parameters'],
                                           self.__xsl.sv_as_dict()[fltr]['data'],
                                           self.__atm.sv_as_dict()[fltr]['data'],
                                           fltrs.index(fltr))
                    if update: svupdate.append(self.__out[fltrs.index(fltr)])
                else:
                    errstr = [m for m in [sxsl, satm] if m is not None]
                    self._failure(errstr[0])
        else:
            errstr = [m for m in [sanc,sfin] if m is not None]
            self._failure(errstr[0])

        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for CERBERUS.{self.name()}')
        return

    def _results(self, trgt, fltr, fin, ancil, xsl, atm, index):
        '''Core code call'''
        resout = crbcore.results(trgt, fltr, fin, ancil, xsl, atm,
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
        self.__out = [crbstates.analysisSV(fltr) for fltr in fltrs]
        return

    # def previous(self):
    #    '''Input State Vectors: cerberus.atmos'''
    #        return [dawgie.ALG_REF(sys.task, self.__fin)]

    def feedback(self):
        '''feedback ds'''
        return []

    def name(self):
        '''Database name for subtask extension'''
        return 'analysis'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        '''traits ds'''
        return [dawgie.SV_REF(crb.task, atmos(),
                              sv) for sv in atmos().state_vectors()]

    def state_vectors(self):
        '''Output State Vectors: cerberus.analysis'''
        return self.__out

    def run(self, aspects:dawgie.Aspect):
        '''Top level algorithm call'''

        svupdate = []
        if len(aspects)==0:
            log.warning('--< CERBERUS ANALYSIS: contains no targets >--')
        else:
            # determine which filters have results from cerb.atmos (in aspects)
            #  (you have to loop through all targets, since filters vary by target)
            filtersWithResults = []
            for trgt in aspects:
                for fltr in fltrs:
                    if (fltr not in filtersWithResults) and \
                       ('cerberus.atmos.'+fltr in aspects[trgt]):
                        # print('This filter exists in the cerb.atmos aspect:',fltr,trgt)
                        filtersWithResults.append(fltr)
            if not filtersWithResults:
                log.warning('--< CERBERUS ANALYSIS: NO FILTERS WITH ATMOS DATA!!!>--')

            # filtersWithResults=['Ariel-sim']  # just one filter, while debugging
            # filtersWithResults=['HST-WFC3-IR-G141-SCAN']  # just one filter, while debugging

            # only consider filters that have cerb.atmos results loaded in as an aspect
            for fltr in filtersWithResults:
                # if 'cerberus.atmos.'+fltr not in aspects[trgt]:
                #    log.warning('--< CERBERUS ANALYSIS: %s not found IMPOSSIBLE!!!!>--', fltr)
                # else:
                log.warning('--< CERBERUS ANALYSIS: %s  >--', fltr)
                update = self._analysis(aspects, fltr, fltrs.index(fltr))
                if update: svupdate.append(self.__out[fltrs.index(fltr)])
        self.__out = svupdate
        if self.__out.__len__() > 0: aspects.ds().update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for CERBERUS.{self.name()}')
        return

    def _analysis(self, aspects, fltr, index):
        '''Core code call'''
        analysisout = crbcore.analysis(aspects, fltr,
                                       self.__out[index], verbose=False)
        return analysisout

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< CERBERUS ANALYSIS: %s >--', errstr)
        return

# -------------------------------------------------------------------
