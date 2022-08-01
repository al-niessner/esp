'''target algorithms ds'''
# -- IMPORTS -- ------------------------------------------------------
import os
import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context

import excalibur
import excalibur.target as trg
import excalibur.target.edit as trgedit
import excalibur.target.core as trgcore
import excalibur.target.monitor as trgmonitor
import excalibur.target.states as trgstates
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# GENERATE DATABASE IDs
genIDs = True
# NEXSCI QUERY
web = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query='
# DATA ON DISK
diskloc = os.path.join(excalibur.context['data_dir'], 'sci')
# MAST MIRRORS
queryform = 'https://archive.stsci.edu/hst/search.php?target='
mirror1 = 'http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/HSTCA/'
mirror2 = 'http://archives.esac.esa.int/ehst-sl-server/servlet/data-action?ARTIFACT_ID='
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class alert(dawgie.Analyzer):
    '''alert ds'''
    def __init__(self):
        '''version 1.2.0 has new 'web' url for Exoplanet Archive queries'''
        self._version_ = dawgie.VERSION(1,2,0)
        self.__out = trgstates.AlertSV()
        return

    def feedback(self):
        '''feedback ds'''
        return [dawgie.V_REF(trg.analysis, self, self.__out, 'known'),
                dawgie.V_REF(trg.analysis, self, self.__out, 'table')]

    def name(self):
        '''Database name for subtask extension'''
        return 'alert_from_variations_of'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        '''traits ds'''
        return [dawgie.V_REF(trg.regress, regress(),
                             regress().state_vectors()[0], 'last')]

    def state_vectors(self):
        '''Output State Vectors: target.alert_from_variations_of'''
        return [self.__out]

    def run(self, aspects:dawgie.Aspect):
        '''Top level algorithm call'''
        c,k,t = trgmonitor.alert (aspects,self.__out['known'],self.__out['table'])
        self.__out['changes'].clear()
        self.__out['known'].clear()
        self.__out['table'].clear()
        self.__out['changes'].extend (c)
        self.__out['known'].extend (k)
        self.__out['table'].extend (t)
        aspects.ds().update()
        return
    pass

class create(dawgie.Analyzer):
    '''Creates a list of targets to be scraped from MAST'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = trgedit.createversion()
        self.__out = [trgstates.TargetSV('starIDs'), trgstates.FilterSV('filters')]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'create'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        '''Aspect Input: None (it s the clutch)'''
        return []

    def state_vectors(self):
        '''Output State Vectors: target.create'''
        return self.__out

    def run(self, aspects:dawgie.Aspect):
        '''Top level algorithm call'''
        trgcore.scrapeids(aspects.ds(), self.__out[0], web, genIDs=genIDs)
        update = trgcore.createfltrs(self.__out[1])
        if update: aspects.ds().update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for TARGET.{self.name()}')
        return
    pass

class autofill(dawgie.Algorithm):
    '''Fills mandatory info to get the ball rolling'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = trgcore.autofillversion()
        self.__create = create()
        self.__out = trgstates.TargetSV('parameters')
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'autofill'

    def previous(self):
        '''Input State Vectors: target.create'''
        return [dawgie.ALG_REF(trg.analysis, self.__create)]

    def state_vectors(self):
        '''Output State Vectors: target.autofill'''
        return [self.__out]

    def run(self, ds, ps):
        '''Top level algorithm call'''
        update = False
        crt = self.__create.sv_as_dict()['starIDs']
        valid, errstring = trgcore.checksv(crt)
        # pylint: disable=protected-access
        if valid and ds._tn() in crt['starID']: update = self._autofill(crt, ds._tn())
        else: self._failure(errstring)
        if update: ds.update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for TARGET.{self.name()}')
        return

    def _autofill(self, crt, thistarget):
        '''Core code call'''
        solved = trgcore.autofill(crt, thistarget, self.__out)
        return solved

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        if errstr is None: errstr = 'TARGET NOT EXPECTED'
        log.warning('--< TARGET AUTOFILL: %s >--', errstr)
        return
    pass

class scrape(dawgie.Algorithm):
    '''Download data or ingest data from disk'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,2,0)
        self.__autofill = autofill()
        self.__out = trgstates.DatabaseSV('databases')
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'scrape'

    def previous(self):
        '''Input State Vectors: target.autofill'''
        return [dawgie.ALG_REF(trg.task, self.__autofill)]

    def state_vectors(self):
        '''Output State Vectors: target.scrape'''
        return [self.__out]

    def run(self, ds, ps):
        '''Top level algorithm call'''
        update = False
        var_autofill = self.__autofill.sv_as_dict()['parameters']
        valid, errstring = trgcore.checksv(var_autofill)
        if valid: update = self._scrape(var_autofill, self.__out)
        else: self._failure(errstring)
        if update: ds.update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for TARGET.{self.name()}')
        return

    @staticmethod
    def _scrape(arg_autofill, out):
        '''Core code call'''
        dbs = os.path.join(dawgie.context.data_dbs, 'mast')
        if not os.path.exists(dbs): os.makedirs(dbs)
        # umast = trgcore.mast(arg_autofill, out, dbs, queryform, mirror1, alt=mirror2)
        udisk = trgcore.disk(arg_autofill, out, diskloc, dbs)
        return udisk

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< TARGET SCRAPE: %s >--', errstr)
        return
    pass

class regress(dawgie.Regression):
    '''regress ds'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,0,1)
        self.__out = trgstates.MonitorSV()
        return

    def feedback(self):
        '''feedback ds'''
        return [dawgie.V_REF(trg.regress, self, self.__out, 'planet'),
                dawgie.V_REF(trg.regress, self, self.__out, 'runid')]

    def name(self):
        '''Database name for subtask extension'''
        return 'variations_of'

    def run(self, ps:int, timeline:dawgie.Timeline):
        '''Top level algorithm call'''
        last, outlier = trgmonitor.regress (self.__out['planet'],
                                            self.__out['runid'],
                                            timeline)
        self.__out['last'].clear()
        self.__out['last'].update (last)
        self.__out['outlier'].clear()
        self.__out['outlier'].update (outlier)
        timeline.ds().update()
        return

    def state_vectors(self):
        '''Output State Vectors: target.variations_of'''
        return [self.__out]

    def variables(self)->[dawgie.SV_REF, dawgie.V_REF]:
        '''variables ds'''
        return [dawgie.SV_REF(trg.task,autofill(),autofill().state_vectors()[0])]
    pass
# -------------------------------------------------------------------
