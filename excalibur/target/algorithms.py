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
genIDs = False
# NEXSCI QUERY
web = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?'
# DATA ON DISK
diskloc = os.path.join(excalibur.context['data_dir'], 'sci')
# MAST MIRRORS
queryform = 'https://archive.stsci.edu/hst/search.php?target='
mirror1 = 'http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/HSTCA/'
mirror2 = 'http://archives.esac.esa.int/ehst-sl-server/servlet/data-action?ARTIFACT_ID='
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------

class alert(dawgie.Analyzer):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,0,0)
        self.__out = trgstates.MonitorSV()
        return

    def name(self):
        return 'alert_from_variations_of'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        return [dawgie.SV_REF(trg.regress, regress(), regress().state_vectors()[0])]

    def state_vectors(self):
        return [self.__out]

    def run(self, aspects:dawgie.Aspect):
        return
    pass

class create(dawgie.Analyzer):
    def __init__(self):
        self._version_ = trgedit.createversion()
        self.__out = [trgstates.TargetSV('starIDs'), trgstates.FilterSV('filters')]
        return

    def name(self):
        return 'create'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        return []

    def state_vectors(self):
        return self.__out

    def run(self, aspects:dawgie.Aspect):
        trgcore.scrapeids(aspects.ds(), self.__out[0], web, genIDs=genIDs)
        update = trgcore.createfltrs(self.__out[1])
        if update: aspects.ds().update()
        return
    pass

class autofill(dawgie.Algorithm):
    def __init__(self):
        self._version_ = trgcore.autofillversion()
        self.__create = create()
        self.__out = trgstates.TargetSV('parameters')
        return

    def name(self):
        return 'autofill'

    def previous(self):
        return [dawgie.ALG_REF(trg.analysis, self.__create)]

    def state_vectors(self):
        return [self.__out]

    def run(self, ds, ps):
        update = False
        var_create = self.__create.sv_as_dict()['starIDs']
        valid, errstring = trgcore.checksv(var_create)
        # pylint: disable=protected-access
        if valid and ds._tn() in var_create['starID']:
            update = self._autofill(var_create, ds._tn())
            pass
        else: self._failure(errstring)
        if update: ds.update()
        return

    def _autofill(self, crt, thistarget):
        solved = trgcore.autofill(crt, thistarget, self.__out)
        return solved

    @staticmethod
    def _failure(errstr):
        if errstr is None: errstr = 'TARGET NOT EXPECTED'
        log.warning('--< TARGET AUTOFILL: '+errstr+' >--')
        return
    pass

class scrape(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__autofill = autofill()
        self.__out = trgstates.DatabaseSV('databases')
        return

    def name(self):
        return 'scrape'

    def previous(self):
        return [dawgie.ALG_REF(trg.task, self.__autofill)]

    def state_vectors(self):
        return [self.__out]

    def run(self, ds, ps):
        update = False
        var_autofill = self.__autofill.sv_as_dict()['parameters']
        valid, errstring = trgcore.checksv(var_autofill)
        if valid: update = self._scrape(var_autofill, self.__out)
        else: self._failure(errstring)
        if update: ds.update()
        return

    @staticmethod
    def _scrape(arg_autofill, out):
        dbs = os.path.join(dawgie.context.data_dbs, 'mast')
        if not os.path.exists(dbs): os.makedirs(dbs)
        # umast = trgcore.mast(arg_autofill, out, dbs, queryform, mirror1, alt=mirror2)
        udisk = trgcore.disk(arg_autofill, out, diskloc, dbs)
        return udisk

    @staticmethod
    def _failure(errstr):
        log.warning('--< TARGET SCRAPE: '+errstr+' >--')
        return
    pass

class regress(dawgie.Regression):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,0,0)
        self.__out = trgstates.MonitorSV()
        return

    def feedback(self):
        return [dawgie.V_REF(trg.regress, self, self.__out, 'runid')]

    def name(self):
        return 'variations_of'

    def run(self, ps:int, timeline:dawgie.Timeline):
        last = trgmonitor.regress (self.__out['planet'],
                                   self.__out['runid'],
                                   timeline)
        self.__out['last'].clear()
        self.__out['last'].update (last)
        timeline.ds().update()
        return

    def state_vectors(self):
        return [self.__out]

    def variables(self)->[dawgie.SV_REF, dawgie.V_REF]:
        return [dawgie.SV_REF(trg.task,autofill(),autofill().state_vectors()[0])]
    pass
# ---------------- ---------------------------------------------------
