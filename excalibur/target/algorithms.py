# -- IMPORTS -- ------------------------------------------------------
import os
import pdb

import dawgie
import dawgie.context

import exo.spec.ae.target as trg
import exo.spec.ae.target.core as trgcore
import exo.spec.ae.target.states as trgstates
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# VERBOSE AND DEBUG
verbose = False
debug = False
# GENERATE DATABASE IDs
genIDs = True
# NEXSCI QUERY
web = 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?'
# DATA ON DISK
diskloc = '/proj/sdp/data/sci'
# MAST MIRRORS
queryform = 'https://archive.stsci.edu/hst/search.php?target='
mirror1 = 'http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/data/pub/HSTCA/'
mirror2 = 'http://archives.esac.esa.int/hst/proxy?file_id='
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class create(dawgie.Analyzer):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__out = [trgstates.TargetSV('starIDs'),
                      trgstates.FilterSV('filters')]
        return
    
    def name(self):
        return 'create'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        return []

    def state_vectors(self):
        return self.__out

    def run(self, aspects:dawgie.Aspect):
        trgcore.scrapeids(aspects.ds(), self.__out[0], web,
                          genIDs=genIDs,
                          verbose=verbose, debug=debug)
        update = trgcore.createfltrs(self.__out[1],
                                     verbose=verbose, debug=debug)
        if update: aspects.ds().update()
        return
    pass

class autofill(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__create = create()
        self.__out = trgstates.TargetSV('parameters')
        return
    
    def name(self):
        return 'autofill'

    def previous(self):
        return [dawgie.ALG_REF(trg.all, self.__create)]

    def state_vectors(self):
        return [self.__out]

    def run(self, ds, ps):
        update = False
        create = self.__create.sv_as_dict()['starIDs']
        valid, errstring = trgcore.checksv(create)
        if (valid and (ds._tn() in create['starID'].keys())):
            update = self._autofill(create, ds._tn(), self.__out)
            pass
        else: self._failure(errstring)
        if update: ds.update()
        return

    def _autofill(self, create, thistarget, out):
        solved = trgcore.autofill(create, thistarget, out,
                                  verbose=verbose, debug=debug)
        return solved
    
    def _failure(self, errstr):
        if errstr is None: errstr = 'TARGET NOT EXPECTED'
        errmess = '--< TARGET AUTOFILL: ' + errstr + ' >--'
        if verbose: print(errmess)
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
        return [dawgie.ALG_REF(trg.factory, self.__autofill)]

    def state_vectors(self):
        return [self.__out]

    def run(self, ds, ps):
        update = False
        autofill = self.__autofill.sv_as_dict()['parameters']
        valid, errstring = trgcore.checksv(autofill)
        if valid: update = self._scrape(autofill, self.__out)
        else: self._failure(errstring)
        if update: ds.update()
        return

    def _scrape(self, autofill, out):
        dbs = os.path.join(dawgie.context.data_dbs, 'mast')
        if (not os.path.exists(dbs)): os.makedirs(dbs)
        umast = trgcore.mast(autofill, out, dbs, queryform,
                             mirror1, alt=mirror2,
                             verbose=verbose, debug=debug)
        udisk = trgcore.disk(autofill, out, diskloc, dbs, 
                             verbose=verbose, debug=debug)
        update = umast or udisk
        return update

    def _failure(self, errstr):
        errmess = '--< TARGET SCRAPE: ' + errstr + ' >--'
        if verbose: print(errmess)
        return
    pass
# ---------------- ---------------------------------------------------
