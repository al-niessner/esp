'''target bot ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur.target.algorithms as trgalg
# ------------- ------------------------------------------------------
# -- A&A -- ----------------------------------------------------------
class Actor(dawgie.Analysis):
    '''Actor ds'''
    def list(self)->[dawgie.Analyzer]:
        '''list ds'''
        return [
            trgalg.alert(),
            trgalg.create()
        ]
    pass

class Agent(dawgie.Task):
    '''Agent ds'''
    def list(self)->[dawgie.Task]:
        '''list ds'''
        return [
            trgalg.autofill(),
            trgalg.scrape()
        ]
    pass

class Regress(dawgie.Regress):
    '''Regress ds'''
    def list(self)->[dawgie.Regression]:
        '''list ds'''
        return [
            trgalg.regress()
            ]
    pass
# --------- ----------------------------------------------------------
