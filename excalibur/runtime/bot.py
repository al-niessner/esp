'''The actors/agents that do something'''

import dawgie
from . import algorithms

class AnalysisTeam(dawgie.Analysis):
    '''Analytical team'''
    def list(self)->[dawgie.Analyzer]:
        '''list of analysis to be done'''
        return [algorithms.create()]
    pass

class TaskTeam(dawgie.Task):
    '''Task team'''
    def __init__(self, *args, table:{str:{}}=None, this_tn:str=None, **kwds):
        '''override task without knowing anything about it'''
        dawgie.Task.__init__(self, *args, **kwds)
        self.__table = table
        self.__tn = this_tn
    def list(self)->[dawgie.Task]:
        '''list of tasks to perform'''
        return [algorithms.autofill(table=self.__table, tn=self.__tn)]
    pass
