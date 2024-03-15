'''The actors/agents that do something'''

import dawgie
import .algorithms

class AnalysisTeam(dawgie.Analysis):
    '''Analytical team'''
    def list(self)->[dawgie.Analyzer]:
        '''list of analysis to be done'''
        return [ algorithms.create() ]
    pass

class TaskTeam(dawgie.Task):
    '''Task team'''
    def list(self)->[dawgie.Task]:
        '''list of tasks to perform'''
        return [ algorithms.autofill() ]
    pass
