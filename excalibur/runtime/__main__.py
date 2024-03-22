'''test the code without a pipeline'''

import dawgie.db
import excalibur.runtime.algorithms

class FakeDawgie:
    '''keep the test runnable without needing a pipeline'''
    def __init__(self):
        self.__name = 'runtime'
    def _bot(self): return self
    def _runid(self): return self
    def name(self): return self.__name
    def ds(self): return self
    def update(self): return self

def connect(_alg, _bit, _tn):
    return FakeDawgie()

def targets():
    '''as not require a real DB running'''
    return ['a','b','c', 'GJ 3193', 'GJ 3193 (taurex sim @TS)']

setattr (dawgie.db, 'connect', connect)
setattr (dawgie.db, 'targets', targets)
test = excalibur.runtime.algorithms.create()
test.run (FakeDawgie())
