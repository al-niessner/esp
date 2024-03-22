'''test the code without a pipeline'''

import dawgie.db
import excalibur.runtime.algorithms

# need to fake some dawgie stuff so that can run the unit independent of any
# pipeline since it is the data in the configuration file that matters not any
# data or state of the pipeline itself. There going to
# pylint: disable=missing-class-docstring,missing-function-docstring
class FakeDawgie:
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
    return ['a','b','c', 'GJ 3193', 'GJ 3193 (taurex sim @TS)']
# pylint: enable=missing-class-docstring,missing-function-docstring

setattr (dawgie.db, 'connect', connect)
setattr (dawgie.db, 'targets', targets)
test = excalibur.runtime.algorithms.create()
test.run (FakeDawgie())
