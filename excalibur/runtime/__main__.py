'''test the code without a pipeline'''

import dawgie.db
import excalibur.runtime.algorithms

def targets():
    '''as not require a real DB running'''
    return ['a','b','c', 'GJ 3193', 'GJ 3193 (taurex sim @TS)']

setattr (dawgie.db, 'targets', targets)
test = excalibur.runtime.algorithms.create()
test.run (None)
