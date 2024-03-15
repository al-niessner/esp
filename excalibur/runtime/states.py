'''Runtime configuration products'''

import dawgie
import excalibur

class ConfigSV(dawgie.StateVector):
    '''State representation of the configuration file'''
    def __init__(self):
        '''init the state vector with empty values'''
        self._version_ = dawgie.VERSION(1,0,0)
        self['pymc'] = excalibur.ValuesDict()
        self['fit_parameters'] = excalibur.ValuesDict()
        self['filter_includes'] = excalibur.ValuesList()
        self['filter_excludes'] = excalibur.ValuesList()
        self['targets_ariel'] = excalibur.ValuesList()
        self['targets_bad'] = excalibur.ValuesList()
        self['targets_spitzer'] = excalibur.ValuesList()
        return

    def name(self): return 'config'
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        return
    pass

    
