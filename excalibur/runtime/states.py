'''Runtime configuration products'''

import dawgie
import excalibur

class BoolValue(dawgie.Value):
    def __bool__ (self): return self.__state
    def __init__ (self, state:bool=False):
        self.__state = state if state else False  # change None to false
        self._version_ = dawgie.VERSION(1,0,0)
        return
    def features(self): return []
    pass

class CompositeSV(dawgie.StateVector):
    '''State representation of the configuration file'''
    def __init__(self, constituents:[dawgie.StateVector]):
        '''init the state vector with empty values'''
        self._version_ = dawgie.VERSION(1,0,0)
        for constituent in constituents:
            self[constituent.name()] = constituent
        return
    def name(self): return 'composite'
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        return
    pass

class ControlsSV(dawgie.StateVector,dawgie.Value):
    '''State representation of the filters to be included/excluded'''
    def __init__(self):
        '''init the state vector with empty values'''
        self._version_ = dawgie.VERSION(1,0,0)
        self['cerberus-atmos-fitCloudParameters'] = BoolValue()
        self['cerberus-atmos-fitNtoO'] = BoolValue()
        self['cerberus-atmos-fitCtoO'] = BoolValue()
        self['cerberus-atmos-fitT'] = BoolValue()
        self['target-autofill-selectMostRecent'] = BoolValue()
        self['ariel-simulate_spectra-includeMetallicityDispersion'] = BoolValue()
        return
    def name(self): return 'controls'
    def features(self): return []
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        return
    pass

class FiltersSV(dawgie.StateVector,dawgie.Value):
    '''State representation of the filters to be included/excluded'''
    def __init__(self):
        '''init the state vector with empty values'''
        self._version_ = dawgie.VERSION(1,0,0)
        self['includes'] = excalibur.ValuesList()
        self['excludes'] = excalibur.ValuesList()
        return
    def name(self): return 'filters'
    def features(self): return []
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        return
    pass

class PymcSV(dawgie.StateVector,dawgie.Value):
    '''State representation of the filters to be included/excluded'''
    def __init__(self, name:str):
        '''init the state vector with empty values'''
        self.__name = f'pymc-{name}'
        self._version_ = dawgie.VERSION(1,0,0)
        self['default'] = excalibur.ValuesScalar()
        self['overrides'] = excalibur.ValuesDict()
        return
    def name(self): return self.__name
    def features(self): return []
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        return
    pass
