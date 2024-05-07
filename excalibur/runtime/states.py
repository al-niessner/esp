'''Runtime configuration products'''

import dawgie
import excalibur
import logging; log = logging.getLogger(__name__)

class BoolValue(dawgie.Value):
    '''helper value for boolean type'''
    def __bool__ (self):
        '''allows class to be treated like boolean using its __state'''
        return self.__state
    def __init__ (self, state:bool=False):
        '''init the boolean'''
        self.__state = bool(state)
        self._version_ = dawgie.VERSION(1,0,0)
        return
    def features(self):
        '''contains no features'''
        return []
    def new(self, state=None):
        '''hide explicit requirement for dawgie'''
        return BoolValue(state if state is not None else self.__state)
    pass

class CompositeSV(dawgie.StateVector):
    '''State representation of the configuration file'''
    def __init__(self, constituents:[dawgie.StateVector]):
        '''init the state vector with empty values'''
        self._version_ = dawgie.VERSION(1,0,0)
        for constituent in constituents:
            self[constituent.name()] = constituent
        return
    def name(self):
        '''database name'''
        return 'composite'
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        for key in sorted(self): self[key].view(visitor)
        return
    pass

class ControlsSV(dawgie.StateVector,dawgie.Value):
    '''State representation of the filters to be included/excluded'''
    def __init__(self):
        '''init the state vector with empty values'''
        self._version_ = dawgie.VERSION(1,0,0)
        self['ariel_simulate_spectra_includeMetallicityDispersion'] = BoolValue()
        self['cerberus_atmos_fitCloudParameters'] = BoolValue()
        self['cerberus_atmos_fitNtoO'] = BoolValue()
        self['cerberus_atmos_fitCtoO'] = BoolValue()
        self['cerberus_atmos_fitT'] = BoolValue()
        self['target_autofill_selectMostRecent'] = BoolValue()
        return
    def features(self):
        '''contains no features'''
        return []
    def name(self):
        '''database name'''
        return 'controls'
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        table = visitor.add_table(['Switch', 'State'],
                                  len(self)+1, 'Processing Control Switches')
        for row,key in enumerate(sorted(self)):
            table.get_cell (row+1,0).add_primitive(key)
            table.get_cell (row+1,1).add_primitive('on' if self[key] else 'off')
        return
    pass

class FilterSV(dawgie.StateVector,dawgie.Value):
    '''State representation of the filters to be included/excluded'''
    def __init__(self):
        '''init the state vector with empty values'''
        self._version_ = dawgie.VERSION(1,0,0)
        self['includes'] = excalibur.ValuesList()
        self['excludes'] = excalibur.ValuesList()
        return
    def features(self):
        '''contains no features'''
        return []
    def name(self):
        '''datebase name'''
        return 'filters'
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        table_len = max(len(self['excludes']),len(self['includes']))
        if table_len == 1:
            visitor.add_declaration('No target filters set', tag='b')
        else:
            self['excludes'].sort()
            self['includes'].sort()
            table = visitor.add_table(['Exclude','Include'], table_len+1,
                                      'Mission-Platform-Instrument-Mode Filters')
            for row in range(table_len):
                for col,filt in enumerate(['excludes','includes']):
                    if col < len(self[filt]):
                        table.get_cell (row+1,col).add_primitive(self[filt][col])
        return
    pass

class PymcSV(dawgie.StateVector,dawgie.Value):
    '''State representation of the filters to be included/excluded'''
    def __init__(self, name:str='undefined'):
        '''init the state vector with empty values'''
        self.__name = name
        self._version_ = dawgie.VERSION(1,0,0)
        self['default'] = excalibur.ValueScalar()
        self['overrides'] = excalibur.ValuesDict()
        return
    def features(self):
        '''contains no features'''
        return []
    def name(self):
        '''database name'''
        return f'pymc-{self.__name}'
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the configutation information'''
        visitor.add_declaration(f'PYMC for {self.__name} '
                                f'default chain length: self["default"]',
                                tag='b')
        if self['overrides']:
            table = visitor.add_table(['Target','Chainlength'],
                                      len(self['overrides'])+1,
                                      f'Overrides for {self.__name}')
            for row,tn in enumerate(sorted(self['overrides'])):
                table.get_cell(row+1,0).add_primitive(tn)
                table.get_cell(row+1,1).add_primitive(self['overrides'][tn])
        else: visitor.add_primitive(' with no overrides')
        return
    pass

class StatusSV(dawgie.StateVector):
    '''State representation of how the AE should view this target'''
    def __init__(self):
        '''init the state vector with empty values'''
        self._version_ = dawgie.VERSION(1,0,0)
        self['allowed_filter_names'] = excalibur.ValuesList()
        self['ariel_simulate_spectra_includeMetallicityDispersion'] = BoolValue()
        self['cerberus_atmos_fitCloudParameters'] = BoolValue()
        self['cerberus_atmos_fitNtoO'] = BoolValue()
        self['cerberus_atmos_fitCtoO'] = BoolValue()
        self['cerberus_atmos_fitT'] = BoolValue()
        self['cerberus_steps'] = excalibur.ValueScalar()
        self['isValidTarget'] = BoolValue()
        self['runTarget'] = BoolValue(True)
        self['spectrum_steps'] = excalibur.ValueScalar()
        self['target_autofill_selectMostRecent'] = BoolValue()
    def name(self):
        '''database name'''
        return 'status'
    def proceed(self, ext:str=None):
        '''determine if those that care should proceed'''
        allowed = ext in self['allowed_exts'] if ext else True
        run = self['runTarget']
        valid = self['isValidTarget']
        if not all([allowed, run, valid]):
            msg = ('Determined that should not process this target for ext:\n'
                   f'  validTarget:    {valid}\n'
                   f'  runnable:       {run}\n'
                   f'  ext is allowed: {allowed}')
            log.info(msg)
            raise dawgie.NoValidInputDataError(msg)
    def view(self, visitor:dawgie.Visitor)->None:
        '''Show the state for this target'''
        return
    pass

class TargetsSV(dawgie.StateVector,dawgie.Value):
    '''State representation of the targets to sequester'''
    def __init__(self, name:str='undefined'):
        '''init the state vector with empty values'''
        self._name = name
        self._version_ = dawgie.VERSION(1,0,0)
        self['targets'] = excalibur.ValuesList()
        return
    def features(self):
        '''contains no features'''
        return []
    def name(self):
        '''database name'''
        return self._name
    def view (self, visitor:dawgie.Visitor)->None:
        '''Show the configuration information'''
        if self._name == 'run_only':
            title = 'Run only thse targets:'
            if not self['targets']: title += ' ALL'
        else:
            title = 'Do not run these targets:'
            if not self['targets']: title += ' NONE'

        visitor.add_declararion(title, tag='b')

        if self['targets']:
            table = visitor.add_table(['Target','Why'],
                                      len(self['targets'])+1,
                                      'Taarget table')
            for row,tn in enumerate(sorted(self['targets'],key=lambda t:t[0])):
                table.get_cell(row+1,0).add_primitive(tn[0])
                table.get_cell(row+1,1).add_primitive(tn[1])
        return
    pass
