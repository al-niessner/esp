'''helper state vector classes to minimize duplicate code'''

import dawgie
import excalibur


class ExcaliburSV(dawgie.StateVector):
    '''do the general name handling and setup the initial state of myself'''

    def __init__(self, name, version):
        self._version_ = version
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller: excalibur.Identity, visitor: dawgie.Visitor) -> None:
        '''pass along the abstract method'''
        raise NotImplementedError()
