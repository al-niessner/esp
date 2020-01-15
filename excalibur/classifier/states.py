# -- IMPORTS -- ------------------------------------------------------
# import io GMR: UNUSED FOR NOW
import dawgie
import excalibur

# import numpy as np GMR: UNUSED FOR NOW
# import matplotlib.pyplot as plt GMR: UNUSED FOR NOW
# ------------- ------------------------------------------------------

# -- SV -- -----------------------------------------------------------
class PredictSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            for p in self['data'].keys():
                visitor.add_declaration('PLANET: ' + p)
                pass
            pass
        pass
    pass
