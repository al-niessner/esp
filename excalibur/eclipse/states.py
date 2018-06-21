# -- IMPORTS -- ------------------------------------------------------
import dawgie

import exo.spec.ae
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class NormSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = exo.spec.ae.ValuesList()
        self['STATUS'].append(False)
        return
    
    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            pass
        pass
# -------- -----------------------------------------------------------
