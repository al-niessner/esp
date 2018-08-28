# -- IMPORTS -- ------------------------------------------------------
import io

import dawgie

import excalibur

import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class xslibSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,2,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            myfig = plt.figure()
            crblogo = img.imread('/proj/sdp/data/CERBERUS/cerberus.png')
            plt.imshow(crblogo)
            plt.axis('off')
            buf = io.BytesIO()
            myfig.savefig(buf, format='png')
            visitor.add_image('...', 'Cerberus ', buf.getvalue())
            plt.close(myfig)
            pass
        return
    pass

class atmosSV(dawgie.StateVector):
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
            myfig = plt.figure()
            crblogo = img.imread('/proj/sdp/data/CERBERUS/cerberus.png')
            plt.imshow(crblogo)
            plt.axis('off')
            buf = io.BytesIO()
            myfig.savefig(buf, format='png')
            visitor.add_image('...', 'Cerberus ', buf.getvalue())
            plt.close(myfig)
            pass
        return
    pass
# -------- -----------------------------------------------------------
