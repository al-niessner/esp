# -- IMPORTS -- ------------------------------------------------------
import io

import dawgie
import exo.spec.ae

import numpy as np
from matplotlib import image as img
from matplotlib import pyplot as plt
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class xslibSV(dawgie.StateVector):
    def __init__(self, name='lines'):
        self.__name = name
        self._version_ = dawgie.VERSION(1,2,0)
        self['XSECS'] = exo.spec.ae.ValuesList()
        self['QTGRID'] = exo.spec.ae.ValuesList()
        self['RES'] = exo.spec.ae.ValuesList()
        return
    def name(self):
        return self.__name
    def view(self, visitor:dawgie.Visitor)->None:
        myfig = plt.figure()
        crblogo = img.imread('/proj/sdp/data/CERBERUS/cerberus.png')
        plt.imshow(crblogo)
        plt.axis('off')
        buf = io.BytesIO()
        myfig.savefig(buf, format='png')
        visitor.add_image('...', 'Cerberus ', buf.getvalue())
        plt.close(myfig)
        return
    pass

class hazelibSV(dawgie.StateVector):
    def __init__(self, name='vdensity'):
        self.__name = name
        self._version_ = dawgie.VERSION(1,1,0)
        self['PROFILE'] = exo.spec.ae.ValuesList()
        return
    def name(self):
        return self.__name
    def view(self, visitor:dawgie.Visitor)->None:
        myfig = plt.figure()
        crblogo = img.imread('/proj/sdp/data/CERBERUS/cerberus.png')
        plt.imshow(crblogo)
        plt.axis('off')
        buf = io.BytesIO()
        myfig.savefig(buf, format='png')
        visitor.add_image('...', 'Cerberus ', buf.getvalue())
        plt.close(myfig)
        return
    pass

class atmosSV(dawgie.StateVector):
    def __init__(self, name='model'):
        self.__name = name
        self._version_ = dawgie.VERSION(1,3,0)
        self['DATA'] = exo.spec.ae.ValuesList()
        self['MODEL'] = exo.spec.ae.ValuesList()
        self['LMPARAMS'] = exo.spec.ae.ValuesList()
        self['MCPOST'] = exo.spec.ae.ValuesList()
        self['MCCHAINS'] = exo.spec.ae.ValuesList()
        return
    def name(self):
        return self.__name
    def view(self, visitor:dawgie.Visitor)->None:
        myfig = plt.figure()
        crblogo = img.imread('/proj/sdp/data/CERBERUS/cerberus.png')
        plt.imshow(crblogo)
        plt.axis('off')
        buf = io.BytesIO()
        myfig.savefig(buf, format='png')
        visitor.add_image('...', 'Cerberus ', buf.getvalue())
        plt.close(myfig)
        return
    pass
# -------- -----------------------------------------------------------
