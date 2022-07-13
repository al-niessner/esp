'''Cerberus Database Products View'''
# -- IMPORTS -- ------------------------------------------------------
import io

import dawgie

import excalibur

import matplotlib.image as img
import matplotlib.pyplot as plt

import os
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class xslibSV(dawgie.StateVector):
    '''cerberus.xslib view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,2,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            myfig = plt.figure()
            crblogo = img.imread(os.path.join (excalibur.context['data_dir'],
                                               'CERBERUS/cerberus.png'))
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
    '''cerberus.atmos view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            myfig = plt.figure()
            crblogo = img.imread(os.path.join (excalibur.context['data_dir'],
                                               'CERBERUS/cerberus.png'))
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
