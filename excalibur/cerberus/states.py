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

class rlsSV(dawgie.StateVector):
    '''
    State Vector (SV) as python dict {}

    > SV.keys()
    dict_keys(['STATUS', 'data’])
    KEY - STATUS
    CONTENT - List of binary values.
    DESCRIPTION - False is always the first element.
    When an algorithms fills the SV shell, it adds one or more True values to the list.
    It is used to check that the SV that has been loaded from the database is an empty shell or not.
    KEY - data
    CONTENT - Dictionary containing the name of the planet(s) of the system
    i.e. ('b', 'c', …, 'e', …)

    > SV['data'].keys()
    dict_keys(['b'])
    KEY - Planet name
    CONTENT - Dictionary

    > SV['data']['b'].keys()
    dict_keys(['atmos', 'corrplot', 'modelplot'])
    KEY - atmos
    CONTENT - numpy array
    1st col: Wavelength [microns]
    2nd col: Planetary radius squared [Stellar radius squared]
    3rd col: Error on Planetary radius squared [Stellar radius squared]
    4th col: Best model [Stellar radius squared]
    KEY - corrplot
    CONTENT - numpy array
    Correlation plot
    KEY - modelplot
    CONTENT - numpy array
    Model + Data plot
    '''
    def __init__(self, name):
        '''1.1.1: GMR - Fixed view for low model selection preference'''
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''dataset name'''
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            for p in self['data']:
                if 'modelplot' in self['data'][p]:
                    myfig = plt.figure(figsize=(10, 6))
                    plt.imshow(self['data'][p]['modelplot'])
                    plt.axis('off')
                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    visitor.add_image('...', p+': Atmos results', buf.getvalue())
                    plt.close(myfig)

                    myfig = plt.figure(figsize=(20, 15))
                    plt.imshow(self['data'][p]['corrplot'])
                    plt.axis('off')
                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    visitor.add_image('...', p+': Profiled best model chains',
                                      buf.getvalue())
                    plt.close(myfig)
                    pass
                else:
                    visitor.add_declaration(p+': No/Low Evidence for Model Selection')
                    pass
                pass
            pass
        else:
            myfig = plt.figure(figsize=(10, 6))
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
