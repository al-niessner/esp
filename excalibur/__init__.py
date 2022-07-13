'''Algorithm Engine

The algorithm engine has N goals:

  1. Separate the programatic steps of accomplishing a science goal. Each of the
     Python package is the encapsulation of an algorithm that ends at a science
     goal. Each package then uses extensions of the classes:
        dawgie.Algorithm
        dawgie.StateVector
        dawgie.Task
        dawgie.Value
     to divide the science algorithm into programtic elements. They also enable
     the division of the PipeLine from the Algorithm Engine.
'''
# -- IMPORTS -- ------------------------------------------------------
import builtins
import dawgie
import numpy
import scipy.stats

import os
# ------------- ------------------------------------------------------
context = {'data_cal':os.environ.get ('DATA_CALIBR', '/proj/data/cal'),
           'data_dir':os.environ.get ('DATA_BASEDIR', '/proj/data'),
           'data_sci':os.environ.get ('DATA_SCIENC', '/proj/data/sci'),
           'ldtk_root':os.environ.get('LDTK_ROOT', '/proj/data/ldtk'),
           'target_list':os.environ.get('TARGET_LIST',
                                        '/proj/data/WFC3_target_list.xlsx')}
os.environ['LDTK_ROOT'] = context['ldtk_root']
__version__ = '${UNDEFINED}'

class ValuesList(dawgie.Value, list):
    '''ValuesList ds'''
    def __init__ (self, *args, **kwds):
        '''__init__ ds'''
        list.__init__ (self, *args, **kwds)
        self._version_ = dawgie.VERSION(1,1,0)
        return
    def features (self):
        '''features ds'''
        return []
    pass

class ValuesDict(dawgie.Value, dict):
    '''ValuesDict ds'''
    def __init__ (self, *args, **kwds):
        '''__init__ ds'''
        dict.__init__ (self, *args, **kwds)
        self._version_ = dawgie.VERSION(1,1,0)
        return
    def features (self):
        '''features ds'''
        return []
    pass

class ValueScalar(dawgie.Value):
    '''ValueScalar'''
    def __init__ (self, content=None):
        '''__init__ ds'''
        dawgie.Value.__init__ (self)
        self.__content = content
        self._version_ = dawgie.VERSION(1,1,0)
        return
    def features (self):
        '''features ds'''
        return []
    def value(self):
        '''value ds'''
        return self.__content
    pass

class Visitor(dawgie.Visitor):
    '''Visitor ds'''
    def add_declaration (self, text:str, **kwds)->None:
        '''add_declaration ds'''
        print ('declaration', text, kwds)
        return

    def add_image (self, alternate:str, label:str, img:bytes)->None:
        '''add_image ds'''
        print ('image', label, alternate, len(img))
        return

    def add_primitive (self, value, label:str=None)->None:
        '''add_primitive ds'''
        print ('primitive', label, value)
        return

    def add_table (self, clabels:[str], rows:int=0,
                   title:str=None)->dawgie.TableVisitor:
        '''add_table ds'''
        print ('table', clabels, rows, title)
        return VisitorTable()
    pass

class VisitorTable(dawgie.TableVisitor):
    '''VisitorTable ds'''
    # pylint: disable=too-few-public-methods
    def get_cell (self, r:int, c:int)->Visitor:
        '''get_cell ds'''
        print ('table element', r, c)
        return Visitor()
    pass
