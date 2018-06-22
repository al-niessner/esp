import builtins
import dawgie
import numpy
import scipy.stats

import os

context = {'data_cal':os.environ.get ('DATA_CALIBR', '/proj/sdp/data/cal'),
           'data_sci':os.environ.get ('DATA_SCIENC', '/proj/sdp/data/sci'),
           'ldtk_root':os.environ.get('LDTK_ROOT', '/proj/sdp/data/ldtk'),
           'target_list':os.environ.get
           ('TARGET_LIST','/proj/sdp/data/WFC3_target_list.xlsx')}
os.environ['LDTK_ROOT'] = context['ldtk_root']

'''Algorithm Engine

The algirthm engine has N goals:

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

class ValuesList(dawgie.Value, list):
    def __init__ (self, *args, **kwds):
        list.__init__ (self, *args, **kwds)
        self._version_ = dawgie.VERSION(1,1,0)
        return
    pass

class ValuesDict(dawgie.Value, dict):
    def __init__ (self, *args, **kwds):
        dict.__init__ (self, *args, **kwds)
        self._version_ = dawgie.VERSION(1,1,0)
        return
    pass

class ValueScalar(dawgie.Value):
    def __init__ (self, content=None):
        dawgie.Value.__init__ (self)
        self.__content = content
        self._version_ = dawgie.VERSION(1,1,0)
        return
    def value(self): return self.__content
    pass

class GENERIC_FITS(dawgie.StateVector):
    def __init__(self, name):
        self['name'] = ValuesList()
        self['time'] = ValuesList()
        self.__name = name
        self._version_ = dawgie.VERSION(3,1,0)
        return
    def name(self): return self.__name
    pass

class Tag(object):
    ## DO NOT extend this class. Extend one of its children.
    def tag_value (self, mask=None, kurt:dict={}, mean:dict={}, median:dict={},
                   skew:dict={}, std:dict={}, **kwds)->None:
        self._mask = mask
        kurt.update (kwds)
        mean.update (kwds)
        median.update (kwds)
        skew.update (kwds)
        std.update (kwds)
        self.kurtosis = self._kurtosis (**kurt)
        self.mean = self._mean (**mean)
        self.median = self._median (**median)
        self.skew = self._skew (**skew)
        self.std = self._std (**std)
        return
    
class ArrayTag(Tag):
    def _kurtosis(self, **kwds):
        return scipy.stats.kurtosis (self.as_array(), **kwds)
    def _mean(self, **kwds): return self.as_array().mean (**kwds)
    def _median(self, **kwds):
        return numpy.median (self.as_array(), **kwds)
    def _skew(self, **kwds):
        return scipy.stats.skew (self.as_array(), **kwds)
    def _std(self, **kwds): return self.as_array().std (**kwds) 

    def as_array(self)->numpy.ndarray: raise NotImplementedError()
    pass

class MaskedTag(Tag):
    def _kurtosis(self, **kwds):
        return scipy.stats.kurtosis (self.as_masked_array(), **kwds)
    def _mean(self, **kwds): return self.as_masked_array().mean (**kwds)
    def _median(self, **kwds): return self.as_masked_array().median (**kwds)
    def _skew(self, **kwds):
        return scipy.stats.skew (self.as_masked_array(), **kwds)
    def _std(self, **kwds): return self.as_masked_array().std (**kwds) 

    def as_masked_array(self)->numpy.ma.MaskedArray: raise NotImplementedError()
    pass

def tag_values (sv:dawgie.StateVector, **kwds)->None:
    for k,v in builtins.filter (lambda i:isinstance (i[1], Tag), sv.items()):
        mask = sv[sv.mask_keys[k]] if ('mask_keys' in dir (sv) and
                                       k in sv.mask_keys and
                                       sv.mask_keys[k]) else None
        v.tag_value (mask=mask, **kwds)
        pass
    return

def view_tags (sv:dawgie.StateVector, visitor:dawgie.Visitor)->None:
    count = numpy.sum ([isinstance (v, Tag) for v in sv.values()])

    if 0 < count:
        visitor.add_declaration ('Tag Table:', tag='h4')
        table = visitor.add_table (['name', 'median', 'mean',
                                    'variance', 'skew', 'kurtosis'],
                                   count)
        for i,(k,v) in enumerate (builtins.filter(lambda t:isinstance(t[1],Tag),
                                                  sv.items())):
            table.get_cell (i,0).add_primitive (k)
            table.get_cell (i,1).add_primitive (v.median)
            table.get_cell (i,2).add_primitive (v.mean)
            table.get_cell (i,3).add_primitive (v.std**2)
            table.get_cell (i,4).add_primitive (v.skew)
            table.get_cell (i,5).add_primitive (v.kurtosis)
            pass
        pass
    return
