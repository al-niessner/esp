'''ancillary states ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur
import excalibur.ancillary.core as anccore
import excalibur.ancillary.plot as ancplot

from collections import Counter
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class EstimateSV(dawgie.StateVector):
    '''EstimateSV ds'''
    def __init__(self, name):
        '''__init__ ds'''
        # version 2.0.0 adds Zellem figure-of-merit, mass-loss from wind, etc
        self._version_ = dawgie.VERSION(2,0,0)
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
            pl_params = ['planets'] + self['data']['planets']
            params = [i for i in list(self['data'].keys())
                      if is_param(i, pl_params)]
            ancplot.rendertable(self['data'], params, visitor)
            # display planetary estimates
            for pl in self['data']['planets']:
                visitor.add_primitive(f'PLANET: {pl}')
                params = [i for i in list(self['data'][pl].keys()) if is_param(i)]
                ancplot.rendertable(self['data'][pl], params, visitor)
            pass
        else:
            visitor.add_declaration('State vector marked as unsuccessful.')
            pass
        return
    pass

class PopulationSV(dawgie.StateVector):
    '''PopulationSV ds'''
    def __init__(self, name):
        self._version_ = dawgie.VERSION(2,0,0)
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
        to_process = [('----------------------Stellar Population Distributions----------------------',
                       self['data']['st_attrs'], self['data']['st_attrs_roudier62'], False),
                      ('---------------------Planetary Population Distributions---------------------',
                       self['data']['pl_attrs'], self['data']['pl_attrs_roudier62'], True)]
        for title, attrs, attrs_roudier62, is_planet in to_process:
            visitor.add_primitive(title)
            for key in attrs:
                estimator = get_estimator(key, is_planet)
                if estimator is not None and estimator.plot() == 'bar':
                    counts = Counter(attrs[key])
                    counts2 = Counter(attrs_roudier62[key])
                    if estimator.scale():
                        ancplot.barplot(key,
                                        estimator.scale(), [counts[i] for i in estimator.scale()],
                                        estimator.scale(), [counts2[i] for i in estimator.scale()],
                                        visitor)
                    else:
                        ancplot.barplot(key,
                                        counts.keys(), counts.values(),
                                        counts2.keys(), counts2.values(),
                                        visitor)
                else:
                    ancplot.distrplot(key, attrs[key], attrs_roudier62[key],
                                      visitor, estimator.units())
                    pass
                pass
            pass
        return
    pass
# -------- -----------------------------------------------------------
# -- HELPER FUNCTIONS ------------------------------------------------
def is_param(key, banlist=None):
    '''is_param ds'''
    lazygen = any(ext in key for ext in anccore.SV_EXTS)
    return not lazygen and (banlist is None or key not in banlist)

def get_estimator(name, is_planet):
    '''get_estimator ds'''
    st_estimators, pl_estimators = anccore.getestimators()
    if is_planet: ests = pl_estimators
    else: ests = st_estimators
    for est in ests:
        if est.name() == name: return est
        pass
    return None
# ------------------- ------------------------------------------------
