# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur
import excalibur.ancillary.core as anccore
import excalibur.ancillary.plot as ancplot

from collections import Counter
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class EstimateSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,0,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            pl_params = ['planets'] + self['data']['planets']
            params = [i for i in list(self['data'].keys())
                      if is_param(i, pl_params)]
            ancplot.rendertable(self['data'], params, visitor)
            # display planetary estimates
            for pl in self['data']['planets']:
                visitor.add_primitive('PLANET: {}'.format(pl))
                params = [i for i in list(self['data'][pl].keys())
                          if is_param(i)]
                ancplot.rendertable(self['data'][pl], params, visitor)
            pass
        else:
            visitor.add_declaration('State vector marked as unsuccessful.')

class PopulationSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,0,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        to_process = [('Stellar Population Distributions', self['data']['st_attrs'], False),
                      ('Planetary Population Distributions', self['data']['pl_attrs'], True)]
        for title, attrs, is_planet in to_process:
            visitor.add_primitive(title)
            for key in attrs:
                estimator = get_estimator(key, is_planet)
                if estimator is not None and estimator.plot() == 'bar':
                    counts = Counter(attrs[key])
                    if estimator.scale():
                        ancplot.barplot(key, estimator.scale(),
                                        [counts[i] for i in estimator.scale()], visitor)
                    else:
                        ancplot.barplot(key, counts.keys(), counts.values(), visitor)
                else:
                    ancplot.distrplot(key, attrs[key], visitor,
                                      estimator.units())
# -------- -----------------------------------------------------------


# ------------- ------------------------------------------------------
# -- HELPER FUNCTIONS ------------------------------------------------
def is_param(key, banlist=None):
    return not any([ext in key for ext in anccore.SV_EXTS]) \
            and (banlist is None or key not in banlist)


def get_estimator(name, is_planet):
    st_estimators, pl_estimators = anccore.getestimators()
    if is_planet:
        ests = pl_estimators
    else:
        ests = st_estimators
    for est in ests:
        if est.name() == name:
            return est
    return None
