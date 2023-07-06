'''ancillary algoithms dc'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

from collections import defaultdict

import logging; log = logging.getLogger(__name__)

import excalibur.ancillary as anc
import excalibur.ancillary.core as anccore
import excalibur.ancillary.states as ancstates

import excalibur.system as sys
import excalibur.system.algorithms as sysalg

from excalibur.target.targetlists import get_target_lists

from excalibur.ancillary.core import savesv

# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# FILTERS
# fltrs = (trgedit.activefilters.__doc__).split('\n')
# fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class estimate(dawgie.Algorithm):
    '''estimate ds'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = anccore.estimateversion()
        self._type = 'ancillary'
        self.__fin = sysalg.finalize()
        self.__out = [ancstates.EstimateSV('parameters')]
        return

    def name(self):
        '''name ds'''
        return 'estimate'

    def previous(self):
        '''previous ds'''
        return [dawgie.ALG_REF(sys.task, self.__fin)]

    def state_vectors(self):
        '''state_vectors ds'''
        return self.__out

    def run(self, ds, ps):
        '''run ds'''
        update = False
        vfin, _ = anccore.checksv(self.__fin.sv_as_dict()['parameters'])
        if vfin:
            update = anccore.estimate(self.__fin.sv_as_dict()['parameters'],
                                      self.__out[0])
            pass
        if update: ds.update()
        else: raise dawgie.NoValidOutputDataError(
                f'No output created for ANCILLARY.{self.name()}')
        return

class population(dawgie.Analyzer):
    '''population ds'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,0,3)
        self.__out = ancstates.PopulationSV('statistics')
        return

    def feedback(self):
        '''feedback ds'''
        return []

    def name(self):
        '''name ds'''
        return 'population'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        '''traits ds'''
        return [dawgie.V_REF(anc.task, estimate(),
                             estimate().state_vectors()[0], 'data')]

    def state_vectors(self):
        '''state_vectors ds'''
        return [self.__out]

    def run(self, aspects:dawgie.Aspect):
        '''run ds'''

        targetlists = get_target_lists()

        # group together values by attribute
        svname = 'ancillary.estimate.parameters'

        # save system-finalize results as a .csv file (in /proj/data/spreadsheets/)
        savesv(aspects, targetlists)

        st_attrs = defaultdict(list)
        pl_attrs = defaultdict(list)
        # include a second set of attributes, for comparison within each histogram
        st_attrs_roudier62 = defaultdict(list)
        pl_attrs_roudier62 = defaultdict(list)

        # for trgt in aspects:
        #    target_sample = '__all__'
        # Only consider the 'active' stars, not aliases/misspellings/dropped/etc
        for trgt in targetlists['active']:
            target_sample = 'active'

            tr_data = aspects[trgt][svname]

            # verify SV succeeded for target
            if tr_data['STATUS'][-1] or 'planets' in tr_data['data']:
                # get stellar attributes
                st_keys = [i for i in tr_data['data'].keys()
                           if is_st_key(i, tr_data['data']['planets'])]
                for key in st_keys:
                    st_attrs[key].append(tr_data['data'][key])
                # get planetary attributes
                for pl in tr_data['data']['planets']:
                    pl_keys = [i for i in tr_data['data'][pl].keys()
                               if not any(ext in i for ext in anccore.SV_EXTS)]
                    for key in pl_keys:
                        pl_attrs[key].append(tr_data['data'][pl][key])

        # Loop through a second group of targets.  (this subset will be overplotted in the histos)
        for trgt in targetlists['roudier62']:
            target_sample = 'roudier62'

            tr_data = aspects[trgt][svname]

            # verify SV succeeded for target
            if tr_data['STATUS'][-1] or 'planets' in tr_data['data']:
                # get stellar attributes
                st_keys = [i for i in tr_data['data'].keys()
                           if is_st_key(i, tr_data['data']['planets'])]
                for key in st_keys:
                    st_attrs_roudier62[key].append(tr_data['data'][key])
                # get planetary attributes
                for pl in tr_data['data']['planets']:
                    pl_keys = [i for i in tr_data['data'][pl].keys()
                               if not any(ext in i for ext in anccore.SV_EXTS)]
                    for key in pl_keys:
                        pl_attrs_roudier62[key].append(tr_data['data'][pl][key])

        # Add to SV
        self.__out['data']['st_attrs'] = st_attrs
        self.__out['data']['pl_attrs'] = pl_attrs
        self.__out['data']['st_attrs_roudier62'] = st_attrs_roudier62
        self.__out['data']['pl_attrs_roudier62'] = pl_attrs_roudier62
        self.__out['data']['sample'] = target_sample
        self.__out['STATUS'].append(True)
        aspects.ds().update()
        return
    pass
# ---------------- ---------------------------------------------------
# -- HELPER FUNCTIONS -- ---------------------------------------------
def is_st_key(key, planets):
    '''Helper function to determine if SV key is for stellar estimate value'''
    # GMR: Using lazy gen here otherwise the new pylint is not happy
    return (not key == 'planets') and (key not in planets) \
            and (not any(ext in key for ext in anccore.SV_EXTS))
# ---------------------- ---------------------------------------------
