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

# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# FILTERS
# fltrs = (trgedit.activefilters.__doc__).split('\n')
# fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class estimate(dawgie.Algorithm):
    def __init__(self):
        self._version_ = anccore.estimateversion()
        self._type = 'ancillary'
        self.__fin = sysalg.finalize()
        self.__out = [ancstates.EstimateSV('parameters')]
        return

    def name(self):
        return 'estimate'

    def previous(self):
        return [dawgie.ALG_REF(sys.task, self.__fin)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        update = False
        vfin, _ = anccore.checksv(self.__fin.sv_as_dict()['parameters'])
        if vfin:
            update = anccore.estimate(self.__fin.sv_as_dict()['parameters'], self.__out[0])
        if update: ds.update()

class population(dawgie.Analyzer):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,0,3)
        self.__out = ancstates.PopulationSV('statistics')
        return

    def feedback(self):
        return []

    def name(self):
        return 'population'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        return [dawgie.V_REF(anc.task, estimate(),
                             estimate().state_vectors()[0], 'data')]

    def state_vectors(self):
        return [self.__out]

    def run(self, aspects:dawgie.Aspect):
        data = aspects
        if 'as_dict' in dir(aspects):  # temporary workaround for dawgie discrepancy
            data = aspects.as_dict()
            temp = {}
            for svn in data:
                for tgn in data[svn]:
                    for vn in data[svn][tgn]:
                        if tgn not in temp: temp[tgn] = {}
                        if svn not in temp[tgn]: temp[tgn][svn] = {}
                        temp[tgn][svn][vn] = data[svn][tgn][vn]
            data = temp
        elif 'keys' not in dir(aspects):
            data = dict([i for i in aspects])
        targets = data

        # now group together values by attribute
        svname = 'ancillary.estimate.parameters'
        st_attrs = defaultdict(list)
        pl_attrs = defaultdict(list)
        for trgt in targets:
            tr_data = data[trgt][svname]
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
                               if not any([ext in i for ext in anccore.SV_EXTS])]
                    for key in pl_keys:
                        pl_attrs[key].append(tr_data['data'][pl][key])

        # Add to SV
        self.__out['data']['st_attrs'] = st_attrs
        self.__out['data']['pl_attrs'] = pl_attrs
        self.__out['STATUS'].append(True)
        aspects.ds().update()
        return
    pass
# ---------------- ---------------------------------------------------

# ---------------------- ---------------------------------------------
# -- HELPER FUNCTIONS -- ---------------------------------------------
def is_st_key(key, planets):
    # Helper function to determine if SV key is for stellar estimate value
    return (not key == 'planets') and (key not in planets) \
            and (not any([ext in key for ext in anccore.SV_EXTS]))
