'''system algorithms ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import dawgie

from collections import defaultdict

import excalibur.system as sys
import excalibur.system.core as syscore
import excalibur.system.states as sysstates
import excalibur.system.overwriter as sysoverwriter

import excalibur.target as trg
import excalibur.target.edit as trgedit
import excalibur.target.algorithms as trgalg

from excalibur.system.consistency import consistency_checks
from excalibur.target.targetlists import get_target_lists
from excalibur.system.core import savesv

# ------------- ------------------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class validate(dawgie.Algorithm):
    '''Pulls out formatted info from NEXSCI and checks what is missing'''
    def __init__(self):
        '''__init__ ds'''
        # self._version_ = dawgie.VERSION(1,1,4)  # typos fixed in core/ssconstants
        self._version_ = dawgie.VERSION(1,2,0)  # new bestValue() parameter selection
        self.__autofill = trgalg.autofill()
        self.__out = sysstates.PriorsSV('parameters')
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'validate'

    def previous(self):
        '''Input State Vectors: target.autofill'''
        return [dawgie.ALG_REF(trg.task, self.__autofill)]

    def state_vectors(self):
        '''Output State Vectors: system.validate'''
        return [self.__out]

    def run(self, ds, ps):
        '''Top level algorithm call'''
        update = False
        autofill = self.__autofill.sv_as_dict()['parameters']
        valid, errstring = syscore.checksv(autofill)
        # pylint: disable=protected-access
        prcd = trgedit.proceed(ds._tn(), 'any filter', verbose=False)
        if valid and prcd:
            update = self._validate(autofill, self.__out)
        else:
            if not prcd: errstring = ['Kicked by edit.processme()']
            self._failure(errstring)
        if update: ds.update()
        elif valid and prcd: raise dawgie.NoValidOutputDataError(
                f'No output created for SYSTEM.{self.name()}')
        return

    @staticmethod
    def _validate(autofill, out):
        '''Core code call'''
        afilled = syscore.buildsp(autofill, out)
        return afilled

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< SYSTEM VALIDATE: %s >--', errstr)
        return
    pass

class finalize(dawgie.Algorithm):
    '''Generates SV used as input in other algorithm calls,
    throws out incomplete systems'''
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,4)
        self.__val = validate()
        self.__out = sysstates.PriorsSV('parameters')
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'finalize'

    def previous(self):
        '''Input State Vectors: system.validate'''
        return [dawgie.ALG_REF(sys.task, self.__val)]

    def state_vectors(self):
        '''Output State Vectors: system.finalize'''
        return [self.__out]

    def run(self, ds, ps):
        '''Top level algorithm call'''

        update = False
        val = self.__val.sv_as_dict()['parameters']
        valid, errstring = syscore.checksv(val)
        # pylint: disable=protected-access
        prcd = trgedit.proceed(ds._tn(), 'any filter', verbose=False)
        if valid and prcd:
            overwrite = sysoverwriter.ppar()
            for key in val: self.__out[key] = val.copy()[key]
            # pylint: disable=protected-access
            if ds._tn() in overwrite:
                # print('UPDATE overwrite start',ds._tn())
                update = self._priority(overwrite[ds._tn()], self.__out)
                if not update:
                    log.warning('>-- STILL MISSING DICT INFO')
                    log.warning('>-- ADD MORE KEYS TO SYSTEM/OVERWRITER')
                pass
            elif not self.__out['PP'][-1]:
                update = True
            else:
                log.warning('>-- MISSING DICT INFO')
                log.warning('>-- ADD KEY TO SYSTEM/OVERWRITER')
                pass

            # consistency checks
            inconsistencies = consistency_checks(self.__out['priors'])
            for inconsistency in inconsistencies:
                self.__out['autofill'].append('inconsistent:'+inconsistency)

            # log warnings moved to the very end (previously were before forcepar)
            log.warning('>-- FORCE PARAMETER: %s', str(self.__out['PP'][-1]))
            log.warning('>-- MISSING MANDATORY PARAMETERS: %s', str(self.__out['needed']))
            log.warning('>-- MISSING PLANET PARAMETERS: %s', str(self.__out['pneeded']))
            log.warning('>-- PLANETS IGNORED: %s', str(self.__out['ignore']))
            log.warning('>-- INCONSISTENCIES: %s', str(inconsistencies))
            log.warning('>-- AUTOFILL: %s', str(self.__out['autofill']))
            pass
        else:
            if not prcd: errstring = ['Kicked by edit.processme()']
            self._failure(errstring)

        if update: ds.update()
        elif valid and prcd: raise dawgie.NoValidOutputDataError(
                f'No output created for SYSTEM.{self.name()}')
        return

    @staticmethod
    def _priority(overwrite, out):
        '''Core code call'''
        ffill = syscore.forcepar(overwrite, out)
        return ffill

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< SYSTEM FINALIZE: %s >--', errstr)
        return
    pass
# ---------------- ---------------------------------------------------

class population(dawgie.Analyzer):
    '''population ds'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,0,3)
        self.__fin = finalize()
        self.__out = sysstates.PopulationSV('statistics')
        return

    def previous(self):
        '''Input State Vectors: system.finalize'''
        return [dawgie.ALG_REF(sys.task, self.__fin)]

    def feedback(self):
        '''feedback ds'''
        return []

    def name(self):
        '''name ds'''
        return 'population'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        '''traits ds'''
        return [dawgie.SV_REF(sys.task, finalize(),
                              finalize().state_vectors()[0])]

    def state_vectors(self):
        '''state_vectors ds'''
        return [self.__out]

    def run(self, aspects:dawgie.Aspect):
        '''run ds'''

        targetlists = get_target_lists()

        # cross-check the 'active' list against __all__
        checkAll = False
        if checkAll:
            for target in targetlists['active']:
                if target not in aspects:
                    print('missing from aspect:',target)
            for target in aspects:
                if target in targetlists['active']:
                    # print('ok')
                    pass
                elif 'taurex sim' in target:
                    # print('ok')
                    pass
                else:
                    print('missing from active',target)
            # exit()

        # group together values by attribute
        svname = 'system.finalize.parameters'

        st_attrs = defaultdict(list)
        pl_attrs = defaultdict(list)
        st_attrs_roudier62 = defaultdict(list)
        pl_attrs_roudier62 = defaultdict(list)

        # for trgt in aspects:
        # for trgt in targetlists['active']:
        for trgt in filter(lambda tgt: 'STATUS' in aspects[tgt][svname], targetlists['active']):

            system_data = aspects[trgt][svname]

            # verify SV succeeded for target
            if system_data['STATUS'][-1]:
                # get stellar attributes
                st_keys = system_data['starmdt']           # mandatory params
                st_keys.extend(system_data['starnonmdt'])  # non-mandatory params
                pl_keys = system_data['planetmdt']
                pl_keys.extend(system_data['planetnonmdt'])

                for key in st_keys:
                    st_attrs[key].append(system_data['priors'][key])

                # get planetary attributes
                for planet_letter in system_data['priors']['planets']:
                    for key in pl_keys:
                        pl_attrs[key].append(system_data['priors'][planet_letter][key])

        #  *** second loop for second (overlapping) histogram ***
        # for trgt in targetlists['roudier62']:
        for trgt in filter(lambda tgt: 'STATUS' in aspects[tgt][svname], targetlists['roudier62']):
            system_data = aspects[trgt][svname]

            # verify SV succeeded for target
            if system_data['STATUS'][-1]:
                st_keys = system_data['starmdt']           # mandatory params
                st_keys.extend(system_data['starnonmdt'])  # non-mandatory params
                pl_keys = system_data['planetmdt']
                pl_keys.extend(system_data['planetnonmdt'])

                # get stellar attributes
                for key in st_keys:
                    st_attrs_roudier62[key].append(system_data['priors'][key])
                # get planetary attributes
                for planet_letter in system_data['priors']['planets']:
                    for key in pl_keys:
                        pl_attrs_roudier62[key].append(system_data['priors'][planet_letter][key])

        # Add to SV
        self.__out['data']['st_attrs'] = st_attrs
        self.__out['data']['st_attrs_roudier62'] = st_attrs_roudier62
        self.__out['data']['pl_attrs'] = pl_attrs
        self.__out['data']['pl_attrs_roudier62'] = pl_attrs_roudier62
        self.__out['STATUS'].append(True)
        aspects.ds().update()

        # save system-finalize results as .csv file (in /proj/data/spreadsheets/)
        savesv(aspects, targetlists)

        return
    pass
# ---------------- ---------------------------------------------------
