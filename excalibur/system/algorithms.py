'''system algorithms ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import dawgie

import excalibur.system as sys
import excalibur.system.core as syscore
import excalibur.system.states as sysstates

import excalibur.target as trg
import excalibur.target.edit as trgedit
import excalibur.target.algorithms as trgalg
# ------------- ------------------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class validate(dawgie.Algorithm):
    '''Pulls out formatted info from NEXSCI and checks what is missing'''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,4)  # typos fixed in core/ssconstants
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
        if valid: update = self._validate(autofill, self.__out)
        else: self._failure(errstring)
        if update: ds.update()
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
        overwrite = trgedit.ppar()
        if valid:
            for key in val: self.__out[key] = val.copy()[key]
            # pylint: disable=protected-access
            if ds._tn() in overwrite:
                update = self._priority(overwrite[ds._tn()], self.__out)
                pass
            elif not self.__out['PP'][-1]: update = True
            else:
                # *** IS THIS OK? ***   (this allows age to work when blank)
                update = True
                # *** IS THIS OK? ***

                log.warning('>-- MISSING DICT INFO')
                log.warning('>-- ADD KEY TO TARGET/EDIT.PY PPAR()')
                pass
            pass
        else: self._failure(errstring)
        if update: ds.update()
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
