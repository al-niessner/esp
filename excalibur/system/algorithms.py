# -- IMPORTS -- ------------------------------------------------------
import pdb

import dawgie

import excalibur.system as sys
import excalibur.system.core as syscore
import excalibur.system.states as sysstates

import excalibur.target as trg
import excalibur.target.edit as trgedit
import excalibur.target.algorithms as trgalg
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# VERBOSE AND DEBUG
verbose = False
debug = False
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class validate(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__autofill = trgalg.autofill()
        self.__out = sysstates.PriorsSV('parameters')
        return

    def name(self):
        return 'validate'

    def previous(self):
        return [dawgie.ALG_REF(trg.factory, self.__autofill)]

    def state_vectors(self):
        return [self.__out]

    def run(self, ds, ps):
        update = False
        autofill = self.__autofill.sv_as_dict()['parameters']
        valid, errstring = syscore.checksv(autofill)
        if valid: update = self._validate(autofill, self.__out)
        else: self._failure(errstring)
        if update: ds.update()
        return

    def _validate(self, autofill, out):
        afilled = syscore.buildsp(autofill, out,
                                  verbose=verbose, debug=debug)
        return afilled

    def _failure(self, errstr):
        errmess = '--< SYSTEM VALIDATE: ' + errstr + ' >--'
        if verbose: print(errmess)
        return
    pass

class finalize(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__val = validate()
        self.__out = sysstates.PriorsSV('parameters')
        return

    def name(self):
        return 'finalize'

    def previous(self):
        return [dawgie.ALG_REF(sys.factory, self.__val)]

    def state_vectors(self):
        return [self.__out]

    def run(self, ds, ps):
        update = False
        val = self.__val.sv_as_dict()['parameters']
        valid, errstring = syscore.checksv(val)
        overwrite = trgedit.ppar()
        if valid:
            for key in val: self.__out[key] = val.copy()[key]
            if ds._tn() in overwrite.keys():
                update = self._priority(overwrite[ds._tn()],
                                        self.__out)
                pass
            elif not(self.__out['PP'][-1]): update = True
            else:
                if verbose:
                    print('>-- MISSING DICT INFO')
                    print('>-- ADD KEY TO TARGET/EDIT.PY PPAR()')
                    pass
                pass
            pass
        else: self._failure(errstring)
        if update: ds.update()
        return

    def _priority(self, overwrite, out):
        ffill = syscore.forcepar(overwrite, out,
                                 verbose=verbose, debug=debug)
        return ffill

    def _failure(self, errstr):
        errmess = '--< SYSTEM FINALIZE: ' + errstr + ' >--'
        if verbose: print(errmess)
        return
    pass
# ---------------- ---------------------------------------------------
