# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import excalibur.data as dat
import excalibur.data.core as datcore
import excalibur.data.states as datstates

import excalibur.target as trg
import excalibur.target.edit as trgedit
import excalibur.target.states as trgstates
import excalibur.target.algorithms as trgalg
import excalibur.system as sys
import excalibur.system.algorithms as sysalg
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# VERBOSE AND DEBUG
verbose = False
debug = False
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '').__len__() > 0]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class collect(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__verbose = verbose
        self.__create = trgalg.create()
        self.__scrape = trgalg.scrape()
        self.__out = trgstates.FilterSV('frames')
        return

    def name(self):
        return 'collect'

    def previous(self):
        return [dawgie.ALG_REF(trg.all, self.__create),
                dawgie.ALG_REF(trg.factory, self.__scrape)]

    def state_vectors(self):
        return [self.__out]

    def run(self, ds, ps):
        update = False
        create = self.__create.sv_as_dict()['filters']
        scrape = self.__scrape.sv_as_dict()['databases']
        valid, errstring = datcore.checksv(scrape)
        if valid:
            for key in create.keys(): self.__out[key] = create[key]
            for name in create['activefilters']['NAMES']:
                ok = self._collect(name, scrape, self.__out)
                update = update or ok
                pass
            if update: ds.update()
            pass
        else: self._failure(errstring)
        return

    def _collect(self, name, scrape, out):
        collected = datcore.collect(name, scrape, out,
                                    verbose=self.__verbose, debug=debug)
        return collected

    def _failure(self, errstr):
        errmess = '--< DATA COLLECT: ' + errstr + ' >--'
        if self.__verbose: print(errmess)
        return
    pass

class calibration(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__verbose = verbose
        self.__collect = collect()
        self.__out = [datstates.CalibrateSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'calibration'

    def previous(self):
        return [dawgie.ALG_REF(dat.factory, self.__collect)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        update = False
        cll = self.__collect.sv_as_dict()['frames']
        validtype = []
        for test in cll['activefilters'].keys():
            if 'SCAN' in test: validtype.append(test)
            pass
        errstring = 'NO DATA'
        svupdate = []
        for datatype in validtype:
            collectin = cll['activefilters'][datatype]
            index = fltrs.index(datatype)
            # pylint: disable=protected-access
            update = self._calib(collectin, ds._tn(), datatype, self.__out[index])
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        else: self._failure(errstring)
        return

    def _calib(self, cll, tid, flttype, out):
        caled = datcore.scancal(cll, tid, flttype, out,
                                verbose=self.__verbose, debug=debug)
        return caled

    def _failure(self, errstr):
        errmess = '--< DATA CALIBRATION: ' + errstr + ' >--'
        if self.__verbose: print(errmess)
        return
    pass

class timing(dawgie.Algorithm):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__verbose = verbose
        self.__fin = sysalg.finalize()
        self.__calib = calibration()
        self.__out = [datstates.TimingSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'timing'

    def previous(self):
        return [dawgie.ALG_REF(sys.factory, self.__fin),
                dawgie.ALG_REF(dat.factory, self.__calib)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        fin = self.__fin.sv_as_dict()['parameters']
        valid, errstring = datcore.checksv(fin)
        svupdate = []
        for ext in fltrs:
            cal = self.__calib.sv_as_dict()[ext]
            vext, verr = datcore.checksv(cal)
            index = fltrs.index(ext)
            update = False
            if valid and vext: update = self._timing(fin, cal, self.__out[index])
            else:
                message = [m for m in [errstring, verr] if m is not None]
                self._failure(message[0])
                pass
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        return

    def _timing(self, fin, cal, out):
        chunked = datcore.timing(fin, cal, out,
                                 verbose=self.__verbose, debug=debug)
        return chunked

    def _failure(self, errstr):
        errmess = '--< DATA TIMING: ' + errstr + ' >--'
        if self.__verbose: print(errmess)
        return
    pass
# ---------------- ---------------------------------------------------
