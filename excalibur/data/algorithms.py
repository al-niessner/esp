# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context

import excalibur.data as dat
import excalibur.data.core as datcore
import excalibur.data.states as datstates

import excalibur.transit.core as trncore

import excalibur.target as trg
import excalibur.target.edit as trgedit
import excalibur.target.states as trgstates
import excalibur.target.algorithms as trgalg
import excalibur.system as sys
import excalibur.system.algorithms as sysalg
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
# fltrs = [f for f in fltrs if 'G430' in f]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class collect(dawgie.Algorithm):
    '''
G. ROUDIER: Data collection by filters
    '''
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__create = trgalg.create()
        self.__scrape = trgalg.scrape()
        self.__out = trgstates.FilterSV('frames')
        return

    def name(self):
        return 'collect'

    def previous(self):
        return [dawgie.ALG_REF(trg.analysis, self.__create),
                dawgie.ALG_REF(trg.task, self.__scrape)]

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

    @staticmethod
    def _collect(name, scrape, out):
        log.warning('--< DATA COLLECT: %s >--', name)
        collected = datcore.collect(name, scrape, out)
        return collected

    @staticmethod
    def _failure(errstr):
        log.warning('--< DATA COLLECT: %s >--', errstr)
        return
    pass

class timing(dawgie.Algorithm):
    '''
G. ROUDIER: Categorize data into 3 science purposes: TRANSIT, ECLIPSE, PHASE CURVE
    '''
    def __init__(self):
        self._version_ = datcore.timingversion()
        self.__fin = sysalg.finalize()
        self.__col = collect()
        self.__out = [datstates.TimingSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'timing'

    def previous(self):
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(dat.task, self.__col)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        update = False
        fin = self.__fin.sv_as_dict()['parameters']
        vfin, efin = datcore.checksv(fin)
        col = self.__col.sv_as_dict()['frames']
        vcol, ecol = datcore.checksv(col)
        validtype = []
        for test in col['activefilters'].keys():
            if ('SCAN' in test) or ('STIS' in test):
                if test in fltrs: validtype.append(test)
                pass
            pass
        svupdate = []
        if vfin and vcol:
            for ext in validtype:
                update = self._timing(fin, ext, col['activefilters'][ext],
                                      self.__out[fltrs.index(ext)])
                if update: svupdate.append(self.__out[fltrs.index(ext)])
                pass
            pass
        else:
            message = [m for m in [efin, ecol] if m is not None]
            self._failure(message[0])
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        else: self._failure('NO DATA')
        return

    @staticmethod
    def _timing(fin, ext, colin, out):
        log.warning('--< DATA TIMING: %s >--', ext)
        chunked = datcore.timing(fin, ext, colin, out, verbose=False)
        return chunked

    @staticmethod
    def _failure(errstr):
        log.warning('--< DATA TIMING: %s >--', errstr)
        return
    pass

class calibration(dawgie.Algorithm):
    '''
G. ROUDIER: Data re-calibration and reduction
    '''
    def __init__(self):
        self._version_ = dawgie.VERSION(1,3,0)
        self.__fin = sysalg.finalize()
        self.__col = collect()
        self.__tim = timing()
        self.__out = [datstates.CalibrateSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'calibration'

    def previous(self):
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(dat.task, self.__col),
                dawgie.ALG_REF(dat.task, self.__tim)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        update = False
        cll = self.__col.sv_as_dict()['frames']
        vcll, ecll = datcore.checksv(cll)
        fin = self.__fin.sv_as_dict()['parameters']
        vfin, sfin = trncore.checksv(fin)
        validtype = []
        for test in cll['activefilters'].keys():
            if ('SCAN' in test) or ('STARE' in test):
                if test in fltrs: validtype.append(test)
                pass
            pass
        svupdate = []
        for datatype in validtype:
            tim = self.__tim.sv_as_dict()[datatype]
            vtim, etim = datcore.checksv(tim)
            if vfin and vcll and vtim:
                # pylint: disable=protected-access
                update = self._calib(fin,cll['activefilters'][datatype], tim, ds._tn(),
                                     datatype, self.__out[fltrs.index(datatype)])
                if update: svupdate.append(self.__out[fltrs.index(datatype)])
                pass
            else:
                message = [m for m in [sfin, ecll, etim] if m is not None]
                self._failure(message[0])
                pass
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        else: self._failure('NO DATA')
        return

    @staticmethod
    def _calib(fin, cll, tim, tid, flttype, out):
        log.warning('--< DATA CALIBRATION: %s >--', flttype)
        caled = False
        if 'SCAN' in flttype:
            caled = datcore.scancal(cll, tim, tid, flttype, out, verbose=False)
            pass
        if 'G750' in flttype:
            caled = datcore.stiscal(fin, cll, tim, tid, flttype, out, verbose=False)
            pass
        if 'G430' in flttype:
            caled = datcore.stiscal_G430L(fin, cll, tim, tid, flttype, out, verbose=False)
            pass
        return caled

    @staticmethod
    def _failure(errstr):
        log.warning('--< DATA CALIBRATION: %s >--', errstr)
        return
    pass
# ---------------- ---------------------------------------------------
