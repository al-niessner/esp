'''data algorithms dc'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

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
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
# KICK SPITZER FOR THE MOMENT
fltrs = [f for f in fltrs if 'HST' in f]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class collect(dawgie.Algorithm):
    '''
    G. ROUDIER: Data collection by filters
    '''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,2)
        self.__create = trgalg.create()
        self.__scrape = trgalg.scrape()
        self.__out = trgstates.FilterSV('frames')
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'collect'

    def previous(self):
        '''Input State Vectors: target.create, target.scrape'''
        return [dawgie.ALG_REF(trg.analysis, self.__create),
                dawgie.ALG_REF(trg.task, self.__scrape)]

    def state_vectors(self):
        '''Output State Vectors: data.collect'''
        return [self.__out]

    def run(self, ds, ps):
        '''Top level algorithm call'''
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
        '''Core code call'''
        log.warning('--< DATA COLLECT: %s >--', name)
        collected = datcore.collect(name, scrape, out)
        return collected

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< DATA COLLECT: %s >--', errstr)
        return
    pass

class timing(dawgie.Algorithm):
    '''
    G. ROUDIER: Categorize data into 3 science purposes: TRANSIT, ECLIPSE, PHASECURVE
    '''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = datcore.timingversion()
        self.__fin = sysalg.finalize()
        self.__col = collect()
        self.__out = [datstates.TimingSV(ext) for ext in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'timing'

    def previous(self):
        '''Input State Vectors: system.finalize, data.collect'''
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(dat.task, self.__col)]

    def state_vectors(self):
        '''Output State Vectors: data.timing'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''
        update = False
        fin = self.__fin.sv_as_dict()['parameters']
        vfin, efin = datcore.checksv(fin)
        col = self.__col.sv_as_dict()['frames']
        vcol, ecol = datcore.checksv(col)
        validtype = []
        for test in col['activefilters'].keys():
            if test in fltrs: validtype.append(test)
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
        '''Core code call'''
        log.warning('--< DATA TIMING: %s >--', ext)
        chunked = datcore.timing(fin, ext, colin, out, verbose=False)
        return chunked

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< DATA TIMING: %s >--', errstr)
        return
    pass

class calibration(dawgie.Algorithm):
    '''
    G. ROUDIER: Data re-calibration and reduction
    '''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,4,4)
        self.__fin = sysalg.finalize()
        self.__col = collect()
        self.__tim = timing()
        self.__out = [datstates.CalibrateSV(ext) for ext in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'calibration'

    def previous(self):
        '''Input State Vectors: system.finalize, data.collect, data.timing'''
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(dat.task, self.__col),
                dawgie.ALG_REF(dat.task, self.__tim)]

    def state_vectors(self):
        '''Output State Vectors: data.calibration'''
        return self.__out

    def run(self, ds, ps):
        '''Top level algorithm call'''
        update = False
        cll = self.__col.sv_as_dict()['frames']
        vcll, ecll = datcore.checksv(cll)
        fin = self.__fin.sv_as_dict()['parameters']
        vfin, sfin = datcore.checksv(fin)
        validtype = []
        for test in cll['activefilters'].keys():
            if test in fltrs: validtype.append(test)
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
        '''Core code call'''
        log.warning('--< DATA CALIBRATION: %s >--', flttype)
        caled = False
        if 'SCAN' in flttype:
            caled = datcore.scancal(cll, tim, tid, flttype, out, verbose=False)
            pass
        if 'G430' in flttype:
            caled = datcore.stiscal_G430L(fin, cll, tim, tid, flttype, out,
                                          verbose=False)
            pass
        if 'G750' in flttype:
            caled = datcore.stiscal_G750L(fin, cll, tim, tid, flttype, out,
                                          verbose=False)
        if 'Spitzer' in flttype:
            caled = datcore.spitzercal(cll, out)
            pass
        if 'NIRISS' in flttype:
            caled = datcore.jwstcal_NIRISS(fin, cll, tim, tid, flttype, out,
                                           verbose=False)
            pass
        return caled

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< DATA CALIBRATION: %s >--', errstr)
        return
    pass
# ---------------- ---------------------------------------------------
