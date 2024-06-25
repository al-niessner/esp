'''data algorithms dc'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.context

import excalibur.data as dat
import excalibur.data.core as datcore
import excalibur.data.states as datstates

import excalibur.target as trg
import excalibur.target.states as trgstates
import excalibur.target.algorithms as trgalg
import excalibur.system as sys
import excalibur.system.algorithms as sysalg
import excalibur.runtime.algorithms as rtalg
import excalibur.runtime.binding as rtbind
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
fltrs = [str(fn) for fn in rtbind.filter_names.values()]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class collect(dawgie.Algorithm):
    '''
    G. ROUDIER: Data collection by filters
    '''
    def __init__(self):
        '''__init__ ds'''
        self._version_ = datcore.collectversion()
        self.__create = trgalg.create()
        self.__rt = rtalg.autofill()
        self.__scrape = trgalg.scrape()
        self.__out = trgstates.FilterSV('frames')
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'collect'

    def previous(self):
        '''Input State Vectors: target.create, target.scrape'''
        return [dawgie.ALG_REF(trg.analysis, self.__create),
                dawgie.ALG_REF(trg.task, self.__scrape)] + \
                self.__rt.refs_for_proceed()

    def state_vectors(self):
        '''Output State Vectors: data.collect'''
        return [self.__out]

    def run(self, ds, ps):
        '''Top level algorithm call'''
        update = False
        create = self.__create.sv_as_dict()
        scrape = self.__scrape.sv_as_dict()['databases']
        valid, errstring = datcore.checksv(scrape)
        if valid:
            for key in create['filters'].keys(): self.__out[key] = create['filters'][key]
            for name in create['filters']['activefilters']['NAMES']:
                self.__rt.proceed(ext=name)
                ok = self._collect(name, scrape, self.__out)
                update = update or ok
                pass
            if update: ds.update()
            else: self._raisenoout(self.name())
            pass
        else: self._failure(errstring)
        return

    @staticmethod
    def _raisenoout(myname):
        '''No output exception'''
        raise dawgie.NoValidOutputDataError(f'No output created for DATA.{myname}')

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
        self.__rt = rtalg.autofill()
        self.__out = [datstates.TimingSV(ext) for ext in fltrs]
        return

    def name(self):
        '''Database name for subtask extension'''
        return 'timing'

    def previous(self):
        '''Input State Vectors: system.finalize, data.collect'''
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(dat.task, self.__col)] + \
                self.__rt.refs_for_proceed()

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
        # validtype list is necessary to filter out hidden non valid filter names
        validtype = []
        for test in col['activefilters'].keys():
            if test in fltrs: validtype.append(test)
            pass
        svupdate = []
        if vfin and vcol:
            for ext in validtype:
                self.__rt.proceed(ext)
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
        else: self._raisenoout(self.name())
        return

    @staticmethod
    def _raisenoout(myname):
        '''No output exception'''
        raise dawgie.NoValidOutputDataError(f'No output created for DATA.{myname}')

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
        self.__rt = rtalg.autofill()
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
                dawgie.ALG_REF(dat.task, self.__tim)] + \
                self.__rt.refs_for_proceed()

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
            self.__rt.proceed(datatype)
            if vfin and vcll and vtim:
                # FIXMEE: access to protected target name will be going away
                # pylint: disable=protected-access
                update = self._calib(fin, cll['activefilters'][datatype], tim, ds._tn(),
                                     datatype, self.__out[fltrs.index(datatype)], ps)
                if update: svupdate.append(self.__out[fltrs.index(datatype)])
                pass
            else:
                message = [m for m in [sfin, ecll, etim] if m is not None]
                self._failure(message[0])
                pass
            pass
        self.__out = svupdate
        if self.__out.__len__() > 0: ds.update()
        else: self._raisenoout(self.name())
        return

    @staticmethod
    def _raisenoout(myname):
        '''No output exception'''
        raise dawgie.NoValidOutputDataError(f'No output created for DATA.{myname}')

    @staticmethod
    def _calib(fin, cll, tim, tid, flttype, out, ps):
        '''Core code call'''
        log.warning('--< DATA CALIBRATION: %s >--', flttype)
        caled = False
        if 'SCAN' in flttype:
            caled = datcore.scancal(cll, tim, tid, flttype, out,
                                    verbose=False)
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
        if 'JWST' in flttype:
            caled = datcore.jwstcal(fin, cll, tim, flttype, out,
                                    ps=ps, verbose=False, debug=False)
            pass
        return caled

    @staticmethod
    def _failure(errstr):
        '''Failure log'''
        log.warning('--< DATA CALIBRATION: %s >--', errstr)
        return
    pass
# ---------------- ---------------------------------------------------
