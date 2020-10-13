# -- IMPORTS -- ------------------------------------------------------
import dawgie
import dawgie.context

import logging; log = logging.getLogger(__name__)

from collections import defaultdict
import numpy as np

import excalibur.transit as trn
import excalibur.transit.core as trncore
import excalibur.transit.states as trnstates

import excalibur.data as dat
import excalibur.data.algorithms as datalg
import excalibur.system as sys
import excalibur.system.algorithms as sysalg
import excalibur.target.edit as trgedit
# ------------- ------------------------------------------------------
# -- ALGO RUN OPTIONS -- ---------------------------------------------
# FILTERS
fltrs = (trgedit.activefilters.__doc__).split('\n')
fltrs = [t.strip() for t in fltrs if t.replace(' ', '')]
# fltrs = [f for f in fltrs if 'Spitzer' not in f]
# fltrs = [f for f in fltrs if 'JWST' not in f]
# fltrs = [f for f in fltrs if 'HST' not in f]
# ---------------------- ---------------------------------------------
# -- ALGORITHMS -- ---------------------------------------------------
class normalization(dawgie.Algorithm):
    '''
    G. ROUDIER: Light curve normalization by Out Of Transit data
    '''
    def __init__(self):
        self._version_ = trncore.normversion()
        self._type = 'transit'
        self.__cal = datalg.calibration()
        self.__tme = datalg.timing()
        self.__fin = sysalg.finalize()
        self.__out = [trnstates.NormSV(ext) for ext in fltrs]
        return

    def name(self):
        return 'normalization'

    def previous(self):
        return [dawgie.ALG_REF(dat.task, self.__cal),
                dawgie.ALG_REF(dat.task, self.__tme),
                dawgie.ALG_REF(sys.task, self.__fin)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        vfin, sfin = trncore.checksv(self.__fin.sv_as_dict()['parameters'])
        for ext in fltrs:
            update = False
            vcal, scal = trncore.checksv(self.__cal.sv_as_dict()[ext])
            vtme, stme = trncore.checksv(self.__tme.sv_as_dict()[ext])
            if vcal and vtme and vfin:
                log.warning('--< %s NORMALIZATION: %s >--', self._type.upper(), ext)
                update = self._norm(self.__cal.sv_as_dict()[ext],
                                    self.__tme.sv_as_dict()[ext],
                                    self.__fin.sv_as_dict()['parameters'],
                                    fltrs.index(ext))
                pass
            else:
                errstr = [m for m in [scal, stme, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[fltrs.index(ext)])
            pass
        self.__out = svupdate
        if self.__out: ds.update()
        return

    def _norm(self, cal, tme, fin, index):
        if 'Spitzer' in fltrs[index]:
            normed = trncore.norm_spitzer(cal, tme, fin, self.__out[index], self._type)
            pass
        elif 'JWST' in fltrs[index]:
            normed = trncore.norm_jwst_niriss(cal, tme, fin, self.__out[index], self._type)
        else:
            normed = trncore.norm(cal, tme, fin, fltrs[index], self.__out[index],
                                  self._type, verbose=False)
            pass
        return normed

    def _failure(self, errstr):
        log.warning('--< %s NORMALIZATION: %s >--', self._type.upper(), errstr)
        return
    pass

class whitelight(dawgie.Algorithm):
    '''
    G. ROUDIER: See inheritance and CI5 thread with A NIESSNER
    for __init__() method and class attributes
    https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/pull/86
    '''
    def __init__(self, nrm=normalization()):
        self._version_ = trncore.wlversion()
        self._type = 'transit'
        self._nrm = nrm
        self.__fin = sysalg.finalize()
        self.__out = [trnstates.WhiteLightSV(ext) for ext in fltrs]
        self.__out.append(trnstates.WhiteLightSV('HST'))
        return

    def name(self):
        return 'whitelight'

    def previous(self):
        return [dawgie.ALG_REF(trn.task, self._nrm),
                dawgie.ALG_REF(sys.task, self.__fin)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        fin = self.__fin.sv_as_dict()['parameters']
        vfin, sfin = trncore.checksv(fin)
        # MERGE PROTOTYPE
        if self._type == "transit":
            allnormdata = []
            allext = []
            hstfltrs = ['HST-WFC3-IR-G141-SCAN', 'HST-WFC3-IR-G102-SCAN',
                        'HST-STIS-CCD-G750L-STARE', 'HST-STIS-CCD-G430L-STARE']
            for ext in hstfltrs:
                update = False
                try:
                    nrm = self._nrm.sv_as_dict()[ext]
                except KeyError:
                    break
                vnrm, snrm = trncore.checksv(nrm)
                if vnrm and vfin:
                    log.warning('--< %s MERGING: %s >--', self._type.upper(), ext)
                    allnormdata.append(nrm)
                    allext.append(ext)
                    update = True
                    pass
                else:
                    errstr = [m for m in [snrm, sfin] if m is not None]
                    self._failure(errstr[0])
                    pass
                pass
            if allnormdata:
                try:
                    update = self._hstwhitelight(allnormdata, fin, self.__out[-1], allext)
                    if update: svupdate.append(self.__out[-1])
                except TypeError:
                    log.warning('>-- HST orbit solution failed %s', self._type.upper())
                    pass
                pass
        # FILTER LOOP
        for ext in fltrs:
            update = False
            index = fltrs.index(ext)
            nrm = self._nrm.sv_as_dict()[ext]
            vnrm, snrm = trncore.checksv(nrm)
            if vnrm and vfin:
                log.warning('--< %s WHITE LIGHT: %s >--', self._type.upper(), ext)
                update = self._whitelight(nrm, fin, self.__out[index], ext)
                pass
            else:
                errstr = [m for m in [snrm, sfin] if m is not None]
                self._failure(errstr[0])
                pass
            if update: svupdate.append(self.__out[index])
            pass
        self.__out = svupdate
        if self.__out: ds.update()
        return

    def _hstwhitelight(self, nrm, fin, out, ext):
        wl = trncore.hstwhitelight(nrm, fin, out, ext, self._type,
                                   chainlen=int(1e4), verbose=False)
        return wl

    def _whitelight(self, nrm, fin, out, ext):
        if 'Spitzer' in ext:
            wl = trncore.lightcurve_spitzer(nrm, fin, out, self._type, ext,
                                            self.__out[-1])
        elif 'JWST' in ext:
            wl = trncore.lightcurve_jwst_niriss(nrm, fin, out, self._type, ext, self.__out[-1], method='ns')
        else:
            wl = trncore.whitelight(nrm, fin, out, ext, self._type,
                                    self.__out[-1], chainlen=int(1e4), verbose=False,
                                    parentprior=True)
            pass
        return wl

    def _failure(self, errstr):
        log.warning('--< %s WHITE LIGHT: %s >--', self._type.upper(), errstr)
        return
    pass

class spectrum(dawgie.Algorithm):
    '''
    G. ROUDIER: See inheritance and CI5 thread with A NIESSNER for __init__() method and class attributes https://github-fn.jpl.nasa.gov/EXCALIBUR/esp/pull/86
    '''
    def __init__(self, nrm=normalization(), wht=whitelight()):
        self._version_ = trncore.spectrumversion()
        self._type = 'transit'
        self.__fin = sysalg.finalize()
        self._nrm = nrm
        self._wht = wht
        self.__out = [trnstates.SpectrumSV(ext) for ext in fltrs]
        # MERGE PROTOTYPE
        self.__out.append(trnstates.SpectrumSV('Composite'))
        return

    def name(self):
        return 'spectrum'

    def previous(self):
        return [dawgie.ALG_REF(sys.task, self.__fin),
                dawgie.ALG_REF(trn.task, self._nrm),
                dawgie.ALG_REF(trn.task, self._wht)]

    def state_vectors(self):
        return self.__out

    def run(self, ds, ps):
        svupdate = []
        vfin, sfin = trncore.checksv(self.__fin.sv_as_dict()['parameters'])
        for index, ext in enumerate(fltrs):
            update = False
            vnrm, snrm = trncore.checksv(self._nrm.sv_as_dict()[ext])
            vwht, swht = trncore.checksv(self._wht.sv_as_dict()[ext])
            if vfin and vnrm and vwht:
                log.warning('--< %s SPECTRUM: %s >--', self._type.upper(), ext)
                update = self._spectrum(self.__fin.sv_as_dict()['parameters'],
                                        self._nrm.sv_as_dict()[ext],
                                        self._wht.sv_as_dict()[ext],
                                        self.__out[index], ext)
                pass
            else:
                errstr = [m for m in [sfin, snrm, swht] if m is not None]
                self._failure(errstr[0])
                pass

            if update: svupdate.append(self.__out[index])
            pass

        for index, ext in enumerate(fltrs):
            self.__out[-1]['data'][ext] = self.__out[index]
            if self.__out[index]['STATUS']:
                self.__out[-1]['STATUS'] = True

        self.__out = svupdate
        if self.__out: ds.update()
        return

    def _spectrum(self, fin, nrm, wht, out, ext):
        if "Spitzer" in ext:
            s = trncore.spitzer_spectrum(wht, out, ext)
        elif "JWST" in ext:
            s = trncore.jwst_niriss_spectrum(nrm, fin, out, self._type, wht, method='lm')
        else:
            s = trncore.spectrum(fin, nrm, wht, out, ext, self._type, chainlen=int(1e4),
                                 verbose=False)
            pass
        return s

    def _failure(self, errstr):
        log.warning('--< %s SPECTRUM: %s >--', self._type.upper(), errstr)
        return
    pass

class population(dawgie.Analyzer):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,0,2)
        self.__out = [trnstates.PopulationSV(ext) for ext in fltrs]
        return

    def feedback(self):
        return []

    def name(self):
        return 'population'

    def traits(self)->[dawgie.SV_REF, dawgie.V_REF]:
        return [*[dawgie.SV_REF(trn.task, spectrum(), spectrum().state_vectors()[i])
                  for i in range(len(spectrum().state_vectors()))],
                *[dawgie.SV_REF(trn.task, whitelight(), whitelight().state_vectors()[i])
                  for i in range(len(whitelight().state_vectors()))]]

    def state_vectors(self):
        return self.__out

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

        # now handle IM parameter distribution
        banned_params = ['rprs']
        sv_prefix = 'transit.spectrum.'
        wl_prefix = 'transit.whitelight.'
        for idx, fltr in enumerate(fltrs):
            im_bins = defaultdict(lambda: defaultdict(list))
            wl_im_bins = defaultdict(lambda: defaultdict(list))
            for trgt in targets:
                svname = sv_prefix + fltr
                wlname = wl_prefix + fltr
                if svname not in data[trgt] or wlname not in data[trgt]:
                    continue
                tr_data = data[trgt][svname]
                wl_data = data[trgt][wlname]
                for pl in tr_data['data']:
                    # verify SV succeeded for target
                    if tr_data['STATUS'][-1] or 'MCTRACE' in tr_data['data'][pl]:
                        # logic for saving spectrum IM parameters
                        if 'MCTRACE' in tr_data['data'][pl]:
                            if pl in wl_data['data']:
                                num_visits = len(wl_data['data'][pl]['visits'])
                            else:  # always at least one visit
                                num_visits = 1
                            trace = tr_data['data'][pl]['MCTRACE']
                            for row in trace:
                                for key in row:
                                    keyparts = key.split('__')
                                    if len(keyparts) > 1 and int(keyparts[1]) >= num_visits:
                                        continue
                                    param_name = keyparts[0]
                                    if param_name in banned_params:
                                        continue
                                    param_val = np.nanmedian(row[key])
                                    im_bins[param_name]['values'].append(param_val)
                                    bins = 60
                                    if 'G430' in fltr:
                                        bins = 30
                                    elif 'G750' in fltr:
                                        bins = 40
                                    elif 'G102' in fltr:
                                        bins = 30
                                    im_bins[param_name]['bins'] = bins
                    if pl not in wl_data['data']:
                        continue
                    if wl_data['STATUS'][-1] or 'mctrace' in wl_data['data'][pl]:
                        # logic for saving whitelight IM parameters
                        if 'mctrace' in wl_data['data'][pl]:
                            if 'visits' not in wl_data['data'][pl]:
                                continue
                            num_visits = len(wl_data['data'][pl]['visits'])
                            trace = wl_data['data'][pl]['mctrace']
                            for key in trace:
                                keyparts = key.split('__')
                                if len(keyparts) > 1 and int(keyparts[1]) >= num_visits:
                                    continue
                                param_name = keyparts[0]
                                if param_name in banned_params:
                                    continue
                                param_val = np.nanmedian(trace[key])
                                wl_im_bins[param_name]['values'].append(param_val)
            self.__out[idx]['data']['IMPARAMS'] = dict(im_bins)
            self.__out[idx]['data']['wl'] = dict()
            self.__out[idx]['data']['wl']['imparams'] = dict(wl_im_bins)
            self.__out[idx]['STATUS'].append(True)

        aspects.ds().update()
        return
    pass
# -------------------------------------------------------------------
