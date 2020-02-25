# -- IMPORTS -- ------------------------------------------------------
import io

import dawgie

import excalibur
from excalibur.transit.core import spitzer_lightcurve, composite_spectrum

import numpy as np
import matplotlib.pyplot as plt
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class NormSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            for p in self['data'].keys():
                visitor.add_declaration('PLANET: ' + p)
                for v, m in zip(self['data'][p]['vignore'], self['data'][p]['trial']):
                    strignore = str(int(v)) + ' ' + m
                    visitor.add_declaration('VISIT: ' + strignore)
                    pass
                vrange = self['data'][p]['vrange']
                for index, v in enumerate(self['data'][p]['visits']):
                    wave = self['data'][p]['wave'][index]
                    nspec = self['data'][p]['nspec'][index]
                    myfig = plt.figure()
                    plt.title('Visit: '+str(v))
                    for w,s in zip(wave, nspec):
                        select = (w > np.min(vrange)) & (w < np.max(vrange))
                        plt.plot(w[select], s[select], 'o')
                        pass
                    plt.ylabel('Normalized Spectra')
                    plt.xlabel('Wavelength [$\\mu$m]')
                    plt.xlim(np.min(vrange), np.max(vrange))
                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    visitor.add_image('...', ' ', buf.getvalue())
                    plt.close(myfig)
                    pass
                pass
            pass
        pass
    pass

class WhiteLightSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            if 'HST' in self.__name:
                mergesv = bool(self.__name == 'HST')
                for p in self['data'].keys():
                    visits = self['data'][p]['visits']
                    phase = self['data'][p]['phase']
                    allwhite = self['data'][p]['allwhite']
                    postim = self['data'][p]['postim']
                    postphase = self['data'][p]['postphase']
                    postlc = self['data'][p]['postlc']
                    postflatphase = self['data'][p]['postflatphase']
                    if mergesv: allfltrs = self['data'][p]['allfltrs']
                    myfig = plt.figure(figsize=(10, 6))
                    plt.title(p)
                    for index, v in enumerate(visits):
                        plt.plot(phase[index], allwhite[index], 'k+')
                        if mergesv:
                            vlabel = allfltrs[index]
                            plt.plot(postphase[index], allwhite[index]/postim[index], 'o',
                                     label=vlabel)
                            pass
                        else:
                            plt.plot(postphase[index], allwhite[index]/postim[index], 'o',
                                     label=str(v))
                            pass
                        pass
                    if len(visits) > 14: ncol = 2
                    else: ncol = 1
                    plt.plot(postflatphase, postlc, '^', label='M')
                    plt.xlabel('Orbital Phase')
                    plt.ylabel('Normalized Post White Light Curve')
                    if mergesv:
                        plt.legend(loc='best', ncol=ncol, mode='expand', numpoints=1,
                                   borderaxespad=0., frameon=False)
                        plt.tight_layout()
                        pass
                    else:
                        plt.legend(bbox_to_anchor=(1 + 0.1*(ncol - 0.5), 0.5),
                                   loc=5, ncol=ncol, mode='expand', numpoints=1,
                                   borderaxespad=0., frameon=False)
                        plt.tight_layout(rect=[0,0,(1 - 0.1*ncol),1])
                        pass
                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    visitor.add_image('...', ' ', buf.getvalue())
                    plt.close(myfig)
            elif 'Spitzer' in self.__name:
                # for each planet
                for p in self['data'].keys():
                    # for each event
                    for i in range(len(self['data'][p])):
                        fig = spitzer_lightcurve(self['data'][p][i])
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        visitor.add_image('...', ' ', buf.getvalue())
                        plt.close(fig)

class SpectrumSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            if self.__name == "Composite":
                plist = []
                for f in self['data'].keys():
                    if self['data'][f]['data'].keys():
                        plist.extend(list(self['data'][f]['data'].keys()))
                for p in plist:
                    try:
                        fig = composite_spectrum(self['data'], 'Composite Spectrum', p)
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        visitor.add_image('...', ' ', buf.getvalue())
                        plt.close(fig)
                    except KeyError:
                        pass
            else:
                for p in self['data'].keys():
                    if 'Teq' in self['data'][p]:
                        Teq = str(int(self['data'][p]['Teq']))
                        pass
                    else: Teq = ''
                    vspectrum = np.array(self['data'][p]['ES'])
                    specerr = np.array(self['data'][p]['ESerr'])
                    specwave = np.array(self['data'][p]['WB'])
                    specerr = abs(vspectrum**2 - (vspectrum + specerr)**2)
                    vspectrum = vspectrum**2
                    # Smooth spectrum
                    binsize = 4
                    nspec = int(specwave.size/binsize)
                    minspec = np.nanmin(specwave)
                    maxspec = np.nanmax(specwave)
                    scale = (maxspec - minspec)/(1e0*nspec)
                    wavebin = scale*np.arange(nspec) + minspec
                    deltabin = np.diff(wavebin)[0]
                    cbin = wavebin + deltabin/2e0
                    specbin = []
                    errbin = []
                    for eachbin in cbin:
                        select = specwave < (eachbin + deltabin/2e0)
                        select = select & (specwave >= (eachbin - deltabin/2e0))
                        select = select & np.isfinite(vspectrum)
                        if np.sum(np.isfinite(vspectrum[select])) > 0:
                            specbin.append(np.nansum(vspectrum[select]/(specerr[select]**2))/np.nansum(1./(specerr[select]**2)))
                            errbin.append(np.nanmedian((specerr[select]))/np.sqrt(np.sum(select)))
                            pass
                        else:
                            specbin.append(np.nan)
                            errbin.append(np.nan)
                            pass
                        pass
                    waveb = np.array(cbin)
                    specb = np.array(specbin)
                    errb = np.array(errbin)
                    myfig, ax = plt.subplots(figsize=(8,6))
                    plt.title(p+' '+Teq)
                    ax.errorbar(specwave, 1e2*vspectrum,
                                fmt='.', yerr=1e2*specerr, color='lightgray')
                    ax.errorbar(waveb, 1e2*specb,
                                fmt='^', yerr=1e2*errb, color='blue')
                    plt.xlabel(str('Wavelength [$\\mu m$]'))
                    plt.ylabel(str('$(R_p/R_*)^2$ [%]'))
                    if ('Hs' in self['data'][p]) and ('RSTAR' in self['data'][p]):
                        rp0hs = np.sqrt(np.nanmedian(vspectrum))
                        Hs = self['data'][p]['Hs'][0]
                        # Retro compatibility for Hs in [m]
                        if Hs > 1: Hs = Hs/(self['data'][p]['RSTAR'][0])
                        ax2 = ax.twinx()
                        ax2.set_ylabel('$\\Delta$ [Hs]')
                        axmin, axmax = ax.get_ylim()
                        ax2.set_ylim((np.sqrt(1e-2*axmin) - rp0hs)/Hs,
                                     (np.sqrt(1e-2*axmax) - rp0hs)/Hs)
                        myfig.tight_layout()
                        pass
                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    visitor.add_image('...', ' ', buf.getvalue())
                    plt.close(myfig)
                    pass
            pass
        pass
    pass
# -------- -----------------------------------------------------------
