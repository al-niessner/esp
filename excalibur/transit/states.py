'''Transit Database Products View'''
# -- IMPORTS -- ------------------------------------------------------
import io

import dawgie

import excalibur
from excalibur.transit.core import composite_spectrum, jwst_lightcurve

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy, norm, t
from scipy.interpolate import interp1d
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class NormSV(dawgie.StateVector):
    '''transit.normalization view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller:excalibur.identity, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            for p in self['data'].keys():
                visitor.add_declaration('PLANET: ' + p)
                for v, m in zip(self['data'][p]['vignore'], self['data'][p]['trial']):
                    strignore = str(int(v)) + '  ' + m
                    visitor.add_declaration('VISIT: ' + strignore)
                    pass
                vrange = self['data'][p]['vrange']
                for index, v in enumerate(self['data'][p]['visits']):

                    if 'plot_normalized_byvisit' in self['data'][p].keys():
                        textlabel = '--------- Visit: '+str(v)+' ---------'
                        if index >= len(self['data'][p]['plot_normalized_byvisit']):
                            # this is an error!  should be a plot for each visit!
                            pass
                        else:
                            visitor.add_image('...', textlabel,
                                              self['data'][p]['plot_normalized_byvisit'][index])

                    # keep the on-the-fly plotting, so that older RUNIDs still work
                    else:
                        wave = self['data'][p]['wave'][index]
                        nspec = self['data'][p]['nspec'][index]
                        myfig = plt.figure()
                        plt.title('Visit: '+str(v))
                        for w,s in zip(wave, nspec):
                            select = (w > np.min(vrange)) & (w < np.max(vrange))
                            plt.plot(w[select], s[select], 'o')
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
    '''transit.whitelight view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,3)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller:excalibur.identity, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            if 'HST' in self.__name:
                mergesv = bool(self.__name == 'HST')
                for p in self['data'].keys():
                    visits = np.array(self['data'][p]['visits'])
                    # phase,allwhite is the data before shifting
                    phase = np.array(self['data'][p]['phase'])
                    allwhite = np.array(self['data'][p]['allwhite'])
                    # postphase,allwhite/postim is the data after shifting
                    postphase = np.array(self['data'][p]['postphase'])
                    postim = np.array(self['data'][p]['postim'])
                    # phase,postlc is the model
                    postflatphase = np.array(self['data'][p]['postflatphase'])
                    postlc = np.array(self['data'][p]['postlc'])

                    # myfig = plt.figure(figsize=(10, 6))
                    # plt.title('Planet '+p, fontsize=14)
                    myfig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                                     gridspec_kw={'height_ratios': [3, 1]})
                    myfig.subplots_adjust(hspace=0.03, right=0.8)
                    ax1.set_title('Planet '+p, fontsize=16)
                    # ax1.set_xlabel('Orbital Phase', fontsize=14)
                    ax1.set_ylabel('Normalized Post-Whitelight Curve', fontsize=14)
                    for index, v in enumerate(visits):
                        # plot the normalized/shifted data
                        if mergesv:
                            vlabel = self['data'][p]['allfltrs'][index]
                        else:
                            vlabel = 'visit '+str(v)
                        ax1.plot(postphase[index], allwhite[index]/postim[index], 'o',
                                 zorder=3, label=vlabel)

                        # plot the pre-correction data
                        if index==len(visits)-1:
                            ax1.plot(phase[index], allwhite[index], 'k+', zorder=2,
                                     label='pre-correction')
                        else:
                            ax1.plot(phase[index], allwhite[index], 'k+', zorder=2,)

                        # add a lower panel showing the data-model residuals
                        model_interpolator = interp1d(postflatphase[np.argsort(postflatphase)],
                                                      postlc[np.argsort(postflatphase)],
                                                      kind='linear', fill_value='extrapolate')
                        model_at_observed_time = model_interpolator(postphase[index])
                        residuals = allwhite[index]/postim[index] - model_at_observed_time

                        ax2.plot(postphase[index], residuals,'o', color='black', markerfacecolor='None')
                        ax2.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
                        ax2.set_xlabel('Orbital Phase', fontsize=14)
                        ax2.set_ylabel('Residuals', fontsize=14)

                    xdatarange = ax1.get_xlim()
                    # use full model (not just at some points) if available
                    if 'modellc' in self['data'][p].keys():
                        modelphase = np.array(self['data'][p]['modelphase'])
                        modellc = np.array(self['data'][p]['modellc'])
                        # the points are ordered by time, not by phase
                        #  so sorting is needed for multi-visit observations
                        # otherwise you get weird wrap-around in the line plots
                        ax1.plot(modelphase[np.argsort(modelphase)],
                                 modellc[np.argsort(modelphase)],
                                 '-', c='k', marker='None', zorder=1,
                                 label='model')
                        # model phases only go from -0.5 to 0.5 (not good for eclipse)
                        # plot the model line a second time, but shifting the phases over by 1
                        ax1.plot(modelphase[np.argsort(modelphase)] + 1,
                                 modellc[np.argsort(modelphase)],
                                 '-', c='k', marker='None', zorder=1)
                    else:
                        ax1.plot(postflatphase, postlc,
                                 '^', zorder=1, label='model')
                        # '^', c='green', zorder=1, label='model')
                    ax1.set_xlim(xdatarange)

                    if len(visits) > 14: ncol = 2
                    else: ncol = 1
                    if mergesv:
                        ax1.legend(loc='best', ncol=ncol, mode='expand', numpoints=1,
                                   borderaxespad=0., frameon=False)
                        # myfig.tight_layout()
                    else:
                        ax1.legend(bbox_to_anchor=(1 + 0.1*(ncol - 0.5), 0.5),
                                   loc=5, ncol=ncol, mode='expand', numpoints=1,
                                   borderaxespad=0., frameon=False)
                        # myfig.tight_layout(rect=[0,0,(1 - 0.1*ncol),1])

                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    visitor.add_image('...', ' ', buf.getvalue())
                    plt.close(myfig)
            elif 'Spitzer' in self.__name:
                # for each planet
                for p in self['data'].keys():
                    # for each event
                    for i in range(len(self['data'][p])):
                        # plots are saved into sv
                        visitor.add_image('...', ' ', self['data'][p][i]['plot_bestfit'])
                        visitor.add_image('...', ' ', self['data'][p][i]['plot_residual_fft'])
                        visitor.add_image('...', ' ', self['data'][p][i]['plot_posterior'])
                        visitor.add_image('...', ' ', self['data'][p][i]['plot_pixelmap'])

            elif 'JWST' in self.__name:
                # for each planet
                for p in self['data'].keys():
                    # for each event
                    for i in range(len(self['data'][p])):
                        # light curve fit
                        fig = jwst_lightcurve(self['data'][p][i])
                        buf = io.BytesIO()
                        fig.savefig(buf, format='png')
                        visitor.add_image('...', ' ', buf.getvalue())
                        plt.close(fig)

class SpectrumSV(dawgie.StateVector):
    '''transit.spectrum view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller:excalibur.identity, visitor:dawgie.Visitor)->None:
        '''view ds'''
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
                    visitor.add_declaration('PLANET: ' + p)
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
                    plt.xlabel(str('Wavelength [$\\mu$m]'))
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
                    # now display unmasked spectrum
                    if 'MCPOST' in self['data'][p]:
                        myfig, ax = plt.subplots(figsize=(8,6))
                        plt.title('Unmasked Spectrum')
                        plt.xlabel(str('Wavelength [$\\mu$m]'))
                        plt.ylabel(str('$(R_p/R_*)^2$ [%]'))
                        buf = io.BytesIO()
                        fullspec = [mc['mean']['rprs'] if np.isnan(rp) else rp
                                    for mc, rp in zip(self['data'][p]['MCPOST'],
                                                      self['data'][p]['ES'])]
                        fullspec = np.array(fullspec)
                        fullspecerr = np.array(self['data'][p]['ESerr'])
                        fullspecwave = np.array(self['data'][p]['WB'])
                        fullspecerr = abs(fullspec**2 - (fullspec + fullspecerr)**2)
                        fullspec = fullspec**2
                        ax.errorbar(fullspecwave, 1e2*fullspec,
                                    fmt='.', yerr=1e2*fullspecerr)
                        # highight high variance channels
                        if 'LCFIT' in self['data'][p]:
                            spec_rsdpn = [np.nanstd(i['residuals'])/i['dnoise']
                                          for i in self['data'][p]['LCFIT']]
                            mask = upper_outlier_mask(spec_rsdpn)
                            if sum(mask) > 0:
                                ax.errorbar(fullspecwave[mask], 1e2*fullspec[mask],
                                            fmt='.', yerr=1e2*fullspecerr[mask],
                                            label='High Light Curve Residual')
                                ax.legend()
                        # show right axis in scale height
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
                    # now display completion plot
                    if 'Hs' in self['data'][p]:
                        Hs = self['data'][p]['Hs'][0]
                        if Hs > 1: Hs = Hs/(self['data'][p]['RSTAR'][0])
                        rp0hs = np.sqrt(np.nanmedian(vspectrum))
                        rprs_in_Hs = abs(np.sqrt(vspectrum) - rp0hs)/Hs
                        # sort non-nan values
                        sorted_Hs = sorted([i for i in rprs_in_Hs if not np.isnan(i)])
                        num_spec = len(sorted_Hs)
                        max_Hs = max(sorted_Hs)
                        # percent wizardry to make stepped graph
                        percent = [(q/num_spec)*100 for i in range(num_spec)
                                   for q in [i,i+1]]
                        # now duplicate values to make tiered step graph
                        sorted_Hs = [i for i in sorted_Hs for j in [0, 0]]
                        # now ensure completion of graph even if > 5 Hs elements
                        sorted_Hs.append(max_Hs)
                        percent.append(percent[-1])
                        myfig, ax = plt.subplots(figsize=(8,6))
                        perc_rejected = len([i for i in vspectrum if np.isnan(i)]) / \
                                            len(vspectrum)*100
                        plt.title(f'Cumulative Spectrum Distribution in Hs ({perc_rejected:.1f}% rejected)')
                        ax.plot(sorted_Hs, percent)
                        ax.axvline(4,0,c='black')
                        plt.xlabel(str('Absolute Modulation [Hs]'))
                        plt.ylabel(str('Percent [%]'))
                        plt.ylim((0, 100))
                        buf = io.BytesIO()
                        myfig.savefig(buf, format='png')
                        visitor.add_image('...', ' ', buf.getvalue())
                        plt.close(myfig)
                    if 'LCFIT' in self['data'][p]:
                        spec_rsdpn = [np.nanstd(i['residuals'])/i['dnoise']
                                      for i in self['data'][p]['LCFIT']]
                        avg_rsdpn = np.nanmean(spec_rsdpn)
                        rsdm_msg = 'RSDM is the standard deviation of the light curve residuals normalized to units of theoretical shot noise.'
                        visitor.add_primitive(f'Average RSDM: {avg_rsdpn} ({rsdm_msg})')
                        myfig, ax = plt.subplots(figsize=(8,6))
                        plt.title('RSDM by Wavelength')
                        ax.plot(self['data'][p]['WB'], spec_rsdpn, 'o')
                        plt.xlabel(str('Wavelength [$\\mu$m]'))
                        plt.ylabel(str('Residual Standard Deviation [Shot Noise]'))
                        buf = io.BytesIO()
                        myfig.savefig(buf, format='png')
                        visitor.add_image('...', ' ', buf.getvalue())
                        plt.close(myfig)
                    if 'LCFIT' in self['data'][p]:  # correlated noise analysis
                        residuals = [i['residuals'] for i in self['data'][p]['LCFIT']]
                        max_depth = 10
                        recursive_resids = merge_mean(residuals, max_depth)
                        x = [2**i for i in range(len(recursive_resids))]
                        resid_std = [np.nanmean([np.nanstd(resids) for resids in row])
                                     for row in recursive_resids]
                        resid_std = np.array(resid_std)
                        relative_expec = [resid_std[0]/np.sqrt(i) for i in x]
                        all_shot = [row['dnoise'] for row in self['data'][p]['LCFIT']]
                        shot_noise_est = np.mean(all_shot)
                        shot_noise_expec = [shot_noise_est/np.sqrt(i) for i in x]
                        myfig, ax = plt.subplots(1, 2, figsize=(8,5))
                        ax[0].set_title('LC Residual SD Behavior')
                        ax[0].set_xlabel('# synthetic channels averaged')
                        ax[0].set_ylabel('Residual Std.')
                        ax[0].plot(x, resid_std, '-o', label='Observed')
                        ax[0].plot(x, relative_expec, c='black', label='Expected (Relative)')
                        ax[0].plot(x, shot_noise_expec, c='purple', label='Expected (Shot Noise)')
                        ax[0].set_xscale('log', base=2)
                        ax[0].legend()
                        ax[1].set_title('Observed to Expected SD Ratio')
                        ax[1].set_xlabel('# synthetic channels averaged')
                        ax[1].set_ylabel('Standard Deviation Ratio')
                        ax[1].plot(x, resid_std/np.array(relative_expec),
                                   label='Empirical Expected Ratio')
                        ax[1].plot(x, resid_std/np.array(shot_noise_expec),
                                   label='Shot Noise Ratio')
                        ax[1].set_xscale('log', base=2)
                        ax[1].legend()
                        buf = io.BytesIO()
                        myfig.savefig(buf, format='png')
                        visitor.add_image('...', ' ', buf.getvalue())
                        plt.close(myfig)
                    pass
            pass
        pass
    pass

class StarspotSV(dawgie.StateVector):
    '''transit.spectrum view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller:excalibur.identity, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            for p in self['data'].keys():
                visitor.add_declaration('PLANET: ' + p)
                visitor.add_image('...',
                                  '------ starspot spectrum for planet '+p+' ------',
                                  self['data'][p]['plot_starspot_spectrum'])
                visitor.add_image('...',
                                  '------ limb darkening for planet '+p+' ------',
                                  self['data'][p]['plot_limbCoeffs'])

class PopulationSV(dawgie.StateVector):
    '''transit.population view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,0,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller:excalibur.identity, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if 'IMPARAMS' in self['data']:
            visitor.add_primitive('Spectrum Instrument Model Distribution')
            for key in self['data']['IMPARAMS']:
                data = self['data']['IMPARAMS'][key]
                bins = data['bins'] if 'bins' in data else 'auto'
                distrplot(key, data['values'], visitor, fit_t=True,
                          bins=bins)
        else:
            visitor.add_primitive('No instrument model statistics detected.')
    pass
# ------------- ------------------------------------------------------
# -- HELPER FUNCTIONS ------------------------------------------------
def merge_mean(data, depth=0):
    '''merge_mean ds'''
    # base cases
    if depth == 0 or len(data) < 2:
        return [data]
    # now recursively build list of increasingly averaged data
    modified_data = []
    for idx in range(int(len(data)/2)):
        first = idx*2
        sec = first+1
        modified_data.append([np.mean([a, b]) for a, b in zip(data[first], data[sec])])
    return [data] + merge_mean(modified_data, depth-1)

def mad(data):
    '''mad ds'''
    median = np.median(data)
    diff = np.abs(data - median)
    mad_est = np.median(diff)
    return mad_est

def calculate_bounds(data, z_thresh=3.5):
    '''computes outlier cutoffs'''
    MAD = mad(data)
    median = np.median(data)
    const = z_thresh * MAD / 0.6745  # convert to z-val under Normality assumption
    return (median - const, median + const)

def upper_outlier_mask(data):
    '''upper_outlier_mask ds'''
    _, upbound = calculate_bounds(data)
    return np.array(data) > upbound

def outlier_aware_hist(data, lower=None, upper=None, bins='auto'):
    '''
    code is taken with little modification from https://stackoverflow.com/questions/15837810/making-pyplot-hist-first-and-last-bins-include-outliers
    GMR: added use of f-strings
    '''
    if not lower or lower < data.min():
        lower = data.min()
        lower_outliers = False
        pass
    else: lower_outliers = True
    if not upper or upper > data.max():
        upper = data.max()
        upper_outliers = False
        pass
    else: upper_outliers = True
    _, _, patches = plt.hist(data, range=(lower, upper), bins=bins, density=True)
    if lower_outliers:
        n_lower_outliers = (data < lower).sum()
        patches[0].set_height(patches[0].get_height() + n_lower_outliers)
        patches[0].set_facecolor('c')
        patches[0].set_label(f'Lower outliers: ({data.min():.2f}, {lower:.2f})')
        pass
    if upper_outliers:
        n_upper_outliers = (data > upper).sum()
        patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
        patches[-1].set_facecolor('m')
        patches[-1].set_label(f'Upper outliers: ({upper:.2f}, {data.max():.2f})')
    if lower_outliers or upper_outliers: plt.legend()
    return

def distrplot(title, values, visitor, units=None, fit_t=False, bins='auto'):
    '''distrplot ds'''
    myfig = plt.figure()
    plt.title(title)
    outlier_aware_hist(np.array(values), *calculate_bounds(values), bins)
    plt.ylabel('Density')
    if units is None: plt.xlabel('Estimate')
    else: plt.xlabel(f'Estimate [{units}]')
    if fit_t:
        cauchy_fit = cauchy.fit(values)
        t_fit = t.fit(values)
        norm_fit = norm.fit(values)
        xlim = plt.xlim()
        x = np.arange(xlim[0], xlim[1], (xlim[1]-xlim[0])/1000)
        plt.plot(x, cauchy.pdf(x, loc=cauchy_fit[0], scale=cauchy_fit[1]),
                 label='fit Lorentzian')
        plt.plot(x, t.pdf(x, df=t_fit[0], loc=t_fit[1], scale=t_fit[2]),
                 label=f'fit T ({t_fit[0]:.1f} DF)')
        plt.plot(x, norm.pdf(x, loc=norm_fit[0], scale=norm_fit[1]),
                 label='fit Gaussian')
        visitor.add_primitive(f'{title} t fit: df={t_fit[0]:.5f}, loc={t_fit[1]:.6f}, scale={t_fit[2]:.6f}')
        visitor.add_primitive(f'{title} Gaussian fit: loc={norm_fit[0]:.6f}, scale={norm_fit[1]:.6f}')
        pass
    plt.legend()
    buf = io.BytesIO()
    myfig.savefig(buf, format='png')
    visitor.add_image('...', ' ', buf.getvalue())
    plt.close(myfig)
    return
# -------- -----------------------------------------------------------
