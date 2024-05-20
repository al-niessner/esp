'''cerberus plotting ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging
log = logging.getLogger(__name__)

import io
import corner
import numpy as np
import matplotlib.pyplot as plt

# import excalibur
from excalibur.ariel.metallicity import massMetalRelation

# --------------------------------------------------------------------
def rebinData(transitdata, binsize=4):
    '''
    rebin a spectrum (blue points calculated from grey ones)
    set uncertainty to median/sqrt(N); ignore correlated/systematic errors
    this is ok, since it's just for visualization; not used in the analysis
    '''

    nspec = int(transitdata['wavelength'].size / binsize)
    minspec = np.nanmin(transitdata['wavelength'])
    maxspec = np.nanmax(transitdata['wavelength'])
    scale = (maxspec - minspec) / float(nspec)
    wavebin = scale*np.arange(nspec) + minspec
    deltabin = np.diff(wavebin)[0]
    cbin = wavebin + deltabin/2
    specbin = []
    errbin = []
    wavebin = []   # recalculate the binned wavelength from actual wave grid
    for eachbin in cbin:
        select = transitdata['wavelength'] < (eachbin + deltabin/2)
        select = select & (transitdata['wavelength'] >= (eachbin - deltabin/2))
        select = select & np.isfinite(transitdata['depth'])
        if np.sum(np.isfinite(transitdata['depth'][select])) > 0:
            specbin.append(np.nansum(transitdata['depth'][select]/
                                     (transitdata['error'][select]**2))/
                           np.nansum(1./(transitdata['error'][select]**2)))
            errbin.append(np.nanmedian(transitdata['error'][select])/
                          np.sqrt(np.sum(select)))
            wavebin.append(np.nansum(transitdata['wavelength'][select]/
                                     transitdata['error'][select]**2)/
                           np.nansum(1./transitdata['error'][select]**2))
        # else:
        #    specbin.append(np.nan)
        #    errbin.append(np.nan)
        #    wavebin.append(np.nan)
    # transitdata['binned_wavelength'] = np.array(cbin)
    transitdata['binned_wavelength'] = np.array(wavebin)
    transitdata['binned_depth'] = np.array(specbin)
    transitdata['binned_error'] = np.array(errbin)

    return transitdata
# --------------------------------------------------------------------
# --------------------------------------------------------------------
def plot_bestfit(transitdata, patmos_model, patmos_modelProfiled, fmcarray,
                 truth_spectrum,
                 ancillary_data, atmos_data,
                 filt, modelName, trgt, p, saveDir, savetodisk=False):
    ''' plot the best fit to the data '''

    # figure, ax = plt.subplots(figsize=(8,4))
    figgy = plt.figure(figsize=(20,4))
    # figgy = plt.figure(figsize=(8,4))
    figgy.subplots_adjust(left=0.05,right=0.7,bottom=0.15,top=0.93,wspace=0.8)
    # ax1 = plt.subplot(1,2,1)
    plt.subplot(1,2,1)

    # 1) plot the data
    plt.errorbar(transitdata['wavelength'],
                 transitdata['depth'] * 100,
                 yerr=transitdata['error'] * 100,
                 fmt='.', color='lightgray', zorder=1,
                 label='raw data')
    # 2) also plot the rebinned data points
    plt.errorbar(transitdata['binned_wavelength'],
                 transitdata['binned_depth'] * 100,
                 yerr=transitdata['binned_error'] * 100,
                 # fmt='o', color='k', markeredgecolor='k', markerfacecolor='w', zorder=5,
                 # fmt='^', color='blue', zorder=5,
                 # fmt='o', color='royalblue', zorder=5,
                 fmt='o', markeredgecolor='k', color='None', ecolor='k', zorder=5,
                 label='rebinned data')
    # 3) plot the best-fit model
    plt.plot(transitdata['wavelength'],
             patmos_modelProfiled * 100,
             # c='k', lw=2, zorder=4,
             c='orange', lw=2, zorder=4,
             label='best fit')
    # 4) plot a selection of walkers, to see spread
    xlims = plt.xlim()
    ylims = plt.ylim()
    # print('median pmodel',np.nanmedian(patmos_model))
    for fmcexample in fmcarray:
        patmos_modeli = fmcexample - np.nanmean(fmcexample) + \
            np.nanmean(transitdata['depth'])
        # print('median pmodel',np.nanmedian(patmos_model))
        if np.sum(patmos_modeli)==666:
            plt.plot(transitdata['wavelength'],
                     patmos_modeli * 100,
                     c='grey', lw=0.2, zorder=2)
    plt.ylim(ylims)  # revert to the original y-bounds, in case messed up
    # 5) plot the true spectrum, if it is a simulation
    if truth_spectrum is not None:
        plt.plot(truth_spectrum['wavelength'], truth_spectrum['depth']*100,
                 c='k', lw=1.5, zorder=3,
                 # c='orange', lw=2, zorder=3,
                 label='truth')

    offsets_model = (patmos_model - transitdata['depth']) / transitdata['error']
    offsets_modelProfiled = (patmos_modelProfiled - transitdata['depth']) / transitdata['error']

    # the 'average' function (which allows for weights) doesn't have a NaN version,
    #  so mask out any NaN regions by hand
    okPart = np.where(np.isfinite(transitdata['depth']))
    flatlineFit = np.average(transitdata['depth'][okPart],
                             weights=1/transitdata['error'][okPart]**2)
    # print('flatline average',flatlineFit)
    offsets_flat = (flatlineFit - transitdata['depth']) / transitdata['error']
    # print('median chi2, flat',np.nanmedian(offsets_flat**2))

    chi2model = np.nansum(offsets_model**2)
    chi2modelProfiled = np.nansum(offsets_modelProfiled**2)
    chi2flat = np.nansum(offsets_flat**2)

    numParam_model = 8
    numParam_truth = 0
    numParam_flat = 1

    numPoints = len(patmos_model)
    # print('numpoints',numPoints)
    chi2model_red = chi2model / (numPoints - numParam_model)
    chi2modelProfiled_red = chi2modelProfiled / (numPoints - numParam_model)
    chi2flat_red = chi2flat / (numPoints - numParam_flat)

    # add some labels off to the right side
    xoffset = 0.03
    if truth_spectrum is not None:
        offsets_truth = (truth_spectrum['depth'] - transitdata['depth']) / transitdata['error']
        chi2truth = np.nansum(offsets_truth**2)
        chi2truth_red = chi2truth / (numPoints - numParam_truth)
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.6,
                 '$\\chi^2$-truth='+f"{chi2truth:5.2f}",fontsize=12)
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.36,
                 '$\\chi^2_{red}$-truth='+f"{chi2truth_red:5.2f}",fontsize=12)
    if 'tier' in atmos_data:
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*1.00,
                 'Tier='+atmos_data['tier'],fontsize=12)
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.93,
                 '# of Visits='+atmos_data['visits'],fontsize=12)
    plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.8,
             'ZFOM='+f"{ancillary_data['ZFOM']:4.1f}"+'-'+
             f"{ancillary_data['ZFOM_max']:4.1f}",fontsize=12)
    plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.73,
             'TSM='+f"{ancillary_data['TSM']:5.2f}",fontsize=12)
    plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.53,
             '$\\chi^2$-model='+f"{chi2modelProfiled:5.2f}",fontsize=12)
    plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.46,
             '$\\chi^2$-flat='+f"{chi2flat:5.2f}",fontsize=12)
    plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.29,
             '$\\chi^2_{red}$-model='+f"{chi2modelProfiled_red:5.2f}",fontsize=12)
    plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.22,
             '$\\chi^2_{red}$-flat='+f"{chi2flat_red:5.2f}",fontsize=12)
    plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.12,
             '$\\chi^2_{red}$(model/flat)='+f"{(chi2modelProfiled_red/chi2flat_red):5.2f}",fontsize=12)
    plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.05,
             '$\\Delta\\chi^2$(flat-model)='+f"{(chi2flat-chi2modelProfiled):5.1f}",fontsize=12)

    if filt=='Ariel-sim':
        plt.xlim(0,8)
    plt.title(trgt+' '+p, fontsize=16)
    plt.xlabel(str('Wavelength [$\\mu m$]'), fontsize=14)
    plt.ylabel(str('$(R_p/R_*)^2$ [%]'), fontsize=14)
    plt.legend()
    # *** add the H scaling back in? ***
    # if ('Hs' in spectrum_dict['data'][p]) and ('RSTAR' in spectrum_dict['data'][p]):
    #    rp0hs = np.sqrt(np.nanmedian(transitdata['depth']))
    #    Hs = atm[p]['Hs'][0]
    #    # Retro compatibility for Hs in [m]
    #    if Hs > 1: Hs = Hs/(fin[p]['RSTAR'][0])
    #    ax2 = ax.twinx()
    #    ax2.set_ylabel('$\\Delta$ [Hs]')
    #    axmin, axmax = ax.get_ylim()
    #    ax2.set_ylim((np.sqrt(1e-2*axmin) - rp0hs)/Hs, (np.sqrt(1e-2*axmax) - rp0hs)/Hs)

    #  Plot the pre-profiled result off to the right, if there's been profiling
    if np.any(patmos_model!=patmos_modelProfiled):
        # ax2 = plt.subplot(1,2,2)
        plt.subplot(1,2,2)

        # 1) plot the data
        plt.errorbar(transitdata['wavelength'],
                     transitdata['depth'] * 100,
                     yerr=transitdata['error'] * 100,
                     fmt='.', color='lightgray', zorder=1,
                     label='raw data')
        # 2) also plot the rebinned data points
        plt.errorbar(transitdata['binned_wavelength'],
                     transitdata['binned_depth'] * 100,
                     yerr=transitdata['binned_error'] * 100,
                     fmt='o', markeredgecolor='k', color='None', ecolor='k', zorder=5,
                     label='rebinned data')
        # 3) plot the best-fit model
        plt.plot(transitdata['wavelength'],
                 patmos_model * 100,
                 c='orange', lw=2, zorder=4,
                 label='best fit')
        xlims = plt.xlim()
        ylims = plt.ylim()
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.53,
                 '$\\chi^2$-model='+f"{chi2model:5.2f}",fontsize=12)
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.46,
                 '$\\chi^2$-flat='+f"{chi2flat:5.2f}",fontsize=12)
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.29,
                 '$\\chi^2_{red}$-model='+f"{chi2model_red:5.2f}",fontsize=12)
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.22,
                 '$\\chi^2_{red}$-flat='+f"{chi2flat_red:5.2f}",fontsize=12)
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.12,
                 '$\\chi^2_{red}$(model/flat)='+f"{(chi2model_red/chi2flat_red):5.2f}",fontsize=12)
        plt.text(xlims[1]+xoffset,ylims[0]+(ylims[1]-ylims[0])*0.05,
                 '$\\Delta\\chi^2$(flat-model)='+f"{(chi2flat-chi2model):5.1f}",fontsize=12)
        if filt=='Ariel-sim': plt.xlim(0,8)
        plt.title(trgt+' '+p+' (no profiling)', fontsize=16)
        plt.xlabel(str('Wavelength [$\\mu m$]'), fontsize=14)
        plt.ylabel(str('$(R_p/R_*)^2$ [%]'), fontsize=14)
        plt.legend()

    # figgy.tight_layout()
    # plt.show()
    if savetodisk: plt.savefig(saveDir + 'bestFit_'+filt+'_'+modelName+'_'+trgt+' '+p+'.png')
    # pdf is so much better, but xv gives error (stick with png for debugging)

    # REDUNDANT SAVE - above saves to disk; below saves as state vector
    buf = io.BytesIO()
    figgy.savefig(buf, format='png')
    save_to_state_vector = buf.getvalue()
    plt.close(figgy)
    return save_to_state_vector
# --------------------------------------------------------------------
def plot_corner(allkeys, alltraces, profiletraces,
                truth_params, prior_ranges,
                filt, modelName, trgt, p, saveDir, savetodisk=False):
    ''' corner plot showing posterior distributions '''

    truthcolor = 'darkgreen'
    fitcolor = 'firebrick'

    mcmcMedian = np.nanmedian(np.array(alltraces), axis=1)
    # print(' params inside of corner plotting',allkeys)
    # print('medians inside of corner plotting',mcmcMedian)
    lo = np.nanpercentile(np.array(alltraces), 16, axis=1)
    hi = np.nanpercentile(np.array(alltraces), 84, axis=1)
    # span = hi - lo
    # Careful!  These are not actually the prior ranges; they're the range of walker values
    priorlo = np.nanmin(np.array(alltraces), axis=1)
    priorhi = np.nanmax(np.array(alltraces), axis=1)
    # OK fixed now. prior ranges are saved as output from atmos and then passed in here
    for ikey,key in enumerate(allkeys):
        # print('param:',key)
        # print(' old prior range:',key,priorlo[ikey],priorhi[ikey])

        # special case for older HST state vectors
        prior_ranges['HScale'] = prior_ranges['Hscale']

        if key in prior_ranges.keys():
            priorlo[ikey],priorhi[ikey] = prior_ranges[key]
        # else:
        #    print('TROUBLE: param not found',prior_ranges.keys())
        # print(' new prior range:',key,priorlo[ikey],priorhi[ikey])
    # priorspan = priorhi - priorlo
    # priormid = (priorhi + priorlo) / 2.

    # put a line showing the equilibrium temperature
    # eqtemp = orbp['T*']*np.sqrt(orbp['R*']*ssc['Rsun/AU']/(2.*orbp[p]['sma']))
    # print(lo[4], mcmcMedian[4], hi[4], 'Teq', eqtemp)

    lodist = np.array(mcmcMedian) - np.array(lo)
    hidist = np.array(hi) - np.array(mcmcMedian)
    lorange = np.array(mcmcMedian) - 3 * lodist
    hirange = np.array(mcmcMedian) + 3 * hidist
    trange = [tuple([x,y]) for x, y in zip(lorange, hirange)]
    # previous lines did 3-sigma range.  better to just use the prior bounds as bounds
    trange = [tuple([x,y]) for x, y in zip(priorlo, priorhi)]
    # print('trange',trange)
    # print('lodist',lodist)
    # print('lorange',lorange)
    truths = None
    if truth_params is not None:
        truths = []
        for thiskey in allkeys:
            if thiskey=='T':
                truths.append(truth_params['Teq'])
            elif thiskey=='[X/H]':
                # truths.append(np.log10(truth_params['metallicity']))
                truths.append(truth_params['metallicity'])
            elif thiskey=='[C/O]':
                # truths.append(np.log10(truth_params['C/O'] / 0.54951))
                truths.append(truth_params['C/O'])
            elif thiskey=='[N/O]':
                truths.append(0)
            elif thiskey in truth_params.keys():
                truths.append(truth_params[thiskey])
            else:
                truths.append(666666)
                log.warning('--< PROBLEM: no truth value for this key: %s >--',thiskey)
        # print('truths in corner plot',truths)
    # figure = corner.corner(np.vstack(np.array(alltraces)).T,
    #                       # bins=int(np.sqrt(np.sqrt(nsamples))),
    #                       bins=10,
    #                       labels=allkeys, range=trange,
    #                       truths=truths, truth_color=truthcolor,
    #                       show_titles=True,
    #                       quantiles=[0.16, 0.50, 0.84])
    figure = corner.corner(np.vstack(np.array(profiletraces)).T,
                           bins=10,
                           labels=allkeys, range=trange,
                           truths=truths, truth_color=truthcolor,
                           show_titles=True,
                           quantiles=[0.16, 0.50, 0.84])
    # smaller size for corner plot might fit better, but this creates a bit of checkerboarding
    # figure.set_size_inches(16,16)  # this actually makes it smaller

    ndim = len(alltraces)
    axes = np.array(figure.axes).reshape((ndim, ndim))
    # use larger font size for the axis labels
    for i in range(ndim):
        ax = axes[ndim-1, i]
        ax.set_xlabel(allkeys[i], fontsize=14)
    for i in range(ndim-1):
        # skipping the first one on the y side (it's a histo, not a 2-D plot)
        ax = axes[i+1, 0]
        ax.set_ylabel(allkeys[i+1], fontsize=14)
    #  draw a point and crosshair for the medians in each subpanel
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(mcmcMedian[xi], color=fitcolor)
            ax.axhline(mcmcMedian[yi], color=fitcolor)
            ax.plot(mcmcMedian[xi], mcmcMedian[yi], marker='s', c=fitcolor)
    for i in range(ndim):
        ax = axes[i, i]
        # draw light-colored vertical lines in each hisogram for the prior
        #  drop this. adds clutter and is redundant with the vsPrior plot following
        # ax.axvline(priorlo[i] + 0.5*priorspan[i], color='grey', zorder=1)
        # ax.axvline(priorlo[i] + 0.16*priorspan[i], color='grey', zorder=1, ls='--')
        # ax.axvline(priorlo[i] + 0.84*priorspan[i], color='grey', zorder=1, ls='--')

        # darken the lines for the fit results
        # ax.axvline(mcmcMedian[i], color='k', lw=2, zorder=2)
        # ax.axvline(lo[i], color='k', lw=2, zorder=2, ls='--')
        # ax.axvline(hi[i], color='k', lw=2, zorder=2, ls='--')
        # actually the fit result lines are o.k. as is, except all three are dashed
        #  make the median fit a solid line
        #  and maybe change the color to match the central panels
        #  hmm, it's not covering up the dashed line; increase lw and maybe zorder
        ax.axvline(mcmcMedian[i], color=fitcolor, lw=2, zorder=12)

    if savetodisk: plt.savefig(saveDir + 'corner_'+filt+'_'+modelName+'_'+trgt+' '+p+'.png')
    # REDUNDANT SAVE - above saves to disk; below saves as state vector
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    save_to_state_vector = buf.getvalue()
    plt.close(figure)
    return save_to_state_vector
# --------------------------------------------------------------------
def plot_vsPrior(allkeys, alltraces, profiledtraces,
                 truth_params, prior_ranges, appliedLimits,
                 filt, modelName, trgt, p, saveDir, savetodisk=False):
    ''' compare the fit results against the original prior information '''

    mcmcMedian = np.nanmedian(np.array(alltraces), axis=1)
    lo = np.nanpercentile(np.array(alltraces), 16, axis=1)
    hi = np.nanpercentile(np.array(alltraces), 84, axis=1)
    span = hi - lo
    priorlo = np.nanmin(np.array(alltraces), axis=1)
    priorhi = np.nanmax(np.array(alltraces), axis=1)

    mcmcMedianProfiled = np.nanmedian(np.array(profiledtraces), axis=1)
    loProfiled = np.nanpercentile(np.array(profiledtraces), 16, axis=1)
    hiProfiled = np.nanpercentile(np.array(profiledtraces), 84, axis=1)
    spanProfiled = hiProfiled - loProfiled
    priorloProfiled = np.nanmin(np.array(profiledtraces), axis=1)
    priorhiProfiled = np.nanmax(np.array(profiledtraces), axis=1)

    for ikey,key in enumerate(allkeys):
        # print('param:',key)
        # print(' old prior range:',priorlo[ikey],priorhi[ikey])
        if key in prior_ranges.keys():
            priorlo[ikey],priorhi[ikey] = prior_ranges[key]
            # *** this needs work; use appliedLimits somehow ***
            priorloProfiled[ikey],priorhiProfiled[ikey] = prior_ranges[key]
        # print(' new prior range:',priorlo[ikey],priorhi[ikey])
    priorspan = priorhi - priorlo
    priormid = (priorhi + priorlo) / 2.
    priorspanProfiled = priorhiProfiled - priorloProfiled
    priormidProfiled = (priorhiProfiled + priorloProfiled) / 2.

    figure = plt.figure(figsize=(12,6))
    Nparam = len(mcmcMedian)
    for iparam in range(Nparam):
        ax = figure.add_subplot(2,int((len(mcmcMedian)+1.)/2.),iparam+1)
        ax.scatter(priormid[iparam],priorspan[iparam]*0.34,
                   facecolor='None',edgecolor='black', s=30, zorder=3)
        ax.scatter(priormidProfiled[iparam],priorspanProfiled[iparam]*0.34,
                   facecolor='black',edgecolor='black', s=30, zorder=3)
        ax.scatter(mcmcMedian[iparam],span[iparam]/2.,
                   facecolor='None',edgecolor='firebrick', s=50, zorder=4)
        ax.scatter(mcmcMedianProfiled[iparam],spanProfiled[iparam]/2.,
                   facecolor='purple',edgecolor='firebrick', s=50, zorder=4)
        ax.plot([priorlo[iparam],priormid[iparam],priorhi[iparam]],
                [0,priorspan[iparam]*0.34,0],
                c='k',ls='--',lw=0.5,zorder=2)
        # ax.plot([priorloProfiled[iparam],priormidProfiled[iparam],priorhiProfiled[iparam]],
        #        [0,priorspanProfiled[iparam]*0.34,0],
        #        c='purple',ls=':',lw=1.5,zorder=2)

        # highlight the background of successful fits (x2 better precision)
        if span[iparam]/2. < priorspan[iparam]*0.34 / 2:
            # print(' this one has reduced uncertainty',allkeys[iparam])
            ax.set_facecolor('lightyellow')

        # add a dashed line for the true value
        if truth_params is not None:
            keyMatch = {'T':'Teq',
                        '[X/H]':'metallicity',
                        '[C/O]':'C/O'}
            if allkeys[iparam] in keyMatch:
                truthparam = keyMatch[allkeys[iparam]]
            else:
                truthparam = allkeys[iparam]
            if truthparam in truth_params:
                truthvalue = float(truth_params[truthparam])
                ax.plot([truthvalue,truthvalue],[0,priorspan[iparam]],
                        c='k',ls='--',zorder=5)
            elif truthparam=='[N/O]':
                ax.plot([0,0],[0,priorspan[iparam]],
                        c='k',ls='--',zorder=5)

        # show if there is some profiling for this parameter
        profiledParams = [limit[0] for limit in appliedLimits]
        profiledValues = [limit[1] for limit in appliedLimits]
        if allkeys[iparam] in profiledParams:
            # print('profiling limit!',allkeys[iparam])
            iprof = profiledParams.index(allkeys[iparam])
            # print('  profiled value',profiledValues[iprof],iprof)
            ax.plot([profiledValues[iprof],profiledValues[iprof]],
                    [0,priorspan[iparam]],
                    c='purple',ls=':',lw=2,zorder=4)

        ax.set_xlim(priorlo[iparam],priorhi[iparam])
        ax.set_ylim(0,priorspan[iparam]*0.4)
        ax.set_xlabel(allkeys[iparam], fontsize=14)
        ax.set_ylabel('uncertainty', fontsize=14)
    figure.tight_layout()
    if savetodisk: plt.savefig(saveDir + 'vsprior_'+filt+'_'+modelName+'_'+trgt+' '+p+'.png')
    # REDUNDANT SAVE - above saves to disk; below saves as state vector
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    save_to_state_vector = buf.getvalue()
    plt.close(figure)
    return save_to_state_vector

# --------------------------------------------------------------------
def plot_walkerEvolution(allkeys, alltraces, profiledtraces,
                         truth_params, prior_ranges, appliedLimits,
                         filt, modelName, trgt, p, saveDir, savetodisk=False,
                         Nchains=4):
    ''' trace whether or not the MCMC walkers converge '''

    mcmcMedian = np.nanmedian(np.array(alltraces), axis=1)
    # mcmcMedianProfiled = np.nanmedian(np.array(profiledtraces), axis=1)
    Nparam = len(mcmcMedian)
    priorlo = np.nanmin(np.array(alltraces), axis=1)
    priorhi = np.nanmax(np.array(alltraces), axis=1)
    priorloProfiled = np.nanmin(np.array(profiledtraces), axis=1)
    priorhiProfiled = np.nanmax(np.array(profiledtraces), axis=1)
    for ikey,key in enumerate(allkeys):
        # print('param:',key)
        # print(' old prior range:',priorlo[ikey],priorhi[ikey])
        if key in prior_ranges.keys():
            priorlo[ikey],priorhi[ikey] = prior_ranges[key]
            priorloProfiled[ikey],priorhiProfiled[ikey] = prior_ranges[key]
        # print(' new prior range:',priorlo[ikey],priorhi[ikey]

    figure = plt.figure(figsize=(12,6))
    linecolors = ['crimson','seagreen','royalblue','darkorange',
                  'coral','deepskyblue','gold','blueviolet']
    chainLength = int(len(alltraces[0]) / Nchains)
    # chainLengthProfiled = int(len(profiledtraces[0]) / Nchains)
    for iparam in range(Nparam):
        ax = figure.add_subplot(2,int((Nparam+1.)/2.),iparam+1)
        for ic in range(Nchains):
            jump = ic * chainLength
            # jumpProfiled = ic * chainLengthProfiled
            # ax.plot(np.arange(chainLengthProfiled)+1,
            #        profiledtraces[iparam][jumpProfiled:jumpProfiled+chainLengthProfiled],
            ax.plot(np.arange(chainLength)+1,
                    alltraces[iparam][jump:jump+chainLength],
                    c=linecolors[ic % len(linecolors)],
                    alpha=1. - (ic/float(Nchains)/2.),
                    ls='-',lw=0.5,zorder=3)
        # add a dashed line for the true value
        if truth_params is not None:
            keyMatch = {'T':'Teq',
                        '[X/H]':'metallicity',
                        '[C/O]':'C/O'}
            if allkeys[iparam] in keyMatch:
                truthparam = keyMatch[allkeys[iparam]]
            else:
                truthparam = allkeys[iparam]
            if truthparam in truth_params:
                truthvalue = float(truth_params[truthparam])
                ax.plot([0,2*chainLength], [truthvalue,truthvalue],
                        c='k',ls='--',zorder=5)
            elif truthparam=='[N/O]':
                ax.plot([0,2*chainLength], [0,0],
                        c='k',ls='--',zorder=5)

        # show if there is some profiling for this parameter
        profiledParams = [limit[0] for limit in appliedLimits]
        profiledValues = [limit[1] for limit in appliedLimits]
        profiledSigns = [limit[2] for limit in appliedLimits]
        # print('appliedLimits',appliedLimits)
        if allkeys[iparam] in profiledParams:
            # print('profiling limit!',allkeys[iparam])
            iprof = profiledParams.index(allkeys[iparam])
            # print('  profiled value',profiledValues[iprof],iprof)
            ax.plot([0,2*chainLength], [profiledValues[iprof],profiledValues[iprof]],
                    c='purple',ls=':',lw=2,zorder=4)
            # add grey shading to the area that's dropped by profiling
            greyscale = 0.6
            if profiledSigns[iprof]=='>':
                ax.axhspan(-1.e5, profiledValues[iprof], alpha=greyscale, color='k', zorder=7)
            elif profiledSigns[iprof]=='<':
                ax.axhspan(1.e5, profiledValues[iprof], alpha=greyscale, color='k', zorder=7)
            # else:
            #    print(profiledSigns)
            #    exit('ERROR: limit has to be < or >')

        ax.set_xlim(0,chainLength+1)
        # ax.set_ylim(priorlo[iparam],priorhi[iparam])
        ax.set_ylim(priorloProfiled[iparam],priorhiProfiled[iparam])
        ax.set_xlabel('MCMC step #', fontsize=14)
        ax.set_ylabel(allkeys[iparam], fontsize=14)
    figure.tight_layout()
    if savetodisk: plt.savefig(saveDir + 'walkerevol_'+filt+'_'+modelName+'_'+trgt+' '+p+'.png')
    # REDUNDANT SAVE - above saves to disk; below saves as state vector
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    # out['data'][p]['plot_walkerevol'] = buf.getvalue()
    plot_as_state_vector = buf.getvalue()
    plt.close(figure)
    return plot_as_state_vector

# --------------------------------------------------------------------
def plot_fitsVStruths(truth_values, fit_values, fit_errors, prior_ranges,
                      filt, saveDir, savetodisk=False):
    '''
    Compare the retrieved values against the original inputs
    Also (optionally) show a histogram of the uncertainty values
    '''

    plot_statevectors = []
    for param in ['T', '[X/H]', '[C/O]', '[N/O]']:

        # figure = plt.figure(figsize=(5,5))
        # ax = figure.add_subplot(1,1,1)
        figure = plt.figure(figsize=(10,5))
        ax = figure.add_subplot(1,2,1)

        for truth,fit,error in zip(truth_values[param],
                                   fit_values[param],
                                   fit_errors[param]):
            # check whether there is any real information beyond the prior
            # let's say you have to improve uncertainty by a factor of 2
            # but note that the original 1-sigma uncertainty is ~2/3 of prior range
            # oh wait also note that errorbar is one-sided, so another factor of 2
            minInfo = 0.5 * 0.68 * 0.5
            newInfo = False
            if param not in prior_ranges:
                log.warning('--< Cerb.analysis plotting: Parameter missing from prior_range 1: %s >--',param)
                if param=='[N/O]':
                    priorRangeDiff = 12
                    if error < minInfo * priorRangeDiff:
                        newInfo = True
            elif param=='T':
                priorRangeFactor = prior_ranges[param][1] / prior_ranges[param][0]
                # prior is normally set to 0.75-1.5 times Teq
                if error < minInfo * (priorRangeFactor-1) * 0.75*truth:
                    newInfo = True
            else:
                priorRangeDiff = prior_ranges[param][1] - prior_ranges[param][0]
                if error < minInfo * priorRangeDiff:
                    newInfo = True
#                if param=='[X/H]':
#                    print('met,fit,err',truth,fit,error,
#                          prior_ranges[param][1] - prior_ranges[param][0],newInfo)
            if newInfo:
                clr = 'k'
                lwid = 1
                zord = 4
                ptsiz = 40
            else:
                clr = 'grey'
                lwid = 0.5
                zord = 2
                ptsiz = 20
            ax.scatter(truth, fit,
                       facecolor=clr,edgecolor=clr, s=ptsiz, zorder=zord+1)
            ax.errorbar(truth, fit, yerr=error,
                        fmt='.', color=clr, lw=lwid, zorder=zord)

            # if param=='T' and truth > 3333:
            #    print('strangely high T in plot',truth)
            # if param=='[X/H]' and truth > 66:
            #    print('strangely high [X/H] in plot',truth)
            # if param=='[C/O]' and truth > 0.5:
            #    print('strangely high [C/O] in plot',truth)

        # previous was plotting them all at once; now it's one point at a time
        # ax.scatter(truth_values[param],
        #           fit_values[param],
        #           facecolor=clr,edgecolor=clr, s=40, zorder=zord+1)
        # ax.errorbar(truth_values[param],
        #            fit_values[param],
        #            yerr=fit_errors[param],
        #            fmt='.', color=clr, lw=lwid, zorder=zord)

        ax.set_xlabel(param+' truth', fontsize=14)
        ax.set_ylabel(param+' fit', fontsize=14)

        xrange = ax.get_xlim()
        # overallmin = min(ax.get_xlim()[0],ax.get_ylim()[0])
        overallmax = max(ax.get_xlim()[1],ax.get_ylim()[1])

        # plot equality as a dashed diagonal line
        ax.plot([-10,10000],[-10,10000],'k--', lw=1, zorder=1)
        if param=='T':  # show T prior (from 0.75 to 1.5 times Teq)
            ax.plot([-10,10000],[-10*0.75,10000*0.75],'k:', lw=1, zorder=1)
            ax.plot([-10,10000],[-10*1.5,10000*1.5],'k:', lw=1, zorder=1)

        # plot C/O=1 as a dotted vertical line
        if param=='[C/O]':
            solarCO = np.log10(0.55)
            ax.plot([-solarCO,-solarCO],[-100,100],'k--', lw=1, zorder=1)

        # ax.set_xlim(overallmin,overallmax)
        # ax.set_ylim(overallmin,overallmax)
        if param=='T':  # the prior for T varies between targets
            ax.set_xlim(0,overallmax)
            ax.set_ylim(0,overallmax)
        elif param not in prior_ranges:
            log.warning('--< Cerb.analysis plotting: Parameter missing from prior_range 2: %s >--',param)
            ax.set_xlim(xrange)
            if param=='[N/O]':
                ax.set_ylim(-6,6)
        else:
            # actually, don't use prior range for X/H and X/O on x-axis
            # ax.set_xlim(prior_ranges[param][0],prior_ranges[param][1])
            ax.set_xlim(xrange)
            ax.set_ylim(prior_ranges[param][0],prior_ranges[param][1])

        # UNCERTAINTY HISTOGRAMS IN SECOND PANEL
        ax = figure.add_subplot(1,2,2)
        if param=='T':
            errors = np.log10(fit_errors[param]/np.array(fit_values[param]))
            ax.set_xlabel('log('+param+' fractional uncertainty)', fontsize=14)
        else:
            errors = np.log10(fit_errors[param])
            ax.set_xlabel('log('+param+' uncertainty)', fontsize=14)
        lower = errors.min()
        upper = errors.max()
        # print('uncertainty range (logged)',param,lower,upper)
        plt.hist(errors, range=(lower, upper), bins=10,
                 cumulative=True, density=True,
                 color='olive', zorder=1, label='')
        ax.set_ylabel(param+' # of planets', fontsize=14)

        figure.tight_layout()

        # ('display' doesn't work for pdf files)
        if savetodisk: plt.savefig(saveDir + 'fitVStruth_'+filt+'_'+param.replace('/',':')+'.png')
        # REDUNDANT SAVE - above saves to disk; below saves as state vector
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        # out['data'][p]['plot_walkerevol'] = buf.getvalue()
        plot_statevectors.append(buf.getvalue())
        plt.close(figure)
    return plot_statevectors
# --------------------------------------------------------------------
def plot_fitUncertainties(fit_values, fit_errors, prior_ranges,
                          filt, saveDir, savetodisk=False):
    '''
    Plot uncertainty as a function of the fit value
    And show a histogram of the uncertainty values
    '''

    plot_statevectors = []
    for param in ['T', '[X/H]', '[C/O]', '[N/O]']:

        figure = plt.figure(figsize=(10,5))
        ax = figure.add_subplot(1,2,1)

        for fitvalue,error in zip(fit_values[param],fit_errors[param]):

            # check whether there is any real information beyond the prior
            # let's say you have to improve uncertainty by a factor of 2
            # but note that the original 1-sigma uncertainty is ~2/3 of prior range
            # oh wait also note that errorbar is one-sided, so another factor of 2
            minInfo = 0.5 * 0.68 * 0.5
            newInfo = False
            if param not in prior_ranges:
                log.warning('--< Cerb.analysis plotting: Parameter missing from prior_range 3: %s >--',param)
                if param=='[N/O]':
                    priorRangeDiff = 12
                    if error < minInfo * priorRangeDiff:
                        newInfo = True
            elif param=='T':
                priorRangeFactor = prior_ranges[param][1] / prior_ranges[param][0]
                # prior is normally set to 0.75-1.5 times Teq
                # if error < minInfo * (priorRangeFactor-1) * 0.75*truth:
                # asdf: this needs work maybe.  should pass in Teq as truth?
                if error < minInfo * (priorRangeFactor-1) * 0.75*fitvalue:
                    newInfo = True
            else:
                priorRangeDiff = prior_ranges[param][1] - prior_ranges[param][0]
                if error < minInfo * priorRangeDiff:
                    newInfo = True
            if newInfo:
                clr = 'k'
                # lwid = 1
                zord = 4
                ptsiz = 40
            else:
                clr = 'grey'
                # lwid = 0.5
                zord = 2
                ptsiz = 20
            ax.scatter(fitvalue, error,
                       facecolor=clr,edgecolor=clr, s=ptsiz, zorder=zord+1)

        ax.set_xlabel(param+' fit value', fontsize=14)
        ax.set_ylabel(param+' fit uncertainty', fontsize=14)

        # plot C/O=1 as a dotted vertical line
        if param=='[C/O]':
            solarCO = np.log10(0.55)
            ax.plot([-solarCO,-solarCO],[-100,100],'k--', lw=1, zorder=1)

        # xrange = ax.get_xlim()
        # ax.set_xlim(xrange)
        # ax.set_ylim(prior_ranges[param][0],prior_ranges[param][1])

        # UNCERTAINTY HISTOGRAMS IN SECOND PANEL
        ax = figure.add_subplot(1,2,2)
        if param=='T':
            errors = np.log10(fit_errors[param]/np.array(fit_values[param]))
            ax.set_xlabel('log('+param+' fractional uncertainty)', fontsize=14)
        else:
            errors = np.log10(fit_errors[param])
            ax.set_xlabel('log('+param+' uncertainty)', fontsize=14)
        lower = errors.min()
        upper = errors.max()
        # print('uncertainty range (logged)',param,lower,upper)
        plt.hist(errors, range=(lower, upper), bins=10,
                 cumulative=True, density=True,
                 color='olive', zorder=1, label='')
        ax.set_ylabel(param+' # of planets', fontsize=14)

        figure.tight_layout()

        # ('display' doesn't work for pdf files)
        if savetodisk: plt.savefig(saveDir + 'fitUncertainties_'+filt+'_'+param.replace('/',':')+'.png')
        # REDUNDANT SAVE - above saves to disk; below saves as state vector
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        # out['data'][p]['plot_walkerevol'] = buf.getvalue()
        plot_statevectors.append(buf.getvalue())
        plt.close(figure)
    return plot_statevectors
# --------------------------------------------------------------------
def plot_massVSmetals(masses, truth_values,
                      fit_values, fit_errors, prior_ranges,
                      filt, saveDir, savetodisk=False):
    ''' how well do we retrieve the input mass-metallicity relation? '''

    figure = plt.figure(figsize=(5,5))
    ax = figure.add_subplot(1,1,1)

    # Note: masses_true is not actually used, just masses. (they should be the same thing)
    masses_true = truth_values['Mp']
    metals_true = truth_values['[X/H]']
    metals_fit = fit_values['[X/H]']
    metals_fiterr = fit_errors['[X/H]']

    ilab1 = False
    ilab2 = False
    for mass,masstrue,metaltrue,metalfit,metalerror in zip(
            masses,masses_true,metals_true,metals_fit,metals_fiterr):
        # check whether there is any real information beyond the prior
        # let's say you have to improve uncertainty by a factor of 2
        # but note that the original 1-sigma uncertainty is ~2/3 of prior range
        # oh wait also note that errorbar is one-sided, so another factor of 2
        minInfo = 0.5 * 0.68 * 0.5
        priorRangeDiff = prior_ranges['[X/H]'][1] - prior_ranges['[X/H]'][0]
        if metalerror < minInfo * priorRangeDiff:
            clr = 'k'
            lwid = 1
            ptsiz = 40
            zord = 5
        else:
            clr = 'grey'
            lwid = 0.5
            ptsiz = 20
            zord = 2
        if metaltrue not in (666, 666666):
            if not ilab1:
                ilab1 = True
                ax.scatter(masstrue, metaltrue,
                           label='true value',
                           facecolor='w',edgecolor=clr, s=ptsiz, zorder=zord+1)
            else:
                ax.scatter(masstrue, metaltrue,
                           facecolor='w',edgecolor=clr, s=ptsiz, zorder=zord+1)
        if not ilab2:
            ilab2 = True
            ax.scatter(mass, metalfit,
                       label='fit value',
                       facecolor=clr,edgecolor=clr, s=ptsiz, zorder=zord+2)
        else:
            ax.scatter(mass, metalfit,
                       facecolor=clr,edgecolor=clr, s=ptsiz, zorder=zord+2)
        ax.errorbar(mass, metalfit, yerr=metalerror,
                    fmt='.', color=clr, lw=lwid, zorder=zord)
    # ax.scatter(masses, metals_true,
    #           facecolor='w',edgecolor='grey', s=40, zorder=3)
    # ax.scatter(masses, metals_fit,
    #           facecolor='k',edgecolor='k', s=40, zorder=4)
    # ax.errorbar(masses, metals_fit, yerr=metals_fiterr,
    #            fmt='.', color='k', zorder=2)
    ax.semilogx()
    ax.set_xlabel('$M_p (M_{\\rm Jup})$', fontsize=14)
    ax.set_ylabel('[X/H]$_p$', fontsize=14)
    xrange = ax.get_xlim()
    yrange = ax.get_ylim()

    # plot the underlying distribution (only is this is a simulation)
    if 'sim' in filt:
        masses = np.logspace(-5,3,100)
        metals = massMetalRelation(0, masses,thorngren=True)
        ax.plot(masses,metals, 'k--', lw=1, zorder=1, label='true relationship')

    plt.legend()

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    figure.tight_layout()

    # *** TROUBLE - need to subtract off the stellar metallicity? ***

    # ('display' doesn't work for pdf files)
    if savetodisk: plt.savefig(saveDir + 'massVSmetals_'+filt+'.png')
    # REDUNDANT SAVE - above saves to disk; below saves as state vector
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    plot_massMetals = buf.getvalue()
    plt.close(figure)

    return plot_massMetals
