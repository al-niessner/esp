'''ariel core ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import excalibur
import excalibur.system.core as syscore
import excalibur.util.cerberus as crbutil

from excalibur.ariel.metallicity import \
    massMetalRelation, massMetalRelationDisp, randomCtoO
from excalibur.ariel.arielInstrumentModel import load_ariel_instrument
from excalibur.ariel.arielObservingPlan import make_tier_table
from excalibur.ariel.forwardModels import makeTaurexAtmos, makeCerberusAtmos
from excalibur.cerberus.core import myxsecs

import os
import io
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst

# ---------------------------- ---------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    '''
    Tests for empty SV shell
    '''
    if sv['STATUS'][-1]:
        valid = True
        errstring = None
    else:
        valid = False
        errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- SIMULATE ARIEL SPECTRA ------------------------------------------
def simulate_spectra(target, system_dict, out):
    '''
    Simulate Ariel spectra, adding noise based on the Ariel instrument model
    Mulitple spectra are now calculated, allowing a choice within cerberus.atmos fitting
    1) both Taurex and Cerberus atmosphere models
    2) with or without clouds
    3) two models for metallicity/mmw (mmw=2.3 or FINESSE mass-metallicity relation)
    '''

    # ** two key parameters for adjusting the instrument noise model **
    # no longer necessary; Ariel-rad already does these two steps
    # noise_factor = 1./0.7
    # noise_floor_ppm = 50.

    observing_plan = make_tier_table()

    system_params = system_dict['priors']

    # specify which models should be calculated (use these as keys within data)
    atmosModels = ['cerberus', 'cerberusNoclouds',
                   'cerberuslowmmw', 'cerberuslowmmwNoclouds',
                   'taurex', 'taurexNoclouds',
                   'taurexlowmmw', 'taurexlowmmwNoclouds']
    # atmosModels = ['taurexNoclouds']
    # atmosModels = ['taurex', 'taurexNoclouds',
    #               'taurexlowmmw', 'taurexlowmmwNoclouds']
    # atmosModels = ['cerberusNoclouds']
    # atmosModels = ['cerberusNoclouds','cerberus',
    #               'cerberuslowmmw', 'cerberuslowmmwNoclouds']
    # atmosModels = ['cerberusNoclouds', 'taurexNoclouds', 'cerberus']
    out['data']['models'] = atmosModels
    # save target,planet names, for plotting (in states.py)
    out['data']['target'] = target
    out['data']['planets'] = system_params['planets']

    completed_at_least_one_planet = False
    for planetLetter in system_params['planets']:
        # set the random seed as a function of target name
        #  (otherwise all spectra have identical noise realizations!)
        #  (also they would have the same C/O and metallicity offset!)
        intFromTarget = 1
        for char in target+' '+planetLetter:
            intFromTarget = (123 * intFromTarget + ord(char)) % 1000000
        np.random.seed(intFromTarget)

        # load in the wavelength bins and the noise model
        # there is a separate SNR file for each planet
        ariel_instrument = load_ariel_instrument(target+' '+planetLetter)

        if ariel_instrument:

            # asdf : LATER : add in uncertainty scatter to these model parameters
            model_params = {
                'T*':system_params['T*'],
                'R*':system_params['R*'],
                'Rp':system_params[planetLetter]['rp'],
                'Mp':system_params[planetLetter]['mass'],
                'logg':system_params[planetLetter]['logg'],
                'sma':system_params[planetLetter]['sma'],
                'Teq':system_params[planetLetter]['teq']}

            # Calculate the atmosphere scale height
            #  cerberus wants it, to normalize the spectrum
            #  (some of this code is copied from transit/core)
            sscmks = syscore.ssconstants(mks=True)
            #  NOTE : sma is system, but should be model value.  EDIT LATER?
            eqtemp = model_params['T*']*np.sqrt(model_params['R*']*sscmks['Rsun/AU']/
                                                (2.*model_params['sma']))
            pgrid = np.exp(np.arange(np.log(10.)-15., np.log(10.)+15./100, 15./99))
            pressure = pgrid[::-1]
            # Assume solar metallicity here but then below use each model's metallicity
            mixratio, fH2, fHe = crbutil.crbce(pressure, eqtemp)
            # X2Hr=cheq['XtoH'])
            # assume solar C/O and N/O for now
            # C2Or=cheq['CtoO'], N2Or=cheq['NtoO'])
            mmwsolar, fH2, fHe = crbutil.getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
            mmw = mmwsolar * cst.m_p  # [kg]
            Hs = cst.Boltzmann * eqtemp / (mmw*1e-2*(10.**float(model_params['logg'])))  # [m]
            HoverRmax = Hs / (model_params['Rp']*sscmks['Rjup'])
            Hssolar = Hs / (model_params['R*']*sscmks['Rsun'])  # this is used for plot scaling

            # skip non-converging atmospheres!!
            # print()
            # print(target,planetLetter,'  scale height / planet radius (solar metallicity) =',HoverRmax)
            # print()
            # this limit corresponds to the maximum scale height allowing for ~1e11 pressure range
            # (our actual range is typically 10 bar to 1 microbar, requiring 0.06 max H/R)
            if HoverRmax > 0.04:
                # print('SKIP:',target,planetLetter,'  scale height / planet radius =',HoverRmax)
                log.warning('--< SKIP UNBOUND ATMOS: %s %s ; scale height / planet radius = %s >--',target,planetLetter,HoverRmax)
            else:

                # planet metallicity is from an assumed mass-metallicity relation
                #  with scatter included
                # planet metallicity should be defined relative to the stellar metallicity
                metallicity_star = system_params['FEH*']
                M_p = model_params['Mp']
                includeMetallicityDispersion = True
                if includeMetallicityDispersion:
                    metallicity_planet = massMetalRelationDisp(metallicity_star, M_p)
                else:
                    metallicity_planet = massMetalRelation(metallicity_star, M_p)

                # Load the instrument model and rescale based on #-of-transits
                uncertainties = ariel_instrument['noise']
                if target+' '+planetLetter in observing_plan:
                    visits = observing_plan[target+' '+planetLetter]['number of visits']
                    tier = observing_plan[target+' '+planetLetter]['tier']
                else:
                    # default to a single transit observation, if it's not in the Ariel list
                    # print(target+' '+planetLetter,'not found in the Ariel observing plan')
                    errstr = target+' '+planetLetter,'not found in observing plan'
                    log.warning('--< ARIEL SIM_SPECTRUM: %s >--', errstr)
                    visits = '1'
                    tier = '1'
                # print('# of visits:',visits,'  tier',tier,'  ',target+' '+planetLetter)
                uncertainties /= np.sqrt(float(visits))

                created_xslib = False  # only calculate cross-sections once per planet

                # ________LOOP OVER ALL SELECTED MODELS_______
                for atmosModel in atmosModels:
                    # print()
                    # print('starting Atmospheric Model:',atmosModel)

                    if 'lowmmw' in atmosModel:
                        # print(' - using a low mmw')
                        model_params['metallicity'] = 0.  # dex
                        model_params['C/O'] = 0.     # [C/O] (relative to solar)
                    else:
                        model_params['metallicity*'] = metallicity_star
                        model_params['metallicity'] = metallicity_star + metallicity_planet

                        # planet C/O ratio is assumed to be solar
                        #  (0.54951 is the default in ACEChemistry, so it actually has no effect)
                        # actually, let's consider a distribution of C/O, as done for FINESSE
                        model_params['C/O'] = np.log10(randomCtoO()/0.54951)

                    if 'cerberus' in atmosModel:
                        # calculate the unbinned model at high resolution
                        Nhires = 10000
                        Nhires = 1000
                        # 10000 is too slow.  why is that?  what is the normal resolution?
                        # this wavelength range goes from 0.316 to 10 microns
                        # the older ariel SNR files are 0.534+-0.379 to 7.54+-0.244
                        wavelength_um = np.logspace(-0.5,1.0,Nhires)

                        # WAIT HOLD ON - shouldn't cross-sections depend on T,P,etc
                        #  otherwise they are the same for all planets
                        #  oof I guess that's taken into account in the arrays
                        # in that case, don't rerun this for each planetLetter <-- TODO
                        #  (but each letter does have to be in the xslib dict though)
                        #  (or else just use placeholder for planetletter)

                        if not created_xslib:
                            xslib = {'data':{},
                                     'STATUS':[]}
                            tempspc = {'data':{planetLetter:{'WB':wavelength_um}}}
                            # print('CALCulating cross-sections START')
                            _ = myxsecs(tempspc, xslib)
                            # print('CALCulating cross-sections DONE')
                            created_xslib = True
                        else:
                            # print('NOT recalculating xslib for additional cerberus models')
                            pass

                        if 'NoClouds' in atmosModel:
                            cerbModel = makeCerberusAtmos(
                                wavelength_um, model_params, xslib, planetLetter, clouds=False)
                        else:
                            cerbModel = makeCerberusAtmos(
                                wavelength_um, model_params, xslib, planetLetter)
                        fluxDepth = cerbModel

                    elif 'taurex' in atmosModel:
                        if 'NoClouds' in atmosModel:
                            taurexModel = makeTaurexAtmos(model_params, clouds=False)
                        else:
                            taurexModel = makeTaurexAtmos(model_params)

                        wn,fluxDepth = taurexModel[:2]
                        wavelength_um = 1e4 / wn
                    else:
                        sys.exit('ERROR: unknown model')

                    # bads = np.where(np.isnan(fluxDepth))
                    # if len(bads[0]) > 0:
                    # print('   bad fluxDepths:',len(bads[0]),'out of',len(fluxDepth))

                    # REBIN to the Ariel spectral resolution
                    fluxDepth_rebin = []
                    wavelength_um_rebin = []
                    for wavelow,wavehigh in zip(ariel_instrument['wavelow'],
                                                ariel_instrument['wavehigh']):
                        thisWaveBin = np.where((wavelength_um >= wavelow) &
                                               (wavelength_um <= wavehigh))
                        fluxDepth_rebin.append(np.average(fluxDepth[thisWaveBin]))
                        wavelength_um_rebin.append(np.average(wavelength_um[thisWaveBin]))
                    wavelength_um_rebin = np.array(wavelength_um_rebin)
                    fluxDepth_rebin = np.array(fluxDepth_rebin)

                    # ADD OBSERVATIONAL NOISE TO THE TRUE SPECTRUM
                    fluxDepth_observed = fluxDepth_rebin + np.random.normal(scale=uncertainties)

                    # SAVE THE RESULTS
                    if planetLetter not in out['data'].keys():
                        out['data'][planetLetter] = {}

                    # careful - ES and ESerr are supposed to be radii, not transit depth
                    #  need to take a sqrt of them
                    # careful2 - watch out for sqrt of negative numbers
                    signedSqrt = np.sign(fluxDepth_observed) * np.sqrt(np.abs(fluxDepth_observed))
                    out['data'][planetLetter][atmosModel] = {
                        'WB':wavelength_um_rebin,
                        'ES':signedSqrt,
                        'ESerr':0.5 * uncertainties / signedSqrt}
                    # 'ES':np.sqrt(fluxDepth_observed),
                    # 'ESerr':0.5 * uncertainties / np.sqrt(fluxDepth_observed)}

                    # cerberus also wants the scale height, to normalize the spectrum
                    #  keep Hs as it's own param (not inside of system_ or model_param)
                    #  it has to be this way to match the formatting for regular spectra itk

                    # redo the chemsitry/mmw calculation for this metallicity
                    # print('metallicity [X/H]dex:',model_params['metallicity'])
                    mixratio, fH2, fHe = crbutil.crbce(pressure, eqtemp,
                                                       X2Hr=model_params['metallicity'])
                    # assume solar C/O and N/O for now
                    # C2Or=cheq['CtoO'], N2Or=cheq['NtoO'])
                    mmwnow, fH2, fHe = crbutil.getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
                    # print('mmwnow,mmwsolar',mmwnow,mmwsolar)
                    out['data'][planetLetter][atmosModel]['Hs'] = Hssolar * mmwsolar / mmwnow

                    # save the true spectrum (both raw and binned)
                    out['data'][planetLetter][atmosModel]['true_spectrum'] = {
                        'fluxDepth':fluxDepth_rebin,
                        'wavelength_um':wavelength_um_rebin,
                        'fluxDepth_norebin':fluxDepth,
                        'wavelength_norebin':wavelength_um}

                    # also save the Tier level and the number of visits; add these to the plot
                    out['data'][planetLetter][atmosModel]['tier'] = tier
                    out['data'][planetLetter][atmosModel]['visits'] = visits

                    # save the parameters used to create the spectrum. some could be useful
                    # should save both observed value and truth with scatter added in
                    #  'system_params' = the info in system() task
                    #  'model_params' = what is actually used to create forward model
                    out['data'][planetLetter][atmosModel]['system_params'] = {
                        'R*':system_params['R*'],
                        'T*':system_params['T*'],
                        'Rp':system_params[planetLetter]['rp'],
                        'Teq':system_params[planetLetter]['teq'],
                        'Mp':system_params[planetLetter]['mass']}

                    out['data'][planetLetter][atmosModel]['model_params'] = model_params

                    # convert to percentage depth (just for plotting, not for the saved results)
                    fluxDepth = 100 * fluxDepth
                    fluxDepth_rebin = 100 * fluxDepth_rebin
                    fluxDepth_observed = 100 * fluxDepth_observed
                    # careful - uncertainties are reused, so don't change them permanently
                    uncertainties_percent = 100 * uncertainties

                    # PLOT THE SPECTRA
                    myfig, ax = plt.subplots(figsize=(6,4))
                    plt.title('Ariel simulation : '+target+' '+planetLetter+' : Tier-'+tier+' '+visits+' visits',
                              fontsize=16)
                    plt.xlabel(str('Wavelength [$\\mu m$]'), fontsize=14)
                    plt.ylabel(str('$(R_p/R_*)^2$ [%]'), fontsize=14)

                    # plot the true (model) spectrum - raw
                    plt.plot(wavelength_um, fluxDepth,
                             color='palegoldenrod',ls='-',lw=0.1,
                             zorder=1, label='truth raw')
                    # plot the true (model) spectrum - binned
                    plt.plot(wavelength_um_rebin, fluxDepth_rebin,
                             color='k',ls='-',lw=1,
                             zorder=3, label='truth binned')
                    # plot the simulated data points
                    plt.scatter(wavelength_um_rebin, fluxDepth_observed,
                                marker='o',s=20, color='None',edgecolor='k',
                                zorder=4, label='simulated data')
                    plt.errorbar(wavelength_um_rebin, fluxDepth_observed,
                                 yerr=uncertainties_percent,
                                 linestyle='None',lw=0.2, color='grey', zorder=2)

                    plt.xlim(0.,8.)
                    plt.legend()

                    # add a scale-height-normalized flux scale on the right axis
                    Hsscaling = out['data'][planetLetter][atmosModel]['Hs']
                    # print('H scaling for this plot (%):',Hsscaling*100)
                    ax2 = ax.twinx()
                    ax2.set_ylabel('$\\Delta$ [H]')
                    axmin, axmax = ax.get_ylim()
                    rpmed = np.sqrt(np.nanmedian(1.e-2*fluxDepth_rebin))

                    # non-convergent spectra can be all NaN, crashing here on the sqrt
                    # should now be stopped up above in that case, but just in case add conditional
                    if np.isnan(np.nanmax(fluxDepth_rebin)):
                        log.warning('--< PROBLEM: spectrum is all NaN %s %s >--',target,planetLetter)
                        pass
                    else:
                        if axmin >= 0:
                            ax2.set_ylim((np.sqrt(1e-2*axmin) - rpmed)/Hsscaling,
                                         (np.sqrt(1e-2*axmax) - rpmed)/Hsscaling)
                        else:
                            ax2.set_ylim((-np.sqrt(-1e-2*axmin) - rpmed)/Hsscaling,
                                         (np.sqrt(1e-2*axmax) - rpmed)/Hsscaling)
                            # print('TROUBLE!! y-axis not scaled by H!!')

                    myfig.tight_layout()

                    # RID = int(os.environ.get('RUNID', None))
                    RID = os.environ.get('RUNID', None)
                    # print('RID',RID)
                    if RID:
                        RID = f'{int(RID):03}'
                    else:
                        RID = '666'
                    # print('RID',RID)

                    # plotDir = excalibur.context['data_dir'] + f"/ariel/RID{RID:03i}"
                    # plotDir = excalibur.context['data_dir'] + f"/ariel/RID{RID:03}"
                    plotDir = excalibur.context['data_dir'] + '/ariel/RID' + RID
                    if not os.path.exists(plotDir): os.mkdir(plotDir)

                    plt.savefig(plotDir +
                                '/ariel_'+atmosModel+'Atmos_' + target+'_'+planetLetter + '.png')

                    # REDUNDANT SAVE - above saves to disk; below saves as state vector
                    # plt.title('Ariel : '+target+' '+planetLetter+'; sv save for RUNID='+RID,
                    #           fontsize=16)
                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    out['data'][planetLetter][atmosModel]['plot_simspectrum'] = buf.getvalue()

                    plt.close(myfig)

                    completed_at_least_one_planet = True

    # print('completed_at_least_one_planet',completed_at_least_one_planet)
    if completed_at_least_one_planet: out['STATUS'].append(True)

    return True
# ------------------------- ------------------------------------------
