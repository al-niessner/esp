'''ariel core ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import excalibur
import excalibur.system.core as syscore
import excalibur.util.cerberus as crbutil

from excalibur.ariel.metallicity import \
    massMetalRelation, massMetalRelationDisp, randomCtoO
from excalibur.ariel.arielInstrumentModel import load_ariel_instrument
from excalibur.ariel.arielObservingPlan import make_numberofTransits_table

import os
import io
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cst

# ACEChemistry is no longer part of taurex
#  has to be pip-installed
# the pip-install is done via the git-repository installation requirements:
#  1) edit the esp/.ci/Dockerfile.base file to include taurex_ace
#  2) run esp/.ci/step_00.sh to do the installation
#  1b) go back and add gfortran to Dockerfile.base, if it's missing there
#  3) run esp/.ci/deploy_04.sh to add it to devel
#  4) restart the pipeline with the newly created devel image
ACEimported = True
# try:
#     import taurex_ace
from taurex_ace import ACEChemistry
# except:
#    ACEimported = False

import taurex
import taurex.log
taurex.log.disableLogging()
from taurex.cache import OpacityCache,CIACache
from taurex.chemistry import TaurexChemistry
# from taurex.chemistry import ACEChemistry
from taurex.chemistry import ConstantGas
from taurex.contributions import AbsorptionContribution
from taurex.contributions import CIAContribution
from taurex.contributions import RayleighContribution
from taurex.contributions import SimpleCloudsContribution
# from taurex.contributions import FlatMieContribution
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.temperature import Guillot2010
# TAUREX_DATA_ROOT = os.environ.get ('TAUREX_DATA_ROOT', '/proj/data/taurex')
# TAUREX_DATA_ROOT_LOCAL = os.environ.get ('TAUREX_DATA_ROOT', '/proj/sdp/data/taurex')
TAUREX_DATA_ROOT = os.environ.get ('TAUREX_DATA_ROOT', excalibur.context['data_dir']+'/taurex')
# try:
OpacityCache().clear_cache()
OpacityCache().set_opacity_path(os.path.join (TAUREX_DATA_ROOT,
                                              'xsec/xsec_sampled_R15000_0.3-50'))
CIACache().set_cia_path(os.path.join (TAUREX_DATA_ROOT, 'cia/HITRAN'))
# except NotADirectoryError:
#    OpacityCache().clear_cache()
#    OpacityCache().set_opacity_path(os.path.join (TAUREX_DATA_ROOT_LOCAL,
#                                                  'xsec/xsec_sampled_R15000_0.3-50'))
#    CIACache().set_cia_path(os.path.join (TAUREX_DATA_ROOT_LOCAL, 'cia/HITRAN'))


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
# -- SIMULATE ARIEL SPECTRUM -- --------------------------------------
def simulate_spectrum(target, system_dict, out):
    '''
    Simulate an Ariel spectrum based on a scaled FINESSE noise model
    '''

    # ** two key parameters for adjusting the instrument noise model **
    noise_factor = 1./0.7
    noise_floor_ppm = 50.

    observing_plan = make_numberofTransits_table()

    system_params = system_dict['priors']

    star = BlackbodyStar(temperature=system_params['T*'],
                         radius=system_params['R*'])

    completed_at_least_one_planet = False
    atmosModels = []
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
                'R*':system_params['R*'],
                'T*':system_params['T*'],
                'Rp':system_params[planetLetter]['rp'],
                'Teq':system_params[planetLetter]['teq'],
                'Mp':system_params[planetLetter]['mass']}

            # define the planet based on mass & radius
            planet = Planet(planet_radius=model_params['Rp'],
                            planet_mass=model_params['Mp'])

            # set the planet's vertical temperature profile
            temperature_profile = Guillot2010(
                T_irr=float(model_params['Teq']))

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
            # it seems that Taurex wants a linear value for metallicity, not the usual dex
            model_params['metallicity*'] = 10.**(metallicity_star)
            model_params['metallicity'] = 10.**(metallicity_star + metallicity_planet)

            # planet C/O ratio is assumed to be solar
            #  (this number is the default in ACEChemistry, so it actually has no effect)
            model_params['C/O'] = 0.54951
            # actually, let's consider a distribution of C/O, as done for FINESSE
            model_params['C/O'] = randomCtoO()

            # this is the old version copied from taurex task
            chemistry = TaurexChemistry(fill_gases=['H2','He'],ratio=0.172)
            water = ConstantGas('H2O', mix_ratio=1.2e-4*model_params['metallicity'])
            chemistry.addGas(water)

            # new version with equilibrium chemistry is preferred
            if ACEimported:
                chemistry = ACEChemistry(metallicity=model_params['metallicity'],
                                         co_ratio=model_params['C/O'])

            # create an atmospheric model based on the above parameters
            tm = TransmissionModel(star=star,
                                   planet=planet,
                                   temperature_profile=temperature_profile,
                                   chemistry=chemistry,
                                   atm_min_pressure=1e0,
                                   atm_max_pressure=1e6,
                                   nlayers=30)
            # add some more physics
            tm.add_contribution(AbsorptionContribution())
            tm.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
            tm.add_contribution(RayleighContribution())

            # include Clouds!
            #  (don't forget to save the cloud parameters down below)
            P_mbar = 1.
            P_pascals = P_mbar * 100
            if P_pascals==666:
                tm.add_contribution(SimpleCloudsContribution(clouds_pressure=P_pascals))
            # model_params['clouds'] = '1mbar'
            model_params['clouds'] = 'none'

            # tau = 0.3
            # Pmin_mbar = 0.1
            # Pmax_mbar = 10.
            # Pmin_pascals = Pmin_mbar * 100
            # Pmax_pascals = Pmin_mbar * 100
            # tm.add_contribution(FlatMieContribution(
            #    flat_mix_ratio=tau,
            #    flat_bottomP=Pmax_pascals,
            #    flat_topP=Pmin_pascals))

            tm.build()
            atmosModel = tm.model()
            atmosModels.append(atmosModel)

            wn,fluxDepth = atmosModel[:2]
            wavelength_um = 1e4 / wn

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

            # RESCALE THE SNR FILE based on #-of-transits, multiplicative factor, and noise floor
            # 1) scale the noise with a multiplicative factor
            uncertainties = (noise_factor * ariel_instrument['noise'])

            # 2) read in the number of observing epochs.  reduce noise by sqrt(N)
            if target+' '+planetLetter in observing_plan:
                numberofTransits = observing_plan[target+' '+planetLetter]
            else:
                # default to a single transit observation, if it's not in the Edwards Ariel list
                numberofTransits = 1
            uncertainties /= np.sqrt(float(numberofTransits))

            # 3) apply a noise floor (e.g. 50 ppm)
            uncertainties[np.where(uncertainties < noise_floor_ppm/1.e6)] = noise_floor_ppm/1.e6

            # ADD OBSERVATIONAL NOISE TO THE TRUE SPECTRUM
            fluxDepth_observed = fluxDepth_rebin + np.random.normal(scale=uncertainties)

            # SAVE THE RESULTS; MATCH TO WHATEVER FORMAT CERBERUS WANTS

            # put the ariel spectrum into standard 'data' format, same as any other spectrum
            #  this is redundant with above, but it avoids special cases within cerberus
            # oh wait a second.  ES and ESerr are actually supposed to be radii, not transit depth
            #  need to take a sqrt of them
            out['data'][planetLetter] = {
                'WB':wavelength_um_rebin,
                'ES':np.sqrt(fluxDepth_observed),
                'ESerr':0.5 * uncertainties / np.sqrt(fluxDepth_observed)}

            # cerberus also wants the scale height, to normalize the spectrum
            #  (copy over some code for this from transit/core)
            sscmks = syscore.ssconstants(mks=True)
            #  NOTE : sma is system, but should be model value.  EDIT LATER?
            eqtemp = model_params['T*']*np.sqrt(model_params['R*']*sscmks['Rsun/AU']/
                                                 (2.*system_params[planetLetter]['sma']))
            pgrid = np.arange(np.log(10.)-15., np.log(10.)+15./100, 15./99)
            pgrid = np.exp(pgrid)
            pressure = pgrid[::-1]
            mixratio, fH2, fHe = crbutil.crbce(pressure, eqtemp)
            mmw, fH2, fHe = crbutil.getmmw(mixratio, protosolar=False, fH2=fH2, fHe=fHe)
            mmw = mmw * cst.m_p  # [kg]
            #  NOTE : logg is system, but should be model value.  EDIT LATER
            Hs = cst.Boltzmann * eqtemp / (mmw*1e-2*(10.**float(system_params[planetLetter]['logg'])))  # [m]
            Hs = Hs / (model_params['R*']*sscmks['Rsun'])
            # keep Hs as it's own param (not inside of system_ or model_param)
            #  it has to be this way to match the formatting for regular spectra itk
            out['data'][planetLetter]['Hs'] = Hs

            # save target,planet name, for plotting (in states.py)
            out['target'].append(target)
            out['planets'].append(planetLetter)

            # save the true spectrum (both raw and binned)
            out['data'][planetLetter]['true_spectrum'] = {
                'fluxDepth':fluxDepth_rebin,
                'wavelength_um':wavelength_um_rebin,
                'fluxDepth_norebin':fluxDepth,
                'wavelength_norebin':wavelength_um}

            # save the parameters used to create the spectrum. some could be useful
            # should save both observed value and truth with scatter added in
            #  'system_params' = the info in system() task
            #  'model_params' = what is actually used to create forward model
            out['data'][planetLetter]['system_params'] = {
                'R*':system_params['R*'],
                'T*':system_params['T*'],
                'Rp':system_params[planetLetter]['rp'],
                'Teq':system_params[planetLetter]['teq'],
                'Mp':system_params[planetLetter]['mass']}

            out['data'][planetLetter]['model_params'] = model_params

            # convert to percentage depth (just for plotting, not for the saved results)
            fluxDepth = 100 * fluxDepth
            fluxDepth_rebin = 100 * fluxDepth_rebin
            fluxDepth_observed = 100 * fluxDepth_observed
            uncertainties = 100 * uncertainties

            # PLOT THE SPECTRA
            myfig, ax = plt.subplots(figsize=(6,4))
            plt.title('Ariel simulation : '+target+' '+planetLetter, fontsize=16)
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
            plt.errorbar(wavelength_um_rebin, fluxDepth_observed, yerr=uncertainties,
                         linestyle='None',lw=0.2, color='grey', zorder=2)

            plt.xlim(0.,8.)
            plt.legend()

            # add a scale-height-normalized flux scale on the right axis
            Hs = out['data'][planetLetter]['Hs']
            ax2 = ax.twinx()
            ax2.set_ylabel('$\\Delta$ [H]')
            axmin, axmax = ax.get_ylim()
            rpmed = np.sqrt(np.nanmedian(1.e-2*fluxDepth_rebin))
            if axmin > 0:
                ax2.set_ylim((np.sqrt(1e-2*axmin) - rpmed)/Hs,
                             (np.sqrt(1e-2*axmax) - rpmed)/Hs)
            else:
                print('TROUBLE!! y-axis not scaled by H!!')

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
                        '/ariel_taurexAtmos_' + target+'_'+planetLetter + '.png')

            # REDUNDANT SAVE - above saves to disk; below saves as state vector
            # plt.title('Ariel : '+target+' '+planetLetter+'; sv save for RUNID='+RID,
            #           fontsize=16)
            buf = io.BytesIO()
            myfig.savefig(buf, format='png')
            out['data'][planetLetter]['plot_simspectrum'] = buf.getvalue()

            plt.close(myfig)

            completed_at_least_one_planet = True

    if completed_at_least_one_planet: out['STATUS'].append(True)

    return True
# ------------------------- ------------------------------------------
