'''ariel forwardModels ds'''
import os
import excalibur
import excalibur.system.core as syscore
from excalibur.cerberus.core import hazelib
from excalibur.cerberus.forwardModel import crbmodel
import taurex
import taurex.log
taurex.log.disableLogging()
# from taurex.chemistry import TaurexChemistry
# from taurex.chemistry import ConstantGas
# from taurex.chemistry import ACEChemistry
from taurex_ace import ACEChemistry
from taurex.contributions import AbsorptionContribution
from taurex.contributions import CIAContribution
from taurex.contributions import RayleighContribution
from taurex.contributions import SimpleCloudsContribution
# from taurex.contributions import FlatMieContribution
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.temperature import Guillot2010
from taurex.cache import OpacityCache,CIACache

TAUREX_DATA_ROOT = os.environ.get('TAUREX_DATA_ROOT', excalibur.context['data_dir']+'/taurex')
OpacityCache().clear_cache()
OpacityCache().set_opacity_path(os.path.join (TAUREX_DATA_ROOT,
                                              'xsec/xsec_sampled_R15000_0.3-50'))
CIACache().set_cia_path(os.path.join (TAUREX_DATA_ROOT, 'cia/HITRAN'))

# ----------------------------------------------------------------------------------------------
def makeTaurexAtmos(model_params, clouds=True):
    '''
    Create a simulated spectrum using the Taurex package
    '''

    # define the star based on temperature & radius
    star = BlackbodyStar(temperature=model_params['T*'],
                         radius=model_params['R*'])

    # define the planet based on mass & radius
    planet = Planet(planet_radius=model_params['Rp'],
                    planet_mass=model_params['Mp'])

    # set the planet's vertical temperature profile
    temperature_profile = Guillot2010(
        T_irr=float(model_params['Teq']))

    # this is the old version copied from taurex task
    # chemistry = TaurexChemistry(fill_gases=['H2','He'],ratio=0.172)
    # water = ConstantGas('H2O', mix_ratio=1.2e-4*model_params['metallicity'])
    # chemistry.addGas(water)

    # assume equilibrium chemistry
    # NOTE that Taurex wants a linear value for metallicity, not the usual dex
    chemistry = ACEChemistry(metallicity=10.**model_params['metallicity'],
                             co_ratio=0.54951*10.**model_params['C/O'])

    # create an atmospheric model based on the above parameters
    tm = TransmissionModel(star=star,
                           planet=planet,
                           temperature_profile=temperature_profile,
                           chemistry=chemistry,
                           # atm_min_pressure=1e0,
                           # atm_max_pressure=1e6,
                           atm_min_pressure=1e-3,  # 1 microbar is better; 1 mbar cuts off many lines
                           atm_max_pressure=1e4,   # switch atmos base to the 10-bar standard
                           nlayers=30)
    # add some more physics
    tm.add_contribution(AbsorptionContribution())
    tm.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
    tm.add_contribution(RayleighContribution())

    if clouds:
        # include Clouds!
        P_mbar = 1.
        P_pascals = P_mbar * 100
        tm.add_contribution(SimpleCloudsContribution(clouds_pressure=P_pascals))

        model_params['clouds'] = '1mbar'
    else:
        model_params['clouds'] = 'none'

        #  Here's an alternate way to add clouds in Taurex:
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
    return tm.model()

# ----------------------------------------------------------------------------------------------
def makeCerberusAtmos(wavelength_um, model_params, xslib, planetLetter, clouds=True):
    '''
    Create a simulated spectrum using the code that's better than the other ones
    '''

    if clouds:
        # CLOUD TOP PRESSURE
        # ctp = -1.  # 0.1 bar = 100 mbar
        ctp = -2.  # 0.01 bar = 10 mbar
        # HAZE SCAT. CROSS SECTION SCALE FACTOR
        hza = 0.  # some haze (1x times the nominal Jupiter model)
        # HAZE POWER INDEX FOR SPHERICAL SHELL
        hzloc = 0
        hzthick = 0
    else:
        # these are the same numbers as set in cerb/forwardModel/clearfmcerberus()
        ctp = 3.    # cloud deck is very deep - 1000 bars
        hza = -10.  # small number means essentially no haze
        hzloc = 0.
        hzthick = 0.

    # print('modelparams',model_params)

    # EQUILIBRIUM TEMPERATURE
    Teq = model_params['Teq']

    # ABUNDANCES
    tceqdict = {}
    tceqdict['XtoH'] = model_params['metallicity']
    tceqdict['CtoO'] = model_params['C/O']
    tceqdict['NtoO'] = 0
    # print('cloudfree forward model input chem =',tceqdict)

    ssc = syscore.ssconstants(mks=True)
    solidr = model_params['Rp']*ssc['Rjup']  # MK
    # orbp = fin['priors'].copy()
    # orbp = model_params   # just hope this covers it ig
    # orbp = {'sma':model_params['sma']}
    # planetLetter = 'placeholder'
    orbp = {'R*':model_params['R*'],
            planetLetter:{'logg':model_params['logg']}}

    crbhzlib = {'PROFILE':[]}
    # hazedir = os.path.join(excalibur.context['data_dir'], 'CERBERUS/HAZE')
    # print('hazedir: ',hazedir)
    hazelib(crbhzlib)
    # print('haze lib',crbhzlib)

    # print('wavelength range',wavelength_um[0],wavelength_um[-1])

    # CERBERUS FORWARD MODEL
    fmc = crbmodel(None, float(hza), float(ctp), solidr, orbp,
                   xslib['data'][planetLetter]['XSECS'],
                   xslib['data'][planetLetter]['QTGRID'],
                   float(Teq), wavelength_um,
                   # np.array(ctxt.spc['data'][ctxt.p]['WB']),
                   hzlib=crbhzlib,  hzp='AVERAGE',
                   hztop=float(hzloc), hzwscale=float(hzthick),
                   cheq=tceqdict, pnet=planetLetter,
                   sphshell=True, verbose=False, debug=False)

    return fmc
