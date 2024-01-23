'''ariel forwardModels ds'''
import os
import excalibur
import excalibur.system.core as syscore
from excalibur.cerberus.core import hazelib
from excalibur.cerberus.forwardModel import crbmodel
import taurex
import taurex.log
taurex.log.disableLogging()
from taurex.chemistry import TaurexChemistry, ConstantGas
from taurex.contributions import AbsorptionContribution, \
    CIAContribution, RayleighContribution, SimpleCloudsContribution
# from taurex.contributions import FlatMieContribution
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.temperature import Guillot2010, Isothermal
from taurex.cache import OpacityCache, CIACache
from taurex_ace import ACEChemistry

TAUREX_DATA_ROOT = os.environ.get('TAUREX_DATA_ROOT', excalibur.context['data_dir']+'/taurex')
OpacityCache().clear_cache()
OpacityCache().set_opacity_path(os.path.join (TAUREX_DATA_ROOT,
                                              'xsec/xsec_sampled_R15000_0.3-50'))
CIACache().set_cia_path(os.path.join (TAUREX_DATA_ROOT, 'cia/HITRAN'))

# ----------------------------------------------------------------------------------------------
def makeTaurexAtmos(model_params, mixratios=False, clouds=True):
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
    #  P-T profile
    temperature_profile = Guillot2010(T_irr=float(model_params['Teq']))
    #  isothermal atmosphere (to match cerberus)
    temperature_profile = Isothermal(T=float(model_params['Teq']))

    if mixratios:
        # use a given dictionary of mixing ratios
        chemistry = TaurexChemistry(fill_gases=['H2','He'],ratio=0.172)
        # water = ConstantGas('H2O', mix_ratio=1.2e-4*model_params['metallicity'])
        water = ConstantGas('H2O', mix_ratio=10.**(mixratios['H2O'] - 6.))
        CO = ConstantGas('CO', mix_ratio=10.**(mixratios['CO'] - 6.))
        N2 = ConstantGas('N2', mix_ratio=10.**(mixratios['N2'] - 6.))
        NH3 = ConstantGas('NH3', mix_ratio=10.**(mixratios['NH3'] - 6.))
        CH4 = ConstantGas('CH4', mix_ratio=10.**(mixratios['CH4'] - 6.))
        chemistry.addGas(water)
        chemistry.addGas(CH4)
        chemistry.addGas(CO)
        chemistry.addGas(N2)
        chemistry.addGas(NH3)

    else:
        # assume equilibrium chemistry
        # NOTE that Taurex wants a linear value for metallicity, not the usual dex
        chemistry = ACEChemistry(metallicity=10.**model_params['metallicity'],
                                 co_ratio=0.54951*10.**model_params['C/O'])

    # create an atmospheric model based on the above parameters
    tm = TransmissionModel(star=star,
                           planet=planet,
                           temperature_profile=temperature_profile,
                           chemistry=chemistry,
                           # atm_min_pressure=3e-1,  # 3 microbar top; same as cerberus standard 15 Hs
                           atm_min_pressure=2e-3,  # 0.02 microbar top; same as cerberus with 20 Hs
                           atm_max_pressure=1e6,   # switch atmos base to the 10-bar standard
                           nlayers=100)  # use the same number of layers as cerberus (was 30 before)
    # add some more physics
    tm.add_contribution(AbsorptionContribution())
    # adding in H2-H and He-H (to match cerberus)
    # tm.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He','H2-H','He-H']))
    # no wait actually these don't exist!  going back to original
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
def makeCerberusAtmos(wavelength_um, model_params, xslib, planetLetter, mixratios=None,
                      Hsmax=15., solrad=10.):
    '''
    Create a simulated spectrum using the code that's better than the other ones
    '''

    # print('modelparams',model_params)

    # EQUILIBRIUM TEMPERATURE
    Teq = model_params['Teq']

    # CLOUD/HAZE PARAMETERS
    ctp = model_params['CTP']
    hza = model_params['HScale']
    hzloc = model_params['HLoc']
    hzthick = model_params['HThick']

    # ABUNDANCES
    if mixratios:
        # use the given mixing ratios
        tceqdict = None
    else:
        # equilibrium chemistry
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
    # fmc, fmc_by_molecule = crbmodel(None, float(hza), float(ctp), solidr, orbp,
    fmc, fmc_by_molecule = crbmodel(mixratios, float(hza), float(ctp), solidr, orbp,
                                    xslib['data'][planetLetter]['XSECS'],
                                    xslib['data'][planetLetter]['QTGRID'],
                                    float(Teq), wavelength_um,
                                    Hsmax=Hsmax, solrad=solrad,
                                    # np.array(ctxt.spc['data'][ctxt.p]['WB']),
                                    hzlib=crbhzlib,  hzp='AVERAGE',
                                    hztop=float(hzloc), hzwscale=float(hzthick),
                                    cheq=tceqdict, pnet=planetLetter,
                                    sphshell=True, verbose=False, debug=False,
                                    break_down_by_molecule=True)

    return fmc, fmc_by_molecule
