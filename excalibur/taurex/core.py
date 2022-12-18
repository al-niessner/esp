'''transit core ds'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import numpy
import os
import taurex
import taurex.log
taurex.log.disableLogging()

from taurex.cache import OpacityCache,CIACache
from taurex.chemistry import TaurexChemistry
from taurex.chemistry import ConstantGas
from taurex.contributions import AbsorptionContribution
from taurex.contributions import CIAContribution
from taurex.contributions import RayleighContribution
from taurex.model import TransmissionModel
from taurex.planet import Planet
from taurex.stellar import BlackbodyStar
from taurex.temperature import Guillot2010

def init():
    '''init the library so that tests do not need to pass this step'''
    TAUREX_DATA_ROOT = os.environ.get ('TAUREX_DATA_ROOT', '/proj/data/taurex')
    OpacityCache().clear_cache()
    OpacityCache().set_opacity_path(os.path.join (TAUREX_DATA_ROOT,
                                                  "xsec/xsec_sampled_R15000_0.3-50"))
    CIACache().set_cia_path(os.path.join (TAUREX_DATA_ROOT, "cia/HITRAN"))
    return

def tsi(spectrums:{}, parameters:{}):
    '''actually inject data into transit.spectrum data'''
    # star is constant for the system (hopefully)
    star = BlackbodyStar(temperature=parameters['priors']['T*'],
                         radius=parameters['priors']['R*'])
    for planet,spectrum in spectrums['data'].items():
        # setup an atmospheric configuration on a planet
        guillot = Guillot2010(T_irr=1200.0)
        planet = Planet(planet_radius=parameters['priors'][planet]['rp'],
                        planet_mass=parameters['priors'][planet]['mass'])
        chemistry = TaurexChemistry(fill_gases=['H2','He'],ratio=0.172)
        h2o = ConstantGas('H2O',mix_ratio=1.2e-4)
        chemistry.addGas(h2o)
        # instantiate the model
        tm = TransmissionModel(planet=planet,
                               temperature_profile=guillot,
                               chemistry=chemistry,
                               star=star,
                               atm_min_pressure=1e-0,
                               atm_max_pressure=1e6,
                               nlayers=30)
        # add the physics
        tm.add_contribution(AbsorptionContribution())
        tm.add_contribution(CIAContribution(cia_pairs=['H2-H2','H2-He']))
        tm.add_contribution(RayleighContribution())
        # build then run the model to product a spectral result
        tm.build()
        wn,model = tm.model()[:2]
        wl = 1e4/wn
        # compute the additive spectrum
        spectrum['ORIG'] = spectrum['ES'].copy()
        spectrum['TR'] = numpy.zeros(spectrum['ES'].shape)
        for i,(m,M) in enumerate(zip(spectrum['WBlow'],spectrum['WBup'])):
            band = numpy.logical_and(m <= wl,wl <= M)
            spectrum['TR'][i] = numpy.sqrt(model[band].mean())
            spectrum['ES'][i] = numpy.sqrt((spectrum['TR'][i]**2 +
                                            spectrum['ORIG'][i]**2) / 2)
            pass
        pass
    return

def tsiversion():
    '''version of algoritm and it is kept here so only need to edit core'''
    return dawgie.VERSION(1,0,1)
