'''ariel planet_metallicity ds'''
# -- IMPORTS -- ------------------------------------------------------
import logging; log = logging.getLogger(__name__)

import numpy as np
# np.random.seed(123)

# ______________________________________________________

def massMetalRelationDisp(logmetStar,Mp):
    '''
    Add some realistic scatter to the mass-metallicity relation
    (not that we know reality)
    '''
    logmet = massMetalRelation(logmetStar,Mp)

    # FINESSE used a dispersion of just 0.3
    #  Swain analysis of Thorgren 2016 finds a lot more scatter (0.8)
    dispersion = 0.8

    logmet += np.random.normal(scale=dispersion)

    return logmet
# ______________________________________________________

def massMetalRelation(logmetStar, Mp):
    '''
    Assume an inverse-linear relationship between planet mass and metallicity
    Include a limit on metallicity of +2.0 dex, relative to the parent star
    The planet mass (Mp) is in Jupiter masses
    '''

    # Mp = 1Jup gives met = 0.5 dex
    # Mp = 10/318 = 10Earths gives met = +2 dex
    # Mp = 1/318 = 1Earth would give met = +3 dex, but capped at +2 dex

    slope = -1.
    maxMetal = 2.
    # Mpivot = 1.   # (earth units)
    # intercept = maxMetal - slope*Mpivot
    # Mpivot = -1.5  # (jupiter units)
    intercept = 0.5  # metallicity for Jupiter mass

    logmet = intercept + slope*np.log10(Mp)
    logmet = min(maxMetal,logmet)
    logmet += logmetStar

    return logmet
# ______________________________________________________

def randomStarMetal():
    '''
    If there's no stellar metallicity, pull from random distribution
    '''

    # this is from my excel check of Hinkel's Hypatia catalog
    #  logmetStar=0.06 + 0.29*random.gauss(0.,1.)
    # from Kepler-detection-based (Buchhave 2011)
    logmetStar=-0.01 + 0.25*np.random.normal()

    return logmetStar
# ______________________________________________________

def randomCtoO():
    '''
    Assign a random C-to-O ratio to each system
    Allow a small fraction (~5%) of stars to have more C than O
    Actually that's too much.  Consensus at the May2023 JWST conference was less than that I think
    Let's go with -0.2+-0.1, which gives 2.3% with more C than O
    '''

    # this is from my excel check of Hinkel's Hypatia catalog
    # logCtoO=-0.13 + 0.19*random.gauss(0.,1.)

    # this is motivated by Fortney's argument that C>O is rare
    # he gets 1-5% and suggests even lower
    # let's just stick with 5% C/O>1,
    # in order to see how well FINESSE can do with some oddballs
    # this gives 4.8% with C>O
    # logCtoO=-0.2 + 0.12*np.random.normal()
    logCtoO=-0.2 + 0.1*np.random.normal()

    # reminder: solar is -0.26
    #  we seem to be below average

    CtoO = 10.**logCtoO
    return CtoO
# ______________________________________________________
