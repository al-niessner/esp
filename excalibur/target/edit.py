# -- TARGET LIST -- --------------------------------------------------
# FIRST COL HAS TO BE SOLVABLE BY
# https://archive.stsci.edu/hst/
# SECOND COL [OPTIONAL] HAS TO BE 'Default Alias' RETURNED FROM
# https://exoplanetarchive.ipac.caltech.edu/index.html
# OR ALIAS KNOWN BY https://archive.stsci.edu/hst/
# G ROUDIER, M SWAIN, A ROWEN  May 29 2018
def targetlist():
    '''
55 Cnc : 
CoRoT-1 :
CoRoT-2 :
GJ 436 :
GJ 1132 : 
GJ 1214 :
GJ 3053 : LHS 1140
GJ 3470 :
GJ 9827 :
HAT-P-1 :
HAT-P-3 :
HAT-P-7 : 
HAT-P-11 :
HAT-P-12 :
HAT-P-13 :
HAT-P-17 :
HAT-P-18 :
HAT-P-26 :
HAT-P-32 :
HAT-P-38 :
HAT-P-41 :
HATS-7 :
HD 106315 : 
HD 149026 : 
HD 189733 : 
HD 209458 : 
HD 219134 :
KELT-1 : 
KELT-7 :
KELT-11 : 
Kepler-9 :
Kepler-11 : 
Kepler-13 : 
Kepler-16 :
Kepler-51 : 
Kepler-79 : 
Kepler-138 : 
TRAPPIST-1 : 
TrES-2 : 
TrES-3 : 
TrES-4 : 
K2-3 : 
K2-18 : 
K2-24 : 
K2-25 : 
K2-33 : 
K2-93 : HIP 41378
K2-96 : HD 3167
WASP-4 : 
WASP-6 : 
WASP-12 : 
WASP-17 : 
WASP-18 : 
WASP-19 : 
WASP-29 : 
WASP-31 :
WASP-33 : 
WASP-39 : 
WASP-43 : 
WASP-52 : 
WASP-62 : 
WASP-63 : 
WASP-67 : 
WASP-69 : 
WASP-74 : 
WASP-76 : 
WASP-79 : 
WASP-80 :
WASP-96 :
WASP-101 : 
WASP-103 :
WASP-107 : 
WASP-121 : 
WASP-127 : 
XO-1 : 
XO-2 : XO-2 N
    '''
    return
# ----------------- --------------------------------------------------
# -- TARGET ON DISK -- -----------------------------------------------
# FIRST COL MATCHES TARGET LIST NAME
# SECOND COL IS THE DATA FOLDER IN /proj/sdp/data/sci
# G ROUDIER May 27 2018
def targetondisk():
    '''
Kepler-16 : KEPLER16
Kepler-51 : Kepler51
    '''
    return
# -------------------- -----------------------------------------------
# -- ACTIVE FILTERS -- -----------------------------------------------
# OBSERVATORY/INSTRUMENT/DETECTOR/FILTER/MODE
# G ROUDIER  May 30 2018
def activefilters():
    '''
HST-WFC3-IR-G141-SCAN
HST-WFC3-IR-G141-STARE
HST-WFC3-IR-G102-SCAN
HST-WFC3-IR-G102-STARE
    '''
    return
# -------------------- -----------------------------------------------
# -- PRIORITY PARAMETERS -- ------------------------------------------
def ppar():
    '''
'starID': stellar ID from the targetlist() i.e: 'WASP-12'
'planet': planet letter i.e: 'b'
'[units]': value in units indicated inside brackets
ref: reference
overwrite[starID] = 
{
    'R*':[Rsun], 'R*_ref':ref,
    'T*':[K], 'T*_lowerr':[K], 'T*_uperr':[K], 'T*_ref':ref,
    'FEH*':[dex], 'FEH*_lowerr':[dex], 'FEH*_uperr':[dex], 
    'FEH*_units':'[Fe/H]', 'FEH*_ref':ref,
    'LOGG*':[dex CGS], 'LOGG*_lowerr':[dex CGS], 'LOGG*_uperr':[dex CGS], 
    'LOGG*_ref':ref,
    planet:
    {
    'inc':[degrees], 'inc_lowerr':[dex CGS], 'inc_uperr':[dex CGS], 
    'inc_ref':ref,
    't0':[JD], 't0_lowerr':[JD], 't0_uperr':[JD], 't0_ref':ref,
    'sma':[AU], 'sma_lowerr':[AU], 'sma_uperr':[AU], 'sma_ref':ref,
    'period':[days], 'period_ref':ref, 
    'ecc':[], 'ecc_ref':ref, 
    'rp':[Rjup], 'rp_ref':ref
    }
}
    '''
    overwrite = {}
    overwrite['WASP-43'] = {
        'FEH*':-0.05, 'FEH*_uperr':0.17, 'FEH*_lowerr':-0.17,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Hellier et al. 2011'
    }
    return overwrite
# ---------------------------- ---------------------------------------
