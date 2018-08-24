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
HD 97658 :
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
HD 106315 : HD106315
K2-96 : K296
Kepler-16 : KEPLER16
Kepler-51 : Kepler51
WASP-33 : WASP33
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
    overwrite['GJ 9827'] = {
        'b':{'logg':3.1445742076096161,
             'logg_uperr':0.11, 'logg_lowerr':-0.11,
             'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'},
        'c':{'logg':2.9479236198317262,
             'logg_uperr':0.18, 'logg_lowerr':-0.18,
             'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'},
        'd':{'logg':2.7275412570285562,
             'logg_uperr':0.15, 'logg_lowerr':-0.15,
             'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'}}
    overwrite['HAT-P-38'] = {
        'FEH*':0.06, 'FEH*_uperr':0.1, 'FEH*_lowerr':-0.1,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Sato et al. 2012'}
    overwrite['HD 106315'] = {
        'b':{'logg':2.5081746355417742,
             'logg_uperr':0.26, 'logg_lowerr':-0.26,
             'logg_ref':'Crossfield et al. 2017', 'logg_units':'log10[cm.s-2]'},
        'c':{'logg':2.97505059402918,
             'logg_uperr':0.3, 'logg_lowerr':-0.3,
             'logg_ref':'Crossfield et al. 2017', 'logg_units':'log10[cm.s-2]'}}
    overwrite['K2-24'] = {
        'b':{'logg':2.9233923050832749,
             'logg_uperr':0.28, 'logg_lowerr':-0.28,
             'logg_ref':'Petigura et al. 2016', 'logg_units':'log10[cm.s-2]'},
        'c':{'logg':3.2300188146519129,
             'logg_uperr':0.3, 'logg_lowerr':-0.3,
             'logg_ref':'Petigura et al. 2016', 'logg_units':'log10[cm.s-2]'}}
    overwrite['K2-25'] = {
        'b':{'logg':3.8403983985773555,
             'logg_uperr':0.38, 'logg_lowerr':-0.38,
             'logg_ref':'Mann et al. 2016', 'logg_units':'log10[cm.s-2]',
             'sma':0.029535117574370662,
             'sma_uperr':0.0021173577439160705,
             'sma_lowerr':-0.0021173577439160705,
             'sma_ref':'Mann et al. 2016'}}
    overwrite['K2-3'] = {
        'b':{'logg':2.5561945544995859,
             'logg_uperr':0.26, 'logg_lowerr':-0.26,
             'logg_ref':'Crossfield et al. 2016', 'logg_units':'log10[cm.s-2]'},
        'c':{'logg':2.659331865218967,
             'logg_uperr':0.27, 'logg_lowerr':-0.27,
             'logg_ref':'Crossfield et al. 2016', 'logg_units':'log10[cm.s-2]'},
        'd':{'logg':2.7703749486490246,
             'logg_uperr':0.28, 'logg_lowerr':-0.28,
             'logg_ref':'Crossfield et al. 2016', 'logg_units':'log10[cm.s-2]'}}
    overwrite['K2-33'] = {
        'FEH*':0.0, 'FEH*_uperr':0.13, 'FEH*_lowerr':-0.14,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Mann et al. 2016'}
    overwrite['K2-93'] = {
        'b':{'logg':3.26,
             'logg_lowerr':0.07, 'logg_uperr':-0.07,
             'logg_ref':'Dressing et al. 2015', 'logg_units':'log10[cm.s-2]',
             'sma':0.08135713445017638,
             'sma_uperr':9.765981248020531e-05,
             'sma_lowerr':-9.765981248020531e-05,
             'sma_ref':'Dressing et al. 2015'}}
    overwrite['KELT-1'] = {
        'FEH*':0.009, 'FEH*_uperr':0.073, 'FEH*_lowerr':-0.073,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Siverd et al. 2012'}
    overwrite['Kepler-16'] = {
        'R*':0.6679689998646066, 'R*_uperr':0.0013, 'R*_lowerr':-0.0013,
        'R*_ref':'Oroz + dilution factor',
        'b':{'inc':89.776139764168605,
             'inc_uperr':0.0323, 'inc_lowerr':-0.04,
             'inc_ref':'Oroz',
             't0':2457914.235774331,
             't0_uperr':0.004, 't0_lowerr':-0.004,
             't0_ref':'Oroz'}}
    overwrite['WASP-39'] = {
        'FEH*':-0.10, 'FEH*_uperr':0.1, 'FEH*_lowerr':-0.1,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Faedi et al. 2011'}
    overwrite['WASP-43'] = {
        'FEH*':-0.05, 'FEH*_uperr':0.17, 'FEH*_lowerr':-0.17,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Hellier et al. 2011'}
    overwrite['WASP-6'] = {
        'b':{'inc':88.47,
             'inc_uperr':0.65, 'inc_lowerr':-0.47,
             'inc_ref':'Gillon et al. 2009',
             't0':2454596.433,
             't0_uperr':0.00015, 't0_lowerr':-0.00015,
             't0_ref':'Gillon et al. 2009',
             'sma':0.0421,
             'sma_uperr':0.0008,
             'sma_lowerr':-0.0008,
             'sma_ref':'Gillon et al. 2009',
             'period':3.3610060,
             'period_ref':'Gillon et al. 2009'}}
    return overwrite
# ---------------------------- ---------------------------------------
