# -- TARGET LIST -- --------------------------------------------------
# FIRST COL HAS TO BE SOLVABLE BY
# https://archive.stsci.edu/hst/
# SECOND COL [OPTIONAL] HAS TO BE 'Default Alias' RETURNED FROM
# https://exoplanetarchive.ipac.caltech.edu/index.html
# OR ALIAS KNOWN BY https://archive.stsci.edu/hst/
# MINOR CHANGE
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
Kepler-1625 :
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
def targetondisk():
    '''
55 Cnc : 55CNC
CoRoT-1 : COROT1
CoRoT-2 : COROT2
GJ 1132 : GJ1132
GJ 1214 : GJ1214
GJ 3053 : GJ3053
GJ 3470 : GJ3470
GJ 436 : GJ436
GJ 9827 : GJ9827
HAT-P-1 : HATP1
HAT-P-11 : HATP11
HAT-P-12 : HATP12
HAT-P-13 : HATP13
HAT-P-17 : HATP17
HAT-P-18 : HATP18
HAT-P-26 : HATP26
HAT-P-3 : HATP3
HAT-P-32 : HATP32
HAT-P-38 : HATP38
HAT-P-41 : HATP41
HAT-P-7 : HATP7
HATS-7 : HATS7
HD 106315 : HD106315
HD 149026 : HD149026
HD 189733 : HD189733
HD 209458 : HD209458
HD 219134 : HD219134
HD 97658 : HD97658
K2-18 : K218
K2-24 : K224
K2-25 : K225
K2-3 : K23
K2-33 : K233
K2-93 : K293
K2-96 : K296
KELT-1 : KELT1
KELT-7 : KELT7
Kepler-11 : KEPLER11
Kepler-13 : KEPLER13
Kepler-138 : KEPLER138
Kepler-16 : KEPLER16
Kepler-1625 : KEPLER1625
Kepler-51 : KEPLER51
Kepler-9 : KEPLER9
TRAPPIST-1 : TRAPPIST1
TrES-2 : TRES2
TrES-3 : TRES3
TrES-4 : TRES4
WASP-101 : WASP101
WASP-103 : WASP103
WASP-107 : WASP107
WASP-12 : WASP12
WASP-121 : WASP121
WASP-17 : WASP17
WASP-18 : WASP18
WASP-19 : WASP19
WASP-29 : WASP29
WASP-31 : WASP31
WASP-33 : WASP33
WASP-39 : WASP39
WASP-4 : WASP4
WASP-43 : WASP43
WASP-52 : WASP52
WASP-6 : WASP6
WASP-62 : WASP62
WASP-63 : WASP63
WASP-67 : WASP67
WASP-69 : WASP69
WASP-74 : WASP74
WASP-76 : WASP76
WASP-79 : WASP79
WASP-80 : WASP80
WASP-96 : WASP96
XO-1 : XO1
XO-2 : XO2
    '''
    return
# -------------------- -----------------------------------------------
# -- ACTIVE FILTERS -- -----------------------------------------------
# OBSERVATORY/INSTRUMENT/DETECTOR/FILTER/MODE
# G ROUDIER  May 30 2018
def activefilters():
    '''
HST-WFC3-IR-G141-SCAN
HST-WFC3-IR-G102-SCAN
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
    overwrite['GJ 436'] = {
        'b':{'inc':86.774,
             'inc_uperr':0.03, 'inc_lowerr':-0.03,
             'inc_ref':'Knutson et al. 2014',
             't0':2456295.431924,
             't0_uperr':0.000045, 't0_lowerr':-0.000045,
             't0_ref':'Knutson et al. 2014',
             'ecc':0, 'ecc_ref':'Knutson et al. 2014',
             'sma':0.030826087286013763,
             'sma_uperr':0.0002,
             'sma_lowerr':-0.0002,
             'sma_ref':'Knutson et al. 2014',
             'period':2.64389782,
             'period_uperr':0.00000008, 'period_lowerr':-0.00000008,
             'period_ref':'Knutson et al. 2014'}}
    overwrite['GJ 9827'] = {
        'R*':0.637, 'R*_uperr':0.063, 'R*_lowerr':-0.063,
        'R*_ref':'Prieto-Arranz et al. 2018',
        'b':{'inc':88.33,
             'inc_uperr':1.15, 'inc_lowerr':-2.1,
             'inc_ref':'Prieto-Arranz et al. 2018',
             't0':2457738.52646,
             't0_uperr':0.00044, 't0_lowerr':-0.00042,
             't0_ref':'Prieto-Arranz et al. 2018',
             'sma':0.0210,
             'sma_uperr':0.0024,
             'sma_lowerr':-0.0026,
             'sma_ref':'Prieto-Arranz et al. 2018',
             'period':1.208966,
             'period_uperr':0.000012, 'period_lowerr':-0.000012,
             'period_ref':'Prieto-Arranz et al. 2018',
             'logg':3.1445742076096161,
             'logg_uperr':0.11, 'logg_lowerr':-0.11,
             'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'},
        'c':{'inc':89.07,
             'inc_uperr':0.59, 'inc_lowerr':-0.92,
             'inc_ref':'Prieto-Arranz et al. 2018',
             't0':2457738.54961,
             't0_uperr':0.00146, 't0_lowerr':-0.00145,
             't0_ref':'Prieto-Arranz et al. 2018',
             'sma':0.0439,
             'sma_uperr':0.0050,
             'sma_lowerr':-0.0055,
             'sma_ref':'Prieto-Arranz et al. 2018',
             'period':3.648227,
             'period_uperr':0.000117, 'period_lowerr':-0.000119,
             'period_ref':'Prieto-Arranz et al. 2018',
             'logg':2.9479236198317262,
             'logg_uperr':0.18, 'logg_lowerr':-0.18,
             'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'},
        'd':{'inc':87.703,
             'inc_uperr':0.081, 'inc_lowerr':-0.253,
             'inc_ref':'Prieto-Arranz et al. 2018',
             't0':2457740.96198,
             't0_uperr':0.00084, 't0_lowerr':-0.00086,
             't0_ref':'Prieto-Arranz et al. 2018',
             'sma':0.0625,
             'sma_uperr':0.0071,
             'sma_lowerr':-0.0078,
             'sma_ref':'Prieto-Arranz et al. 2018',
             'period':6.201419,
             'period_uperr':0.000128, 'period_lowerr':-0.000128,
             'period_ref':'Prieto-Arranz et al. 2018',
             'logg':2.7275412570285562,
             'logg_uperr':0.15, 'logg_lowerr':-0.15,
             'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'}}
    overwrite['HAT-P-11'] = {
        'b':{'inc':89.549,
             'inc_uperr':0.114, 'inc_lowerr':-0.114,
             'inc_ref':'Fraine et al. 2014',
             'rp':0.3895538538577743,
             'rp_uperr':0.0016542969842779612, 'rp_lowerr':-0.0016542969842779612,
             'rp_ref':'Fraine et al. 2014',
             'sma':0.05415469125390434,
             'sma_uperr':0.00018973906424725608, 'sma_lowerr':-0.00018973906424725608,
             'sma_ref':'Fraine et al. 2014',
             'ecc':0.232,
             'ecc_uperr':0.054, 'ecc_lowerr':-0.054,
             'ecc_ref':'Fraine et al. 2014',
             'period':4.8878018,
             'period_uperr':0.0000071, 'period_lowerr':-0.0000071,
             'period_ref':'Fraine et al. 2014'}}
    overwrite['HAT-P-17'] = {
        'R*':0.838, 'R*_uperr':0.021, 'R*_lowerr':-0.021,
        'R*_ref':'Howard et al. 2012',
        'b':{'inc':89.2,
             'inc_uperr':0.2, 'inc_lowerr':-0.1,
             'inc_ref':'Howard et al. 2012',
             't0':2454801.16945,
             't0_uperr':0.0002, 't0_lowerr':-0.0002,
             't0_ref':'Howard et al. 2012',
             'sma':0.06,
             'sma_uperr':0.0014, 'sma_lowerr':-0.0014,
             'sma_ref':'Howard et al. 2012 + high ecc',
             'period':10.338523,
             'period_ref':'Howard et al. 2012'}}
    overwrite['HAT-P-3'] = {
        'b':{'inc':87.24,
             'inc_uperr':0.69, 'inc_lowerr':-0.69,
             'inc_ref':'Torres et al. 2017',
             'sma':0.03894,
             'sma_uperr':0.0007, 'sma_lowerr':-0.0007,
             'sma_ref':'Torres et al. 2017',
             't0':2454218.81,
             't0_uperr':0.0029, 't0_lowerr':-0.0029,
             't0_ref':'Torres et al. 2017 + GMR',
             'logg':3.310,
             'logg_uperr':0.066, 'logg_lowerr':-0.066,
             'logg_ref':'Torres et al. 2017', 'logg_units':'log10[cm.s-2]'}}
    overwrite['HAT-P-38'] = {
        'FEH*':0.06, 'FEH*_uperr':0.1, 'FEH*_lowerr':-0.1,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Sato et al. 2012'}
    overwrite['HAT-P-41'] = {
        'R*':1.683, 'R*_uperr':0.058, 'R*_lowerr':-0.036,
        'R*_ref':'Hartman et al. 2012',
        'b':{'inc':87.7,
             'inc_uperr':1.0, 'inc_lowerr':-1.0,
             'inc_ref':'Hartman et al. 2012',
             't0':2454983.86167,
             't0_uperr':0.00107, 't0_lowerr':-0.00107,
             't0_ref':'Hartman et al. 2012',
             'sma':0.0426,
             'sma_uperr':0.0005, 'sma_lowerr':-0.0005,
             'sma_ref':'Hartman et al. 2012',
             'period':2.694047,
             'period_ref':'Hartman et al. 2012'}}
    overwrite['HD 106315'] = {
        'b':{'logg':2.5081746355417742,
             'logg_uperr':0.26, 'logg_lowerr':-0.26,
             'logg_ref':'Crossfield et al. 2017', 'logg_units':'log10[cm.s-2]'},
        'c':{'logg':2.97505059402918,
             'logg_uperr':0.3, 'logg_lowerr':-0.3,
             'logg_ref':'Crossfield et al. 2017', 'logg_units':'log10[cm.s-2]'}}
    overwrite['HD 97658'] = {
        'b':{'inc':89.85,
             'inc_uperr':0.15, 'inc_lowerr':-0.48,
             'inc_ref':'Knutson et al. 2014',
             't0':2456523.12537,
             't0_uperr':0.00049, 't0_lowerr':-0.00049,
             't0_ref':'Knutson et al. 2014',
             'sma':0.09030091308645881,
             'sma_uperr':0.004164, 'sma_lowerr':-0.004164,
             'sma_ref':'Knutson et al. 2014',
             'ecc':0, 'ecc_ref':'Knutson et al. 2014',
             'period':9.489264,
             'period_ref':'Knutson et al. 2014'}}
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
    overwrite['K2-96'] = {
        'b':{'logg':3.22900042686,
             'logg_lowerr':0.3, 'logg_uperr':-0.3,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'},
        'c':{'logg':3.02377443746,
             'logg_lowerr':0.3, 'logg_uperr':-0.3,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'}}
    overwrite['KELT-1'] = {
        'FEH*':0.009, 'FEH*_uperr':0.073, 'FEH*_lowerr':-0.073,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Siverd et al. 2012',
        'b':{'inc':87.6,
             'inc_uperr':1.4, 'inc_lowerr':-1.9,
             'inc_ref':'Siverd et al. 2012',
             't0':2455933.61,
             't0_uperr':0.00041, 't0_lowerr':-0.00039,
             't0_ref':'Siverd et al. 2012 + GMR',
             'sma':0.02470,
             'sma_uperr':0.00039, 'sma_lowerr':-0.00039,
             'sma_ref':'Siverd et al. 2012',
             'ecc':0, 'ecc_ref':'Siverd et al. 2012',
             'period':1.217513,
             'period_ref':'Siverd et al. 2012'}}
    overwrite['Kepler-16'] = {
        'R*':0.665924608009903, 'R*_uperr':0.0013, 'R*_lowerr':-0.0013,
        'R*_ref':'Oroz + GMR',
        'b':{'inc':89.7511397641686,
             'inc_uperr':0.0323, 'inc_lowerr':-0.04,
             'inc_ref':'Oroz + GMR',
             't0':2457914.235774330795,
             't0_uperr':0.004, 't0_lowerr':-0.004,
             't0_ref':'Oroz',
             'sma':0.7048,
             'sma_uperr':0.0011,
             'sma_lowerr':-0.0011,
             'sma_ref':'Doyle et al. 2011',
             'period':228.776,
             'period_ref':'Doyle et al. 2011'}}
    overwrite['Kepler-1625'] = {
        'b':{'logg':4.00,
             'logg_uperr':0.4, 'logg_lowerr':-0.4,
             'logg_ref':'Teachey et al. 2018', 'logg_units':'log10[cm.s-2]',
             'sma':0.898,
             'sma_uperr':0.1,
             'sma_lowerr':-0.1,
             'sma_ref':'Morton et al. 2016'}}
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
