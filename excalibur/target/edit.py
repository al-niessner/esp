# -- CREATE -- ------------------------------------------------------
def createversion():
    '''
    1.2.0: Added Spitzer targets
    '''
    import dawgie
    return dawgie.VERSION(1,2,0)
# ------------ -------------------------------------------------------
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
CoRoT-7 :
GJ 1132 :
GJ 1214 :
GJ 3053 : LHS 1140
GJ 3470 :
GJ 436 :
GJ 9827 :
HAT-P-1 :
HAT-P-11 :
HAT-P-12 :
HAT-P-13 :
HAT-P-15 :
HAT-P-16 :
HAT-P-17 :
HAT-P-18 :
HAT-P-19 :
HAT-P-2 :
HAT-P-20 :
HAT-P-22 :
HAT-P-23 :
HAT-P-26 :
HAT-P-3 :
HAT-P-30 :
HAT-P-32 :
HAT-P-33 :
HAT-P-34 :
HAT-P-38 :
HAT-P-4 :
HAT-P-40 :
HAT-P-41 :
HAT-P-5 :
HAT-P-6 :
HAT-P-7 :
HAT-P-8 :
HATS-28 :
HATS-3 :
HATS-65 :
HATS-7 :
HD 106315 :
HD 149026 :
HD 17156 :
HD 189733 :
HD 209458 :
HD 219134 :
HD 97658 :
K2-124 :
K2-132 :
K2-136 :
K2-138 :
K2-167 :
K2-174 :
K2-18 :
K2-19 :
K2-21 :
K2-212 :
K2-22 :
K2-24 :
K2-25 :
K2-26 :
K2-28 :
K2-289 :
K2-3 :
K2-31 :
K2-32 :
K2-33 :
K2-36 :
K2-52 :
K2-53 :
K2-55 :
K2-58 :
K2-79 :
K2-87 :
K2-9 :
K2-90 :
K2-93 : HIP 41378
K2-95 :
K2-96 : HD 3167
K2-97 :
KELT-1 :
KELT-11 :
KELT-14 :
KELT-16 :
KELT-20 :
KELT-3 :
KELT-7 :
KELT-9 :
Kepler-11 :
Kepler-13 :
Kepler-138 :
Kepler-16 :
Kepler-1625 :
Kepler-51 :
Kepler-9 :
LHS 3844 :
OGLE-TR-056 :
OGLE-TR-10 :
Qatar-1 :
Qatar-2 :
TrES-1 :
TrES-2 :
TrES-3 :
TrES-4 :
TRAPPIST-1 :
WASP-1 :
WASP-10 :
WASP-100 :
WASP-101 :
WASP-103 :
WASP-104 :
WASP-107 :
WASP-11 :
WASP-12 :
WASP-121 :
WASP-127 :
WASP-13 :
WASP-131 :
WASP-14 :
WASP-140 :
WASP-15 :
WASP-16 :
WASP-17 :
WASP-18 :
WASP-19 :
WASP-2 :
WASP-21 :
WASP-24 :
WASP-26 :
WASP-28 :
WASP-29 :
WASP-3 :
WASP-31 :
WASP-32 :
WASP-33 :
WASP-34 :
WASP-35 :
WASP-36 :
WASP-37 :
WASP-38 :
WASP-39 :
WASP-4 :
WASP-43 :
WASP-46 :
WASP-48 :
WASP-49 :
WASP-5 :
WASP-50 :
WASP-52 :
WASP-6 :
WASP-62 :
WASP-63 :
WASP-64 :
WASP-65 :
WASP-67 :
WASP-69 :
WASP-7 :
WASP-72 :
WASP-74 :
WASP-75 :
WASP-76 :
WASP-78 :
WASP-79 :
WASP-8 :
WASP-80 :
WASP-95 :
WASP-96 :
WASP-97 :
XO-1 :
XO-2 : XO-2 N
XO-3 :
XO-4 :
XO-5 :
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
CoRoT-7 : COROT7
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
HAT-P-15 : HATP15
HAT-P-16 : HATP16
HAT-P-17 : HATP17
HAT-P-18 : HATP18
HAT-P-19 : HATP19
HAT-P-2 : HATP2
HAT-P-20 : HATP20
HAT-P-22 : HATP22
HAT-P-23 : HATP23
HAT-P-26 : HATP26
HAT-P-3 : HATP3
HAT-P-30 : HATP30
HAT-P-32 : HATP32
HAT-P-33 : HATP33
HAT-P-34 : HATP34
HAT-P-38 : HATP38
HAT-P-4 : HATP4
HAT-P-40 : HATP40
HAT-P-41 : HATP41
HAT-P-5 : HATP5
HAT-P-6 : HATP6
HAT-P-7 : HATP7
HAT-P-8 : HATP8
HATS-28 : HATS28
HATS-3 : HATS3
HATS-65 : HATS65
HATS-7 : HATS7
HD 106315 : HD106315
HD 149026 : HD149026
HD 17156 : HD17156
HD 189733 : HD189733
HD 209458 : HD209458
HD 219134 : HD219134
HD 97658 : HD97658
K2-124 : K2124
K2-132 : K2132
K2-136 : K2136
K2-138 : K2138
K2-167 : K2167
K2-174 : K2174
K2-18 : K218
K2-19 : K219
K2-21 : K221
K2-212 : K2212
K2-22 : K222
K2-24 : K224
K2-25 : K225
K2-26 : K226
K2-28 : K228
K2-289 : K2289
K2-3 : K23
K2-31 : K231
K2-32 : K232
K2-33 : K233
K2-36 : K236
K2-52 : K252
K2-53 : K253
K2-55 : K255
K2-58 : K258
K2-79 : K279
K2-87 : K287
K2-9 : K29
K2-90 : K290
K2-93 : K293
K2-95 : K295
K2-96 : K296
K2-97 : K297
KELT-1 : KELT1
KELT-11 : KELT11
KELT-14 : KELT14
KELT-16 : KELT16
KELT-20 : KELT20
KELT-3 : KELT3
KELT-7 : KELT7
KELT-9 : KELT9
Kepler-11 : KEPLER11
Kepler-13 : KEPLER13
Kepler-138 : KEPLER138
Kepler-16 : KEPLER16
Kepler-1625 : KEPLER1625
Kepler-51 : KEPLER51
Kepler-9 : KEPLER9
LHS 3844 : LHS3844
OGLE-TR-056 : OGLETR056
OGLE-TR-10 : OGLETR10
Qatar-1 : QATAR1
Qatar-2 : QATAR2
TRAPPIST-1 : TRAPPIST1
TrES-1 : TRES1
TrES-2 : TRES2
TrES-3 : TRES3
TrES-4 : TRES4
WASP-1 : WASP1
WASP-10 : WASP10
WASP-100 : WASP100
WASP-101 : WASP101
WASP-103 : WASP103
WASP-104 : WASP104
WASP-107 : WASP107
WASP-11 : WASP11
WASP-12 : WASP12
WASP-121 : WASP121
WASP-127 : WASP127
WASP-13 : WASP13
WASP-131 : WASP131
WASP-14 : WASP14
WASP-140 : WASP140
WASP-15 : WASP15
WASP-16 : WASP16
WASP-17 : WASP17
WASP-18 : WASP18
WASP-19 : WASP19
WASP-2 : WASP2
WASP-21 : WASP21
WASP-24 : WASP24
WASP-26 : WASP26
WASP-28 : WASP28
WASP-29 : WASP29
WASP-3 : WASP3
WASP-31 : WASP31
WASP-32 : WASP32
WASP-33 : WASP33
WASP-34 : WASP34
WASP-35 : WASP35
WASP-36 : WASP36
WASP-37 : WASP37
WASP-38 : WASP38
WASP-39 : WASP39
WASP-4 : WASP4
WASP-43 : WASP43
WASP-46 : WASP46
WASP-48 : WASP48
WASP-49 : WASP49
WASP-5 : WASP5
WASP-50 : WASP50
WASP-52 : WASP52
WASP-6 : WASP6
WASP-62 : WASP62
WASP-63 : WASP63
WASP-64 : WASP64
WASP-65 : WASP65
WASP-67 : WASP67
WASP-69 : WASP69
WASP-7 : WASP7
WASP-72 : WASP72
WASP-74 : WASP74
WASP-75 : WASP75
WASP-76 : WASP76
WASP-78 : WASP78
WASP-79 : WASP79
WASP-8 : WASP8
WASP-80 : WASP80
WASP-95 : WASP95
WASP-96 : WASP96
WASP-97 : WASP97
XO-1 : XO1
XO-2 : XO2
XO-3 : XO3
XO-4 : XO4
XO-5 : XO5
    '''
    return
# -------------------- -----------------------------------------------
# -- ACTIVE FILTERS -- -----------------------------------------------
# OBSERVATORY/INSTRUMENT/DETECTOR/FILTER/MODE
# HST-STIS-FUV.MAMA-G140M-STARE
# Spitzer-IRAC-IR-36-SUB
# Spitzer-IRAC-IR-45-SUB
# HST-WFC3-IR-G141-SCAN
# HST-WFC3-IR-G102-SCAN
# HST-STIS-CCD-G750L-STARE

def activefilters():
    '''
Spitzer-IRAC-IR-36-SUB
Spitzer-IRAC-IR-45-SUB
HST-WFC3-IR-G141-SCAN
HST-WFC3-IR-G102-SCAN
HST-STIS-CCD-G750L-STARE
HST-STIS-CCD-G430L-STARE
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
    overwrite['GJ 3053'] = {
        'b':{'t0':2456915.70654,
             't0_uperr':0.00004, 't0_lowerr':-0.00004,
             't0_ref':'GMR'}}
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
    overwrite['HAT-P-1'] = {
        'b':{'t0':2453979.92802,
             't0_uperr':0.00004, 't0_lowerr':-0.00004,
             't0_ref':'GMR'}}
    overwrite['HAT-P-11'] = {
        'R*':0.683, 'R*_uperr':0.009, 'R*_lowerr':-0.009,
        'R*_ref':'Yee et al. 2018',
        'b':{'inc':90,
             'inc_uperr':0.114, 'inc_lowerr':-0.114,
             'inc_ref':'Yee et al. 2018',
             't0':2454957.8132067,
             't0_uperr':0.0000663, 't0_lowerr':-0.0000663,
             't0_ref':'Yee et al. 2018',
             'rp':0.3889738152520562,
             'rp_uperr':0.0017, 'rp_lowerr':-0.0017,
             'rp_ref':'Yee et al. 2018',
             'sma':0.05254,
             'sma_uperr':0.00064, 'sma_lowerr':-0.00064,
             'sma_ref':'Yee et al. 2018',
             'ecc':0,
             'ecc_uperr':0.034, 'ecc_lowerr':-0.034,
             'ecc_ref':'Yee et al. 2018',
             'period':4.887802443,
             'period_uperr':0.0000071, 'period_lowerr':-0.0000071,
             'period_ref':'Yee et al. 2018'}}
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
             'period_ref':'Howard et al. 2012',
             'omega':200.5,
             'omega_lowerr':-1.3,
             'omega_uperr':1.3,
             'omega_ref':"Bonomo et al. 2017"}}
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
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Sato et al. 2012',
        'b':{'omega':240,
             'omega_lowerr':-104,
             'omega_uperr':104,
             'omega_ref':"Sato et al. 2012"}}
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
        'R*':1.18, 'R*_uperr':0.11, 'R*_lowerr':-0.11,
        'R*_ref':'Crossfield et al. 2017',
        'T*':6290, 'T*_uperr':60, 'T*_lowerr':-60,
        'T*_ref':'Crossfield et al. 2017',
        'FEH*':-0.24, 'FEH*_uperr':0.04, 'FEH*_lowerr':-0.04,
        'FEH*_ref':'Crossfield et al. 2017',
        'LOGG*':4.29, 'LOGG*_uperr':0.07, 'LOGG*_lowerr':-0.07,
        'LOGG*_ref':'Crossfield et al. 2017',
        'b':{'inc':88.4,
             'inc_uperr':1.1, 'inc_lowerr':-2.1,
             'inc_ref':'Crossfield et al. 2017',
             't0':2457605.6521,
             't0_uperr':0.0042, 't0_lowerr':-0.0045,
             't0_ref':'Crossfield et al. 2017',
             'sma':0.09012,
             'sma_uperr':0.00083, 'sma_lowerr':-0.00085,
             'sma_ref':'Crossfield et al. 2017',
             'ecc':0,
             'ecc_ref':'Crossfield et al. 2017',
             'rp':0.199,
             'rp_ref':'Crossfield et al. 2017',
             'period':9.5521,
             'period_ref':'Crossfield et al. 2017',
             'logg':2.51,
             'logg_uperr':0.26, 'logg_lowerr':-0.26,
             'logg_ref':'Crossfield et al. 2017', 'logg_units':'log10[cm.s-2]'},
        'c':{'inc':89.42,
             'inc_uperr':0.4, 'inc_lowerr':-0.67,
             'inc_ref':'Crossfield et al. 2017',
             't0':2457611.1310,
             't0_uperr':0.0012, 't0_lowerr':-0.0012,
             't0_ref':'Crossfield et al. 2017',
             'sma':0.1526,
             'sma_uperr':0.0014, 'sma_lowerr':-0.0014,
             'sma_ref':'Crossfield et al. 2017',
             'ecc':0,
             'ecc_ref':'Crossfield et al. 2017',
             'rp':0.352,
             'rp_ref':'Crossfield et al. 2017',
             'period':21.0576,
             'period_ref':'Crossfield et al. 2017',
             'logg':2.98,
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
        'R*':1.4, 'R*_uperr':0.19, 'R*_lowerr':-0.19,
        'R*_ref':'Vanderburg et al. 2016',
        'T*':6199, 'T*_uperr':50, 'T*_lowerr':-50,
        'T*_ref':'Vanderburg et al. 2016',
        'FEH*':-0.11, 'FEH*_uperr':0.08, 'FEH*_lowerr':-0.08,
        'FEH*_ref':'Vanderburg et al. 2016',
        'LOGG*':4.18, 'LOGG*_uperr':0.1, 'LOGG*_lowerr':-0.1,
        'LOGG*_ref':'Vanderburg et al. 2016',
        'b':{'inc':88.4,
             'inc_uperr':1.6, 'inc_lowerr':-1.6,
             'inc_ref':'Vanderburg et al. 2016',
             't0':2457152.2844,
             't0_uperr':0.0021, 't0_lowerr':-0.0021,
             't0_ref':'Vanderburg et al. 2016',
             'sma':0.12695775622426692,
             'sma_uperr':0.0029, 'sma_lowerr':-0.0029,
             'sma_ref':'Vanderburg et al. 2016',
             'ecc':0,
             'ecc_ref':'Vanderburg et al. 2016',
             'rp':0.2561240978011526,
             'rp_ref':'Vanderburg et al. 2016',
             'period':15.5712,
             'period_ref':'Vanderburg et al. 2016',
             'logg':3.0,
             'logg_lowerr':0.3, 'logg_uperr':-0.3,
             'logg_ref':'Unknown', 'logg_units':'log10[cm.s-2]'}}
    overwrite['K2-96'] = {
        'R*':0.872, 'R*_uperr':0.057, 'R*_lowerr':-0.057,
        'R*_ref':'Christiansen et al. 2017',
        'T*':5261, 'T*_uperr':60, 'T*_lowerr':-60,
        'T*_ref':'Christiansen et al. 2017',
        'FEH*':0.04, 'FEH*_uperr':0.05, 'FEH*_lowerr':-0.05,
        'FEH*_ref':'Christiansen et al. 2017',
        'LOGG*':4.47, 'LOGG*_uperr':0.05, 'LOGG*_lowerr':-0.05,
        'LOGG*_ref':'Christiansen et al. 2017',
        'b':{'inc':83.4,
             'inc_uperr':4.6, 'inc_lowerr':-7.7,
             'inc_ref':'Christiansen et al. 2017',
             't0':2457394.37454,
             't0_uperr':0.00043, 't0_lowerr':-0.00043,
             't0_ref':'Christiansen et al. 2017',
             'sma':0.01815,
             'sma_uperr':0.00023, 'sma_lowerr':-0.00023,
             'sma_ref':'Christiansen et al. 2017',
             'ecc':0,
             'ecc_ref':'Christiansen et al. 2017',
             'rp':0.152,
             'rp_ref':'Christiansen et al. 2017',
             'period':0.959641,
             'period_ref':'Christiansen et al. 2017',
             'logg':3.22900042686,
             'logg_lowerr':0.3, 'logg_uperr':-0.3,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'},
        'c':{'inc':89.30,
             'inc_uperr':0.5, 'inc_lowerr':-0.96,
             'inc_ref':'Christiansen et al. 2017',
             't0':2457394.9788,
             't0_uperr':0.0012, 't0_lowerr':-0.0012,
             't0_ref':'Christiansen et al. 2017',
             'sma':0.1795,
             'sma_uperr':0.0023, 'sma_lowerr':-0.0023,
             'sma_ref':'Christiansen et al. 2017',
             'ecc':0,
             'ecc_ref':'Christiansen et al. 2017',
             'rp':0.269,
             'rp_ref':'Christiansen et al. 2017',
             'period':29.8454,
             'period_ref':'Christiansen et al. 2017',
             'logg':3.02377443746,
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
        'b':{'logg':3.0132,
             'logg_uperr':0.15, 'logg_lowerr':-0.15,
             'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]',
             'sma':0.898,
             'sma_uperr':0.1,
             'sma_lowerr':-0.1,
             'sma_ref':'Morton et al. 2016'}}
    overwrite['WASP-12'] = {
        'R*':1.59, 'R*_uperr':0.18, 'R*_lowerr':-0.18,
        'R*_ref':'Stassun et al. 2017',
        'T*':6300, 'T*_uperr':150, 'T*_lowerr':-150,
        'T*_ref':'Stassun et al. 2017',
        'FEH*':0.3, 'FEH*_uperr':0.14, 'FEH*_lowerr':-0.17,
        'FEH*_ref':'Stassun et al. 2017',
        'LOGG*':4.38, 'LOGG*_uperr':0.1, 'LOGG*_lowerr':-0.1,
        'LOGG*_ref':'Stassun et al. 2017',
        'b':{'inc':82.5,
             'inc_uperr':0.75, 'inc_lowerr':-0.75,
             'inc_ref':'Stassun et al. 2017',
             't0':2456176.6683,
             't0_uperr':0.000078, 't0_lowerr':-0.000078,
             't0_ref':'Collins et al. 2017',
             'sma':0.0234,
             'sma_uperr':0.00056, 'sma_lowerr':-0.0005,
             'sma_ref':'Collins et al. 2017',
             'period':1.09142245,
             'period_ref':'Stassun et al. 2017',
             'omega':272.7,
             'omega_lowerr':-1.3,
             'omega_uperr':2.4,
             'omega_ref':"Knutson et al. 2014"}}
    overwrite['WASP-39'] = {
        'FEH*':-0.10, 'FEH*_uperr':0.1, 'FEH*_lowerr':-0.1,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Faedi et al. 2011'}
    overwrite['WASP-43'] = {
        'FEH*':-0.05, 'FEH*_uperr':0.17, 'FEH*_lowerr':-0.17,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Hellier et al. 2011'}
    overwrite['WASP-6'] = {
        'b':{'inc':88.38,
             'inc_uperr':0.31, 'inc_lowerr':-0.31,
             'inc_ref':'Tregloan-Reed et al. 2015',
             't0':2456132.41081,
             't0_uperr':0.00015, 't0_lowerr':-0.00015,
             't0_ref':'Tregloan-Reed et al. 2015',
             'sma':0.0414,
             'sma_uperr':0.001,
             'sma_lowerr':-0.001,
             'sma_ref':'Tregloan-Reed et al. 2015',
             'period':3.361,
             'period_ref':'Tregloan-Reed et al. 2015'}}
    overwrite['XO-2'] = {
        'b':{'t0':2455565.54048,
             't0_uperr':0.005, 't0_lowerr':-0.005,
             't0_ref':'GMR'}}
    overwrite['XO-3'] = {
        'b':{
            'inc': 84.20,
            'inc_uperr':0.54,
            'inc_lowerr':-0.54,
            'inc_ref':'Bonomo et al. 2017',
            'omega':347,
            'omega_lowerr':-3,
            'omega_uperr':3,
            'omega_ref':'Wong et al. 2014',
            'ecc':0.29,
            'ecc_lowerr':-0.01,
            'ecc_uperr':0.01,
        }
    }
    overwrite['HAT-P-19'] = {
        'b':{
            'omega':256,
            'omega_lowerr':-77,
            'omega_uperr':77,
            'omega_ref':"Hartman et al. 2011"
        }
    }
    overwrite['HAT-P-15'] = {
        'b':{
            'omega':262.5,
            'omega_lowerr':-2.9,
            'omega_uperr':2.4,
            'omega_ref':"Bonomo et al. 2017"
        }
    }
    overwrite['HAT-P-16'] = {
        'b':{
            'omega':213.0,
            'omega_lowerr':-5.5,
            'omega_uperr':5.1,
            'omega_ref':"Bonomo et al. 2017"
        }
    }
    overwrite['HAT-P-18'] = {
        'b':{
            'omega':104.0,
            'omega_lowerr':-50,
            'omega_uperr':50,
            'omega_ref':"Esposito et al. 2014"
        }
    }
    overwrite['HAT-P-2'] = {
        'b':{
            'omega':188.0,
            'omega_lowerr':-0.2,
            'omega_uperr':0.2,
            'omega_ref':"Ment et al. 2018"
        }
    }
    overwrite['HAT-P-23'] = {
        'b':{
            'omega':121.0,
            'omega_lowerr':-9,
            'omega_uperr':11,
            'omega_ref':"Stassun et al. 2017"
        }
    },
    overwrite['HAT-P-26'] = {
        'b':{
            'omega':46.0,
            'omega_lowerr':-71,
            'omega_uperr':33,
            'omega_ref':"Knutson et al. 2014"
        }
    }
    overwrite['HAT-P-30'] = {
        'b':{
            'omega':114.0,
            'omega_lowerr':-77,
            'omega_uperr':200,
            'omega_ref':"Knutson et al. 2014"
        }
    }
    overwrite['HAT-P-32'] = {
        'b':{
            'omega':50.0,
            'omega_lowerr':-18,
            'omega_uperr':27,
            'omega_ref':"Wang et al. 2019"
        }
    }
    overwrite['HAT-P-34'] = {
        'b':{
            'omega':21.9,
            'omega_lowerr':-5.6,
            'omega_uperr':5.2,
            'omega_ref':"Bonomo et al. 2017"
        }
    }
    overwrite['HD 17156'] = {
        'b':{
            'omega':121.51,
            'omega_lowerr':-0.3,
            'omega_uperr':0.3,
            'omega_ref':"Ment et al. 2018"
        }
    }
    overwrite['K2-132'] = {
        'b':{
            'omega':82.6,
            'omega_lowerr':-4.2,
            'omega_uperr':4.0,
            'omega_ref':"Jones et al. 2018"
        }
    }
    overwrite['K2-18'] = {
        'b':{
            'omega':-5.7,
            'omega_lowerr':-33.8,
            'omega_uperr':46.40,
            'omega_ref':"Sarkis et al. 2018"
        }
    }
    overwrite['K2-22'] = {
        'b':{
            'omega':46.7,
            'omega_lowerr':-144,
            'omega_uperr':90,
            'omega_ref':"Dressing et al. 2017"
        }
    }
    overwrite['WASP-10'] = {
        'b':{
            'omega':151.9,
            'omega_lowerr':-8.8,
            'omega_uperr':11,
            'omega_ref':"Bonomo et al. 2017"
        }
    }
    overwrite['WASP-14'] = {
        'b':{
            'omega':251.61,
            'omega_lowerr':-0.41,
            'omega_uperr':0.41,
            'omega_ref':"Bonomo et al. 2017"
        }
    }
    overwrite['WASP-34'] = {
        'b':{
            'omega':215,
            'omega_lowerr':-150,
            'omega_uperr':77,
            'omega_ref':"Knutson et al. 2014"
        }
    }
    overwrite['WASP-5'] = {
        'b':{
            'omega':34.38,
            'omega_lowerr':-22,
            'omega_uperr':-26.93,
            'omega_ref':"Gillon et al. 2009"
        }
    }
    overwrite['WASP-7'] = {
        'b':{
            'omega':109,
            'omega_lowerr':-55,
            'omega_uperr':170,
            'omega_ref':"Knutson et al. 2014"
        }
    }
    overwrite['WASP-8'] = {
        'b':{
            'omega':274.21,
            'omega_lowerr':-0.33,
            'omega_uperr':0.33,
            'omega_ref':"Bonomo et al. 2017"
        }
    }
    overwrite['Kepler-11'] = {
        'g':{
            'omega':97.0,
            'omega_lowerr':-30,
            'omega_uperr':30,
            'omega_ref':"Borsato et al. 2014"
        },
        'b':{
            'omega':71.46,
            'omega_lowerr':-17,
            'omega_uperr':17,
            'omega_ref':"Borsato et al. 2014"
        }
    }
    overwrite['Kepler-9'] = {
        'b':{
            'omega':357.0,
            'omega_lowerr':-0.4,
            'omega_uperr':0.5,
            'omega_ref':"Borsato et al. 2019"
        },
        'c':{
            'omega':167.5,
            'omega_lowerr':-0.1,
            'omega_uperr':0.1,
            'omega_ref':"Borsato et al. 2019"
        }
    }
    overwrite['LHS 3844'] = {
        'FEH*':0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':"Kyle's best guess",
        'b':{
            'mass':0.0118,
            'mass_uperr':0.0051,
            'mass_lowerr':-0.0036,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.323,
            'logg_lowerr':-0.15, 'logg_uperr':0.16,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-136'] = {
        'b':{
            'mass':0.0105,
            'mass_uperr':0.0045,
            'mass_lowerr':-0.0032,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.513,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.0272,
            'mass_uperr':0.0116,
            'mass_lowerr':-0.0081,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9791,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]',
            'omega':24.6,
            'omega_lowerr':-74.0,
            'omega_uperr':141.0,
            'omega_ref':"Mann et al. 2017"
        },
        'd':{
            'mass':0.0130,
            'mass_uperr':0.0056,
            'mass_lowerr':-0.0039,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.2749,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-25'] = {
        'b':{
            'mass':0.0335,
            'mass_uperr':0.0143,
            'mass_lowerr':-0.0100,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.948,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]',
            'sma':0.029535117574370662,
            'sma_uperr':0.0021173577439160705,
            'sma_lowerr':-0.0021173577439160705,
            'sma_ref':'Mann et al. 2016',
            'omega':62,
            'omega_lowerr':-39,
            'omega_uperr':44,
            'omega_ref':"Mann et al. 2016"
        }
    }
    overwrite['LHS 3844'] = {
        'FEH*':0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':"Kyle's best guess",
        'b':{
            'mass':0.0118,
            'mass_uperr':0.0051,
            'mass_lowerr':-0.0036,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.323,
            'logg_lowerr':-0.15, 'logg_uperr':0.16,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-124'] = {
        'b':{
            'mass':0.0259,
            'mass_uperr':0.0110,
            'mass_lowerr':-0.0077,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9791,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-138'] = {
        'b':{
            'mass':0.0135,
            'mass_uperr':0.0058,
            'mass_lowerr':-0.0040,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.2305,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.0213,
            'mass_uperr':0.0091,
            'mass_lowerr':-0.0064,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0195,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'd':{
            'mass':0.0229,
            'mass_uperr':0.0097,
            'mass_lowerr':-0.0068,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0025,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'e':{
            'mass':0.0313,
            'mass_uperr':0.0133,
            'mass_lowerr':-0.0093,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9538,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'f':{
            'mass':0.0246,
            'mass_uperr':0.0105,
            'mass_lowerr':-0.0074,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9871,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-167'] = {
        'b':{
            'mass':0.0246,
            'mass_uperr':0.0105,
            'mass_lowerr':-0.0074,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9871,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-19'] = {
        'b':{
            'mass':0.2554,
            'mass_uperr':0.109,
            'mass_lowerr':-0.0764,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.1228,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]',
            'omega':179,
            'omega_lowerr':-52,
            'omega_uperr':52,
            'omega_ref':"Barros et al. 2015"
        },
        'c':{
            'mass':0.0683,
            'mass_uperr':0.0292,
            'mass_lowerr':-0.0204,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9542,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'd':{
            'mass':0.011,
            'mass_uperr':0.0047,
            'mass_lowerr':-0.0033,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.4207,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-21'] = {
        'b':{
            'mass':0.0153,
            'mass_uperr':0.0065,
            'mass_lowerr':-0.0046,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.1488,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]',
            'omega':34.38,
            'omega_lowerr':-133,
            'omega_uperr':101,
            'omega_ref':"Dressing et al. 2017"
        },
        'c':{
            'mass':0.0210,
            'mass_uperr':0.0089,
            'mass_lowerr':-0.0063,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0235,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]',
            'omega':59.96,
            'omega_lowerr':-120,
            'omega_uperr':75,
            'omega_ref':"Dressing et al. 2017"
        }
    }
    overwrite['K2-212'] = {
        'b':{
            'mass':0.0213,
            'mass_uperr':0.0091,
            'mass_lowerr':-0.0064,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0195,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-26'] = {
        'b':{
            'mass':0.0230,
            'mass_uperr':0.0098,
            'mass_lowerr':-0.0069,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0014,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-28'] = {
        'b':{
            'mass':0.0193,
            'mass_uperr':0.0082,
            'mass_lowerr':-0.0058,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0487,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-289'] = {
        'b':{
            'mass':0.4283,
            'mass_uperr':0.1824,
            'mass_lowerr':-0.1279,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.2067,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-3'] = {
        'b':{
            'mass':0.0181,
            'mass_uperr':0.0077,
            'mass_lowerr':-0.0054,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0730,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.0154,
            'mass_uperr':0.0066,
            'mass_lowerr':-0.0046,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.1462,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'd':{
            'mass':0.0154,
            'mass_uperr':0.0066,
            'mass_lowerr':-0.0046,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.1462,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
    }
    overwrite['K2-32'] = {
        'b':{
            'mass':0.0717,
            'mass_uperr':0.0306,
            'mass_lowerr':-0.0215,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9578,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.0238,
            'mass_uperr':0.0101,
            'mass_lowerr':-0.0071,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9940,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'd':{
            'mass':0.0310,
            'mass_uperr':0.0132,
            'mass_lowerr':-0.0092,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9548,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
    }
    overwrite['K2-33'] = {
        'FEH*':0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':"Kyle's best guess"}
    overwrite['K2-52'] = {
        'b':{
            'mass':1.0821,
            'mass_uperr':0.4619,
            'mass_lowerr':-0.3237,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0168,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-52'] = {
        'b':{
            'mass':0.0222,
            'mass_uperr':0.0094,
            'mass_lowerr':-0.0066,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0095,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-55'] = {
        'FEH*':0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':"Kyle's best guess",
        'b':{
            'mass':0.0539,
            'mass_uperr':0.0230,
            'mass_lowerr':-0.0161,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9414,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-58'] = {
        'b':{
            'mass':0.0221,
            'mass_uperr':0.0094,
            'mass_lowerr':-0.0066,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0107,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.0138,
            'mass_uperr':0.0059,
            'mass_lowerr':-0.0041,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.2137,
            'logg_lowerr':-0.15, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'd':{
            'mass':0.0144,
            'mass_uperr':0.0061,
            'mass_lowerr':-0.0043,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.1854,
            'logg_lowerr':-0.1540, 'logg_uperr':0.15,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-79'] = {
        'b':{
            'mass':0.0382,
            'mass_uperr':0.0163,
            'mass_lowerr':-0.0114,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9409,
            'logg_lowerr':-0.1541, 'logg_uperr':0.1541,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-87'] = {
        'b':{
            'mass':0.2410,
            'mass_uperr':0.1029,
            'mass_lowerr':-0.0721,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.1135,
            'logg_lowerr':-0.1544, 'logg_uperr':0.1544,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-9'] = {
        'b':{
            'mass':0.0187,
            'mass_uperr':0.008,
            'mass_lowerr':-0.0056,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0604,
            'logg_lowerr':-0.1540, 'logg_uperr':0.1540,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-90'] = {
        'b':{
            'mass':0.0222,
            'mass_uperr':0.0094,
            'mass_lowerr':-0.0066,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0095,
            'logg_lowerr':-0.1539, 'logg_uperr':0.1539,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.0119,
            'mass_uperr':0.0051,
            'mass_lowerr':-0.0036,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.3391,
            'logg_lowerr':-0.1548, 'logg_uperr':0.1548,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]',
            'omega':56.7,
            'omega_lowerr':-129,
            'omega_uperr':79.9,
            'omega_ref':"Dressing et al. 2017"
        }
    }
    overwrite['K2-93'] = {
        'b':{
            'mass':0.0258,
            'mass_uperr':0.0110,
            'mass_lowerr':-0.0077,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9791,
            'logg_lowerr':-0.1539, 'logg_uperr':0.1539,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-95'] = {
        'b':{
            'mass':0.0420,
            'mass_uperr':0.0179,
            'mass_lowerr':-0.0125,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9385,
            'logg_lowerr':-0.1541, 'logg_uperr':0.1541,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-96'] = {
        'b':{
            'mass':0.0143,
            'mass_uperr':0.0061,
            'mass_lowerr':-0.0043,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.1884,
            'logg_lowerr':-0.1543, 'logg_uperr':0.1543,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.0252,
            'mass_uperr':0.0107,
            'mass_lowerr':-0.0075,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':2.9825,
            'logg_lowerr':-0.1539, 'logg_uperr':0.1539,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['K2-97'] = {
        'b':{
            'mass':0.6808,
            'mass_uperr':0.2892,
            'mass_lowerr':-0.2030,
            'mass_ref':'Pearson 2019',
            'mass_units':'Jupiter mass',
            'logg':3.2746,
            'logg_lowerr':-0.1538, 'logg_uperr':0.1538,
            'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        }
    }
    overwrite['Kepler-9'] = {
        'b':{
            'mass':0.137,
            'mass_uperr':0.008,
            'mass_lowerr':-0.01,
            'mass_ref':'Hadden & Lithwick et al. 2017',
            'mass_units':'Jupiter mass',
        }
    }

    overwrite["CoRoT-2"]= {"b":{"Spitzer-IRAC-IR-45-SUB": {
        "rprs": 0.15417,
        "ars": 6.60677,
        "inc": 88.08,
        "ref": "KAP"
    }}}

    overwrite['GJ 1132'] = {
        'b':{'inc':88.41,
             'inc_uperr':2.6, 'inc_lowerr':-2.7,
             'inc_ref':'KAP'
            }
    }
#     overwrite["GJ 1132"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.03591,
#         "ars": 15.67466,
#         "inc": 87.48,
#         "ref": "KAP"
#     }
#     overwrite["GJ 9827"]["c"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.01535,
#         "ars": 14.81933,
#         "inc": 89.07,
#         "ref": "KAP"
#     }
#     overwrite["GJ 9827"]["d"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.02939,
#         "ars": 21.09813,
#         "inc": 87.5,
#         "ref": "KAP"
#     }
    overwrite["HAT-P-11"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.05472,
        "ars": 16.68736,
        "inc": 90.0,
        "ref": "KAP"
    }
    overwrite["HAT-P-11"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.05737,
        "ars": 16.69862,
        "inc": 90.0,
        "ref": "KAP"
    }
    overwrite["HAT-P-12"] = {"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.13596,
        "ars": 11.70552,
        "inc": 89.22,
        "ref": "KAP"
    }}}
    overwrite["HAT-P-12"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.12542,
        "ars": 11.79633,
        "inc": 89.22,
        "ref": "KAP"
    }
    overwrite["HAT-P-18"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.13101,
        "ars": 15.91216,
        "inc": 89.14,
        "ref": "KAP"
    }
    overwrite["HAT-P-18"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.1341,
        "ars": 16.03188,
        "inc": 89.14,
        "ref": "KAP"
    }
#     overwrite["HAT-P-23"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
#         "rprs": 0.10812,
#         "ars": 5.08796,
#         "inc": 85.1,
#         "ref": "KAP"
#     }
#     overwrite["HAT-P-23"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.10907,
#         "ars": 5.11757,
#         "inc": 85.1,
#         "ref": "KAP"
#    }
    overwrite["HAT-P-26"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.0627,
        "ars": 11.88456,
        "inc": 87.97,
        "ref": "KAP"
    }
    overwrite["HAT-P-26"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.06869,
        "ars": 11.83711,
        "inc": 87.97,
        "ref": "KAP"
    }
    overwrite["HAT-P-32"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.14229,
        "ars": 5.30456,
        "inc": 89.96,
        "ref": "KAP"
    }
    overwrite["HAT-P-32"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.14319,
        "ars": 5.28352,
        "inc": 89.96,
        "ref": "KAP"
    }
    overwrite["HAT-P-41"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.09061,
        "ars": 5.45354,
        "inc": 89.71,
        "ref": "KAP"
    }
    overwrite["HAT-P-41"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.09406,
        "ars": 5.46372,
        "inc": 89.71,
        "ref": "KAP"
    }
    overwrite["HAT-P-7"]= {"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.07729,
        "ars": 4.00235,
        "inc": 83.11,
        "ref": "KAP"
    }}}
    overwrite["HAT-P-7"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.07431,
        "ars": 4.00443,
        "inc": 83.11,
        "ref": "KAP"
    }
#     overwrite["HAT-P-8"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.08855,
#         "ars": 6.15786,
#         "inc": 87.8,
#         "ref": "KAP"
#     }
#     overwrite["HD 106315"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.0,
#         "ars": 16.42263,
#         "inc": 88.4,
#         "ref": "KAP"
#     }
#     overwrite["HD 106315"]["c"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.0,
#         "ars": 27.8084,
#         "inc": 88.13,
#         "ref": "KAP"
#     }
    overwrite["HD 149026"]= {"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.04715,
        "ars": 6.7022,
        "inc": 86.67,
        "ref": "KAP"
    }}}
    overwrite["HD 149026"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.0508,
        "ars": 6.69519,
        "inc": 86.67,
        "ref": "KAP"
    }
#     overwrite["HD 209458"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
#         "rprs": 0.08969,
#         "ars": 8.50999,
#         "inc": 86.33,
#         "ref": "KAP"
#     }
#     overwrite["HD 209458"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.09393,
#         "ars": 8.49389,
#         "inc": 86.33,
#         "ref": "KAP"
#     }
    overwrite["K2-132"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.02366,
        "ars": 5.10981,
        "inc": 77.46,
        "ref": "KAP"
    }
#     overwrite["K2-136"]["c"]["Spitzer-IRAC-IR-36-SUB"] = {
#         "rprs": 0.05995,
#         "ars": 37.51501,
#         "inc": 89.6,
#         "ref": "KAP"
#     }
#     overwrite["K2-136"]["c"]["Spitzer-IRAC-IR-45-SUB"] = {
#         "rprs": 0.02834,
#         "ars": 38.19971,
#         "inc": 89.6,
#         "ref": "KAP"
#     }
    overwrite["K2-212"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.02706,
        "ars": 26.01843,
        "inc": 89.19,
        "ref": "KAP"
    }
    overwrite["K2-24"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.0315,
        "ars": 28.52787,
        "inc": 89.08,
        "ref": "KAP"
    }
#     overwrite["K2-25"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
#         "rprs": NaN,
#         "ars": NaN,
#         "inc": NaN,
#         "ref": "KAP"
#     }
    overwrite["K2-25"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.105559,
        "ars": 21.82933,
        "inc": 88.13,
        "ref": "KAP"
    }
    overwrite["K2-28"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.05347,
        "ars": 15.843,
        "inc": 88.9,
        "ref": "KAP"
    }
    overwrite["K2-33"]["b"]= {"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.032,
        "ars": 8.38191,
        "inc": 83.72,
        "ref": "KAP"
    }}
    overwrite["K2-33"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.03212,
        "ars": 8.38039,
        "inc": 83.72,
        "ref": "KAP"
    }
    overwrite["KELT-1"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.07319,
        "ars": 3.6293,
        "inc": 84.95,
        "ref": "KAP"
    }
    overwrite["KELT-1"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.06731,
        "ars": 3.63812,
        "inc": 84.95,
        "ref": "KAP"
    }
    overwrite["KELT-14"] = {"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.11846,
        "ars": 4.28352,
        "inc": 78.3,
        "ref": "KAP"
    }}}
    overwrite["KELT-14"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.11612,
        "ars": 4.24534,
        "inc": 78.3,
        "ref": "KAP"
    }
    overwrite["KELT-16"]= {"b":{"Spitzer-IRAC-IR-45-SUB":{
        "rprs": 0.11004,
        "ars": 3.239,
        "inc": 84.4,
        "ref": "KAP"
    }}}
    overwrite["KELT-20"]= {"b":{"Spitzer-IRAC-IR-45-SUB":{
        "rprs": 0.11524,
        "ars": 7.46091,
        "inc": 86.12,
        "ref": "KAP"
    }}}
    overwrite["KELT-7"]={"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.09075,
        "ars": 5.23451,
        "inc": 82.82,
        "ref": "KAP"
    }}}
    overwrite["KELT-7"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.08985,
        "ars": 5.23938,
        "inc": 82.82,
        "ref": "KAP"
    }
    overwrite["KELT-9"]={"b":{"Spitzer-IRAC-IR-45-SUB":{
        "rprs": 0.08249,
        "ars": 3.19869,
        "inc": 86.79,
        "ref": "KAP"
    }}}
    overwrite["LHS 3844"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.05646,
        "ars": 7.0373,
        "inc": 88.5,
        "ref": "KAP"
    }
    overwrite["Qatar-1"]={"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.14617,
        "ars": 6.27199,
        "inc": 84.08,
        "ref": "KAP"
    }}}
    overwrite["Qatar-1"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.14198,
        "ars": 6.3013,
        "inc": 84.08,
        "ref": "KAP"
    }
    overwrite["Qatar-2"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.16336,
        "ars": 5.8899,
        "inc": 86.12,
        "ref": "KAP"
    }}}
    overwrite["Qatar-2"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.16304,
        "ars": 5.94104,
        "inc": 86.12,
        "ref": "KAP"
    }
    overwrite["TrES-3"]={"b":{"Spitzer-IRAC-IR-45-SUB":{
        "rprs": 0.16639,
        "ars": 5.98068,
        "inc": 81.85,
        "ref": "KAP"
    }}}
    overwrite["WASP-1"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.09646,
        "ars": 5.71308,
        "inc": 90.0,
        "ref": "KAP"
    }}}
    overwrite["WASP-1"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.09964,
        "ars": 5.67935,
        "inc": 90.0,
        "ref": "KAP"
    }
    overwrite["WASP-101"]={"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.10685,
        "ars": 8.32318,
        "inc": 84.76,
        "ref": "KAP"
    }}}
    overwrite["WASP-101"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.10875,
        "ars": 8.30395,
        "inc": 84.76,
        "ref": "KAP"
    }
    overwrite["WASP-107"]={"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.13341,
        "ars": 18.10372,
        "inc": 89.93,
        "ref": "KAP"
    }}}
    overwrite["WASP-107"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.14157,
        "ars": 18.04266,
        "inc": 89.93,
        "ref": "KAP"
    }
    overwrite["WASP-121"]={"b":{"Spitzer-IRAC-IR-36-SUB":{
        "rprs": 0.11918,
        "ars": 3.75692,
        "inc": 89.93,
        "ref": "KAP"
    }}}
    overwrite["WASP-121"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.12097,
        "ars": 3.77292,
        "inc": 89.93,
        "ref": "KAP"
    }
    overwrite["WASP-127"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.09465,
        "ars": 8.06482,
        "inc": 88.7,
        "ref": "KAP"
    }}}
    overwrite["WASP-127"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.10356,
        "ars": 8.02415,
        "inc": 88.7,
        "ref": "KAP"
    }
    overwrite["WASP-13"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.08714,
        "ars": 8.05456,
        "inc": 85.43,
        "ref": "KAP"
    }}}
    overwrite["WASP-13"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.09283,
        "ars": 8.02884,
        "inc": 85.43,
        "ref": "KAP"
    }
    overwrite["WASP-131"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.07731,
        "ars": 8.43526,
        "inc": 85.0,
        "ref": "KAP"
    }}}
    overwrite["WASP-131"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.07629,
        "ars": 8.47147,
        "inc": 85.0,
        "ref": "KAP"
    }
    overwrite["WASP-140"]={"b":{"Spitzer-IRAC-IR-45-SUB": {
        "rprs": 0.15882,
        "ars": 8.00504,
        "inc": 83.3,
        "ref": "KAP"
    }}}
    overwrite["WASP-17"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.11219,
        "ars": 7.54489,
        "inc": 89.97,
        "ref": "KAP"
    }}}
    overwrite["WASP-17"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.11898,
        "ars": 7.46699,
        "inc": 89.97,
        "ref": "KAP"
    }
    overwrite["WASP-19"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.14216,
        "ars": 3.47724,
        "inc": 78.51,
        "ref": "KAP"
    }}}
    overwrite["WASP-19"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.14102,
        "ars": 3.49199,
        "inc": 78.51,
        "ref": "KAP"
    }
    overwrite["WASP-21"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.09412,
        "ars": 9.97247,
        "inc": 88.75,
        "ref": "KAP"
    }}}
    overwrite["WASP-21"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.04464,
        "ars": 9.72834,
        "inc": 88.75,
        "ref": "KAP"
    }
    overwrite["WASP-29"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.09419,
        "ars": 12.30495,
        "inc": 89.93,
        "ref": "KAP"
    }}}
    overwrite["WASP-29"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.09394,
        "ars": 12.31367,
        "inc": 89.93,
        "ref": "KAP"
    }
    overwrite["WASP-31"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.10103,
        "ars": 8.02383,
        "inc": 84.21,
        "ref": "KAP"
    }}}
    overwrite["WASP-31"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.11153,
        "ars": 8.01901,
        "inc": 84.21,
        "ref": "KAP"
    }
    overwrite["WASP-33"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.10682,
        "ars": 3.5413,
        "inc": 86.63,
        "ref": "KAP"
    }}}
    overwrite["WASP-33"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.1084,
        "ars": 3.6099,
        "inc": 86.63,
        "ref": "KAP"
    }
    overwrite["WASP-36"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.12992,
        "ars": 5.88029,
        "inc": 83.15,
        "ref": "KAP"
    }}}
    overwrite["WASP-36"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.13419,
        "ars": 5.84215,
        "inc": 83.15,
        "ref": "KAP"
    }
    overwrite["WASP-39"]["b"] ={"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.13816,
        "ars": 11.58054,
        "inc": 88.03,
        "ref": "KAP"
    }}
    overwrite["WASP-39"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.14286,
        "ars": 11.58493,
        "inc": 88.03,
        "ref": "KAP"
    }
    overwrite["WASP-4"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.13608,
        "ars": 5.48493,
        "inc": 89.06,
        "ref": "KAP"
    }}}
    overwrite["WASP-4"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.15113,
        "ars": 5.47592,
        "inc": 89.06,
        "ref": "KAP"
    }
    overwrite["WASP-43"]["b"] = {"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.15356,
        "ars": 5.26328,
        "inc": 84.25,
        "ref": "KAP"
    }}
    overwrite["WASP-43"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.15523,
        "ars": 5.26395,
        "inc": 84.25,
        "ref": "KAP"
    }
    overwrite["WASP-52"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.16534,
        "ars": 7.29716,
        "inc": 85.25,
        "ref": "KAP"
    }}}
    overwrite["WASP-52"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.16452,
        "ars": 7.31186,
        "inc": 85.25,
        "ref": "KAP"
    }
    overwrite["WASP-6"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.13768,
        "ars": 11.5668,
        "inc": 89.99,
        "ref": "KAP"
    }
    overwrite["WASP-6"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.13993,
        "ars": 11.77256,
        "inc": 89.99,
        "ref": "KAP"
    }
    overwrite["WASP-62"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.10694,
        "ars": 9.63148,
        "inc": 88.25,
        "ref": "KAP"
    }}}
    overwrite["WASP-62"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.08844,
        "ars": 9.84845,
        "inc": 88.25,
        "ref": "KAP"
    }
    overwrite["WASP-63"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.07479,
        "ars": 6.63717,
        "inc": 89.37,
        "ref": "KAP"
    }}}
    overwrite["WASP-63"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.07187,
        "ars": 6.65457,
        "inc": 89.37,
        "ref": "KAP"
    }
    overwrite["WASP-67"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.13746,
        "ars": 12.92629,
        "inc": 86.05,
        "ref": "KAP"
    }}}
    overwrite["WASP-67"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.13229,
        "ars": 12.91299,
        "inc": 86.05,
        "ref": "KAP"
    }
    overwrite["WASP-69"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.12707,
        "ars": 11.05987,
        "inc": 86.11,
        "ref": "KAP"
    }}}
    overwrite["WASP-69"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.12998,
        "ars": 11.06867,
        "inc": 86.11,
        "ref": "KAP"
    }
    overwrite["WASP-74"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.09019,
        "ars": 5.36976,
        "inc": 81.07,
        "ref": "KAP"
    }}}
    overwrite["WASP-74"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.09479,
        "ars": 5.31398,
        "inc": 81.07,
        "ref": "KAP"
    }
    overwrite["WASP-76"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.09439,
        "ars": 3.9565,
        "inc": 85.98,
        "ref": "KAP"
    }}}
    overwrite["WASP-76"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.10368,
        "ars": 3.99989,
        "inc": 85.98,
        "ref": "KAP"
    }
    overwrite["WASP-79"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.10751,
        "ars": 7.17676,
        "inc": 85.57,
        "ref": "KAP"
    }}}
    overwrite["WASP-79"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.10644,
        "ars": 7.2146,
        "inc": 85.57,
        "ref": "KAP"
    }
    overwrite["WASP-80"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.16864,
        "ars": 11.70746,
        "inc": 87.89,
        "ref": "KAP"
    }}}
    overwrite["WASP-80"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.17257,
        "ars": 11.76194,
        "inc": 87.89,
        "ref": "KAP"
    }
    overwrite["WASP-95"]={"b":{"Spitzer-IRAC-IR-45-SUB": {
        "rprs": 0.0997,
        "ars": 6.15376,
        "inc": 88.4,
        "ref": "KAP"
    }}}
    overwrite["XO-1"]={"b":{"Spitzer-IRAC-IR-36-SUB": {
        "rprs": 0.13111,
        "ars": 11.20137,
        "inc": 88.31,
        "ref": "KAP"
    }}}
    overwrite["XO-1"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.13096,
        "ars": 11.43554,
        "inc": 88.31,
        "ref": "KAP"
    }
    overwrite["XO-2"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.09526,
        "ars": 8.00034,
        "inc": 89.32,
        "ref": "KAP"
    }
    overwrite["XO-2"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.07278,
        "ars": 7.98249,
        "inc": 89.32,
        "ref": "KAP"
    }
    overwrite["XO-3"]["b"]["Spitzer-IRAC-IR-36-SUB"] = {
        "rprs": 0.08461,
        "ars": 6.3953,
        "inc": 84.13,
        "ref": "KAP"
    }
    overwrite["XO-3"]["b"]["Spitzer-IRAC-IR-45-SUB"] = {
        "rprs": 0.08647,
        "ars": 6.37613,
        "inc": 84.26,
        "ref": "KAP"
    }

    # spitzer orbit parameters
    return overwrite
# -------------------------------------------------------------------
