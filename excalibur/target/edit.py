'''target task inputs and parameter overwrite'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
# ------------- ------------------------------------------------------
# -- CREATE -- -------------------------------------------------------
def createversion():
    '''
    1.2.0: Added Spitzer targets
    1.3.0: Added JWST-NIRISS-NIS-CLEAR-G700XD
    1.4.0: Added more targets (242 total)
    1.5.0: New limb darkening coefficients for spitzer targest
    1.6.0: Added a few FEH* values and one Hmag; removed some redundant settings
    '''
    return dawgie.VERSION(1,6,0)
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
GJ 1252 :
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
HD 213885 :
HD 219134 :
HD 219666 :
HD 23472 :
HD 97658 :
HR 858 :
K2-124 :
K2-132 :
K2-136 :
K2-138 :
K2-141 :
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
Kepler-10 :
Kepler-102 :
Kepler-104 :
Kepler-1083 :
Kepler-12 :
Kepler-125 :
Kepler-126 :
Kepler-127 :
Kepler-1339 :
Kepler-138 :
Kepler-14 :
Kepler-1485 :
Kepler-1492 :
Kepler-156 :
Kepler-1568 :
Kepler-158 :
Kepler-16 :
Kepler-1625 :
Kepler-1651 :
Kepler-167 :
Kepler-17 :
Kepler-18 :
Kepler-19 :
Kepler-20 :
Kepler-205 :
Kepler-218 :
Kepler-236 :
Kepler-249 :
Kepler-25 :
Kepler-26 :
Kepler-293 :
Kepler-297 :
Kepler-309 :
Kepler-32 :
Kepler-37 :
Kepler-395 :
Kepler-45 :
Kepler-454 :
Kepler-48 :
Kepler-482 :
Kepler-49 :
Kepler-5 :
Kepler-504 :
Kepler-505 :
Kepler-570 :
Kepler-582 :
Kepler-598 :
Kepler-6 :
Kepler-603 :
Kepler-61 :
Kepler-62 :
Kepler-68 :
Kepler-7 :
Kepler-705 :
Kepler-737 :
Kepler-769 :
Kepler-786 :
Kepler-9 :
Kepler-93 :
Kepler-94 :
LHS 3844 :
OGLE-TR-056 :
OGLE-TR-10 :
Qatar-1 :
Qatar-2 :
TOI-270 :
TOI-700 :
TOI-849 :
TrES-1 :
TrES-2 :
TrES-3 :
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
WASP-77 : WASP-77 A
WASP-78 :
WASP-79 :
WASP-8 :
WASP-80 :
WASP-87 :
WASP-94 : WASP-94 A
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
GJ 1252 : GJ1252
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
HD 213885 : HD213885
HD 219134 : HD219134
HD 219666 : HD219666
HD 23472 : HD23472
HD 3167 : HD3167
HD 97658 : HD97658
HR 858 : HR858
K2-124 : K2124
K2-132 : K2132
K2-136 : K2136
K2-138 : K2138
K2-141 : K2141
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
Kepler-10 : KEPLER10
Kepler-102 : KEPLER102
Kepler-104 : KEPLER104
Kepler-1083 : KEPLER1083
Kepler-12 : KEPLER12
Kepler-125 : KEPLER125
Kepler-126 : KEPLER126
Kepler-127 : KEPLER127
Kepler-1339 : KEPLER1339
Kepler-138 : KEPLER138
Kepler-14 : KEPLER14
Kepler-1485 : KEPLER1485
Kepler-1492 : KEPLER1492
Kepler-156 : KEPLER156
Kepler-1568 : KEPLER1568
Kepler-158 : KEPLER158
Kepler-16 : KEPLER16
Kepler-1625 : KEPLER1625
Kepler-1651 : KEPLER1651
Kepler-167 : KEPLER167
Kepler-17 : KEPLER17
Kepler-18 : KEPLER18
Kepler-19 : KEPLER19
Kepler-20 : KEPLER20
Kepler-205 : KEPLER205
Kepler-218 : KEPLER218
Kepler-236 : KEPLER236
Kepler-249 : KEPLER249
Kepler-25 : KEPLER25
Kepler-26 : KEPLER26
Kepler-293 : KEPLER293
Kepler-297 : KEPLER297
Kepler-309 : KEPLER309
Kepler-32 : KEPLER32
Kepler-37 : KEPLER37
Kepler-395 : KEPLER395
Kepler-45 : KEPLER45
Kepler-454 : KEPLER454
Kepler-48 : KEPLER48
Kepler-482 : KEPLER482
Kepler-49 : KEPLER49
Kepler-5 : KEPLER5
Kepler-504 : KEPLER504
Kepler-505 : KEPLER505
Kepler-570 : KEPLER570
Kepler-582 : KEPLER582
Kepler-598 : KEPLER598
Kepler-6 : KEPLER6
Kepler-603 : KEPLER603
Kepler-61 : KEPLER61
Kepler-62 : KEPLER62
Kepler-68 : KEPLER68
Kepler-7 : KEPLER7
Kepler-705 : KEPLER705
Kepler-737 : KEPLER737
Kepler-769 : KEPLER769
Kepler-786 : KEPLER786
Kepler-9 : KEPLER9
Kepler-93 : KEPLER93
Kepler-94 : KEPLER94
LHS 1140 : LHS1140
LHS 3844 : LHS3844
OGLE-TR-056 : OGLETR056
OGLE-TR-10 : OGLETR10
Qatar-1 : QATAR1
Qatar-2 : QATAR2
TOI-270 : TOI270
TOI-700 : TOI700
TOI-849 : TOI849
TRAPPIST-1 : TRAPPIST1
TrES-1 : TRES1
TrES-2 : TRES2
TrES-3 : TRES3
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
WASP-77 : WASP77A
WASP-78 : WASP78
WASP-79 : WASP79
WASP-8 : WASP8
WASP-80 : WASP80
WASP-87 : WASP87
WASP-94 : WASP94A
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
# JWST-NIRISS-NIS-CLEAR-GR700XD

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
    # GEOFF UPDATE
    overwrite['GJ 3053'] = {
        # 'b':{'t0':2456915.70654,
        #     't0_uperr':0.00004, 't0_lowerr':-0.00004,
        #     't0_ref':'GMR'}
        'b':{'t0':2458399.9303,
             't0_uperr':0.0012, 't0_lowerr':-0.0013,
             't0_ref':'Lillo-Box et al. 2020'}
    }
    overwrite['GJ 436'] = {
        # 'b':{'inc':86.774,
        #     'inc_uperr':0.03, 'inc_lowerr':-0.03,
        #     'inc_ref':'Knutson et al. 2014',
        #     't0':2456295.431924,
        #     't0_uperr':0.000045, 't0_lowerr':-0.000045,
        #     't0_ref':'Knutson et al. 2014',
        #     'ecc':0, 'ecc_ref':'Knutson et al. 2014',
        #     'sma':0.030826087286013763,
        #     'sma_uperr':0.0002,
        #     'sma_lowerr':-0.0002,
        #     'sma_ref':'Knutson et al. 2014',
        #     'period':2.64389782,
        #     'period_uperr':0.00000008, 'period_lowerr':-0.00000008,
        #     'period_ref':'Knutson et al. 2014'}
        'b':{'inc':86.44,
             'inc_uperr':0.17, 'inc_lowerr':-0.16,
             'inc_ref':'Maciejewski et al. 2014',
             't0':2454510.80162,
             't0_uperr':0.00007, 't0_lowerr':-0.00007,
             't0_ref':'Maciejewski et al. 2014',
             'ecc':0.13827, 'ecc_ref':'Maciejewski et al. 2014',
             'sma':0.0291,
             'sma_uperr':0.0015,
             'sma_lowerr':-0.0015,
             'sma_ref':'Maciejewski et al. 2014',
             'period':2.64388312,
             'period_uperr':0.00000057, 'period_lowerr':-0.00000057,
             'period_ref':'Maciejewski et al. 2014'}
    }
    overwrite['GJ 9827'] = {
        'R*':0.637, 'R*_uperr':0.063, 'R*_lowerr':-0.063,
        'R*_ref':'Prieto-Arranz et al. 2018',
        # 'b':{'inc':88.33,
        #     'inc_uperr':1.15, 'inc_lowerr':-2.1,
        #     'inc_ref':'Prieto-Arranz et al. 2018',
        #     't0':2457738.52646,
        #     't0_uperr':0.00044, 't0_lowerr':-0.00042,
        #     't0_ref':'Prieto-Arranz et al. 2018',
        #     'sma':0.0210,
        #     'sma_uperr':0.0024,
        #     'sma_lowerr':-0.0026,
        #     'sma_ref':'Prieto-Arranz et al. 2018',
        #     'period':1.208966,
        #     'period_uperr':0.000012, 'period_lowerr':-0.000012,
        #     'period_ref':'Prieto-Arranz et al. 2018',
        #     'logg':3.1445742076096161,
        #     'logg_uperr':0.11, 'logg_lowerr':-0.11,
        #     'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'},
        # 'c':{'inc':89.07,
        #     'inc_uperr':0.59, 'inc_lowerr':-0.92,
        #     'inc_ref':'Prieto-Arranz et al. 2018',
        #     't0':2457738.54961,
        #     't0_uperr':0.00146, 't0_lowerr':-0.00145,
        #     't0_ref':'Prieto-Arranz et al. 2018',
        #     'sma':0.0439,
        #     'sma_uperr':0.0050,
        #     'sma_lowerr':-0.0055,
        #     'sma_ref':'Prieto-Arranz et al. 2018',
        #     'period':3.648227,
        #     'period_uperr':0.000117, 'period_lowerr':-0.000119,
        #     'period_ref':'Prieto-Arranz et al. 2018',
        #     'logg':2.9479236198317262,
        #     'logg_uperr':0.18, 'logg_lowerr':-0.18,
        #     'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'},
        # 'd':{'inc':87.703,
        #     'inc_uperr':0.081, 'inc_lowerr':-0.253,
        #     'inc_ref':'Prieto-Arranz et al. 2018',
        #     't0':2457740.96198,
        #     't0_uperr':0.00084, 't0_lowerr':-0.00086,
        #     't0_ref':'Prieto-Arranz et al. 2018',
        #     'sma':0.0625,
        #     'sma_uperr':0.0071,
        #     'sma_lowerr':-0.0078,
        #     'sma_ref':'Prieto-Arranz et al. 2018',
        #     'period':6.201419,
        #     'period_uperr':0.000128, 'period_lowerr':-0.000128,
        #     'period_ref':'Prieto-Arranz et al. 2018',
        #     'logg':2.7275412570285562,
        #     'logg_uperr':0.15, 'logg_lowerr':-0.15,
        #     'logg_ref':'Prieto-Arranz et al. 2018', 'logg_units':'log10[cm.s-2]'}
        'b':{'inc':86.07,
             'inc_uperr':0.41, 'inc_lowerr':-0.34,
             'inc_ref':'Rice et al. 2019',
             't0':2457738.82586,
             't0_uperr':0.00026, 't0_lowerr':-0.00026,
             't0_ref':'Rice et al. 2019',
             'sma':0.0188,
             'sma_uperr':0.0002,
             'sma_lowerr':-0.0002,
             'sma_ref':'Rice et al. 2019',
             'period':1.2089819,
             'period_uperr':6.93e-06, 'period_lowerr':-7.14e-06,
             'period_ref':'Rice et al. 2019',
             'logg':3.284795731186922,
             'logg_uperr':0.3284795731186922, 'logg_lowerr':-0.3284795731186922,
             'logg_ref':'System Prior Auto Fill',
             'logg_units':'log10[cm.s-2]'},
        'c':{'inc':88.19,
             'inc_uperr':0.21, 'inc_lowerr':-0.18,
             'inc_ref':'Rice et al. 2019',
             't0':2457742.1993,
             't0_uperr':0.00072, 't0_lowerr':-0.00073,
             't0_ref':'Rice et al. 2019',
             'sma':0.03925,
             'sma_uperr':0.00042,
             'sma_lowerr':-0.00029,
             'sma_ref':'Rice et al. 2019',
             'period':3.6480957,
             'period_uperr':6.33e-05, 'period_lowerr':-6.21e-05,
             'period_ref':'Rice et al. 2019',
             'logg':2.7252634420333446,
             'logg_uperr':0.27252634420333444, 'logg_lowerr':-0.27252634420333444,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'},
        'd':{'inc':87.443,
             'inc_uperr':0.045, 'inc_lowerr':-0.045,
             'inc_ref':'Rice et al. 2019',
             't0':2457740.96115,
             't0_uperr':0.00044, 't0_lowerr':-0.00045,
             't0_ref':'Rice et al. 2019',
             'sma':0.05591,
             'sma_uperr':0.00059,
             'sma_lowerr':-0.00041,
             'sma_ref':'Rice et al. 2019',
             'period':6.2014698,
             'period_uperr':6.26e-05, 'period_lowerr':-6.26e-05,
             'period_ref':'Rice et al. 2019',
             'logg':2.987906013084224,
             'logg_uperr':0.2987906013084224, 'logg_lowerr':-0.2987906013084224,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'}
    }
    overwrite['HAT-P-1'] = {
        # 'b':{'t0':2453979.92802,
        #     't0_uperr':0.00004, 't0_lowerr':-0.00004,
        #     't0_ref':'GMR'}
        'b':{'t0':2453979.93202,
             't0_uperr':0.00024, 't0_lowerr':-0.00024,
             't0_ref':'Nikolov et al. 2014'}
    }
    overwrite['HAT-P-11'] = {
        'R*':0.683, 'R*_uperr':0.009, 'R*_lowerr':-0.009,
        'R*_ref':'Yee et al. 2018',
        # 'b':{'inc':90,
        #     'inc_uperr':0.114, 'inc_lowerr':-0.114,
        #     'inc_ref':'Yee et al. 2018',
        #     't0':2454957.8132067,
        #     't0_uperr':0.0000663, 't0_lowerr':-0.0000663,
        #     't0_ref':'Yee et al. 2018',
        #     'rp':0.3889738152520562,
        #     'rp_uperr':0.0017, 'rp_lowerr':-0.0017,
        #     'rp_ref':'Yee et al. 2018',
        #     'sma':0.05254,
        #     'sma_uperr':0.00064, 'sma_lowerr':-0.00064,
        #     'sma_ref':'Yee et al. 2018',
        #     'ecc':0,
        #     'ecc_uperr':0.034, 'ecc_lowerr':-0.034,
        #     'ecc_ref':'Yee et al. 2018',
        #     'period':4.887802443,
        #     'period_uperr':0.0000071, 'period_lowerr':-0.0000071,
        #     'period_ref':'Yee et al. 2018'}
        'b':{'inc':88.5,
             'inc_uperr':0.6, 'inc_lowerr':-0.6,
             'inc_ref':'Stassun et al. 2017',
             't0':2455109.335119,
             't0_uperr':2.1e-5, 't0_lowerr':-2.1e-5,
             't0_ref':'Kokori et al. 2021',
             'rp':0.389,
             'rp_uperr':0.005, 'rp_lowerr':-0.005,
             'rp_ref':'Yee et al. 2018',
             'sma':0.05254,
             'sma_uperr':0.00064, 'sma_lowerr':-0.00064,
             'sma_ref':'Yee et al. 2018',
             'ecc':0.218,
             'ecc_uperr':0.034, 'ecc_lowerr':-0.034,
             'ecc_ref':'Yee et al. 2018',
             'period':4.88780244,
             'period_uperr':3e-7, 'period_lowerr':-3e-7,
             'period_ref':'Yee et al. 2018'}
    }
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
             'omega_ref':"Bonomo et al. 2017"}
    }
    overwrite['HAT-P-3'] = {
        # 'b':{'inc':87.24,
        #     'inc_uperr':0.69, 'inc_lowerr':-0.69,
        #     'inc_ref':'Torres et al. 2017',
        #     'sma':0.03894,
        #     'sma_uperr':0.0007, 'sma_lowerr':-0.0007,
        #     'sma_ref':'Torres et al. 2017',
        #     't0':2454218.81,
        #     't0_uperr':0.0029, 't0_lowerr':-0.0029,
        #     't0_ref':'Torres et al. 2017 + GMR',
        #     'logg':3.310,
        #     'logg_uperr':0.066, 'logg_lowerr':-0.066,
        #     'logg_ref':'Torres et al. 2017', 'logg_units':'log10[cm.s-2]'}
        'b':{'inc':87.24,
             'inc_uperr':0.69, 'inc_lowerr':-0.69,
             'inc_ref':'Stassun et al. 2017',
             'sma':0.03878,
             'sma_uperr':0.00065, 'sma_lowerr':-0.00065,
             'sma_ref':'Kokori et al. 2021',
             't0':2455694.72623,
             't0_uperr':8e-05, 't0_lowerr':-8e-05,
             't0_ref':'Kokori et al. 2021',
             'logg':3.2609631221802866,
             'logg_uperr':0.32609631221802865, 'logg_lowerr':-0.32609631221802865,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'}
    }
    overwrite['HAT-P-38'] = {
        'FEH*':0.06, 'FEH*_uperr':0.1, 'FEH*_lowerr':-0.1,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Sato et al. 2012',
        'b':{'omega':240,
             'omega_lowerr':-104,
             'omega_uperr':104,
             'omega_ref':"Sato et al. 2012"}
    }
    overwrite['HAT-P-41'] = {
        'R*':1.683, 'R*_uperr':0.058, 'R*_lowerr':-0.036,
        'R*_ref':'Hartman et al. 2012',
        # 'b':{'inc':87.7,
        #     'inc_uperr':1.0, 'inc_lowerr':-1.0,
        #     'inc_ref':'Hartman et al. 2012',
        #     't0':2454983.86167,
        #     't0_uperr':0.00107, 't0_lowerr':-0.00107,
        #     't0_ref':'Hartman et al. 2012',
        #     'sma':0.0426,
        #     'sma_uperr':0.0005, 'sma_lowerr':-0.0005,
        #     'sma_ref':'Hartman et al. 2012',
        #     'period':2.694047,
        #     'period_ref':'Hartman et al. 2012'}
        'b':{'inc':87.7,
             'inc_uperr':1.0, 'inc_lowerr':-1.0,
             'inc_ref':'Stassun et al. 2017',
             't0':2458071.24389,
             't0_uperr':0.00012, 't0_lowerr':-0.00012,
             't0_ref':'Kokori et al. 2021',
             'sma':0.04258,
             'sma_uperr':0.00047, 'sma_lowerr':-0.00048,
             'sma_ref':'Kokori et al. 2021',
             'period':2.69405,
             'period_ref':'Stassun et al. 2017'}
    }
    overwrite['HD 106315'] = {
        'R*':1.18, 'R*_uperr':0.11, 'R*_lowerr':-0.11,
        'R*_ref':'Crossfield et al. 2017',
        'T*':6290, 'T*_uperr':60, 'T*_lowerr':-60,
        'T*_ref':'Crossfield et al. 2017',
        'FEH*':-0.24, 'FEH*_uperr':0.04, 'FEH*_lowerr':-0.04,
        'FEH*_ref':'Crossfield et al. 2017',
        'LOGG*':4.29, 'LOGG*_uperr':0.07, 'LOGG*_lowerr':-0.07,
        'LOGG*_ref':'Crossfield et al. 2017',
        # 'b':{'inc':88.4,
        #     'inc_uperr':1.1, 'inc_lowerr':-2.1,
        #     'inc_ref':'Crossfield et al. 2017',
        #     't0':2457605.6521,
        #     't0_uperr':0.0042, 't0_lowerr':-0.0045,
        #     't0_ref':'Crossfield et al. 2017',
        #     'sma':0.09012,
        #     'sma_uperr':0.00083, 'sma_lowerr':-0.00085,
        #     'sma_ref':'Crossfield et al. 2017',
        #     'ecc':0,
        #     'ecc_ref':'Crossfield et al. 2017',
        #     'rp':0.199,
        #     'rp_ref':'Crossfield et al. 2017',
        #     'period':9.5521,
        #     'period_ref':'Crossfield et al. 2017',
        #     'logg':2.51,
        #     'logg_uperr':0.26, 'logg_lowerr':-0.26,
        #     'logg_ref':'Crossfield et al. 2017', 'logg_units':'log10[cm.s-2]'},
        # 'c':{'inc':89.42,
        #     'inc_uperr':0.4, 'inc_lowerr':-0.67,
        #     'inc_ref':'Crossfield et al. 2017',
        #     't0':2457611.1310,
        #     't0_uperr':0.0012, 't0_lowerr':-0.0012,
        #     't0_ref':'Crossfield et al. 2017',
        #     'sma':0.1526,
        #     'sma_uperr':0.0014, 'sma_lowerr':-0.0014,
        #     'sma_ref':'Crossfield et al. 2017',
        #     'ecc':0,
        #     'ecc_ref':'Crossfield et al. 2017',
        #     'rp':0.352,
        #     'rp_ref':'Crossfield et al. 2017',
        #     'period':21.0576,
        #     'period_ref':'Crossfield et al. 2017',
        #     'logg':2.98,
        #     'logg_uperr':0.3, 'logg_lowerr':-0.3,
        #     'logg_ref':'Crossfield et al. 2017', 'logg_units':'log10[cm.s-2]'}
        'b':{'inc':87.54,
             'inc_uperr':0.32, 'inc_lowerr':-0.32,
             'inc_ref':'Barros et al. 2017',
             't0':2457586.5487,
             't0_uperr':0.0029, 't0_lowerr':-0.0029,
             't0_ref':'Barros et al. 2017',
             'sma':0.0907,
             'sma_uperr':0.001, 'sma_lowerr':-0.001,
             'sma_ref':'Barros et al. 2017',
             'ecc':0.093,
             'ecc_ref':'Barros et al. 2017',
             'rp':0.218,
             'rp_ref':'Barros et al. 2017',
             'period':9.55237,
             'period_ref':'Barros et al. 2017',
             'logg':3.3155261313408566,
             'logg_uperr':0.3315526131340857, 'logg_lowerr':-0.3315526131340857,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'},
        'c':{'inc':88.61,
             'inc_uperr':0.7, 'inc_lowerr':-0.25,
             'inc_ref':'Barros et al. 2017',
             't0':2457569.0173,
             't0_uperr':0.0014, 't0_lowerr':-0.0014,
             't0_ref':'Barros et al. 2017',
             'sma':0.1536,
             'sma_uperr':0.0017, 'sma_lowerr':-0.0017,
             'sma_ref':'Barros et al. 2017',
             'ecc':0.22,
             'ecc_ref':'Barros et al. 2017',
             'rp':0.388,
             'rp_ref':'Barros et al. 2017',
             'period':21.05704,
             'period_ref':'Barros et al. 2017',
             'logg':2.8962515933281696,
             'logg_uperr':0.28962515933281696, 'logg_lowerr':-0.28962515933281696,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'}
    }
    overwrite['HD 97658'] = {
        # 'b':{'inc':89.85,
        #     'inc_uperr':0.15, 'inc_lowerr':-0.48,
        #     'inc_ref':'Knutson et al. 2014',
        #     't0':2456523.12537,
        #     't0_uperr':0.00049, 't0_lowerr':-0.00049,
        #     't0_ref':'Knutson et al. 2014',
        #     'sma':0.09030091308645881,
        #     'sma_uperr':0.004164, 'sma_lowerr':-0.004164,
        #     'sma_ref':'Knutson et al. 2014',
        #     'ecc':0, 'ecc_ref':'Knutson et al. 2014',
        #     'period':9.489264,
        #     'period_ref':'Knutson et al. 2014'}
        'b':{'inc':89.05,
             'inc_uperr':0.41, 'inc_lowerr':-0.24,
             'inc_ref':'Ellis et al. 2021',
             't0':2458904.9366,
             't0_uperr':0.0008, 't0_lowerr':-0.0008,
             't0_ref':'Ellis et al. 2021',
             'sma':0.0805,
             'sma_uperr':0.001, 'sma_lowerr':-0.001,
             'sma_ref':'ExoFOP-TESS TOI',
             'ecc':0.05, 'ecc_ref':'Ellis et al. 2021',
             'period':9.4897116,
             'period_ref':'Ellis et al. 2021'}
    }
    overwrite['K2-24'] = {
        # 'b':{'logg':2.9233923050832749,
        #     'logg_uperr':0.28, 'logg_lowerr':-0.28,
        #     'logg_ref':'Petigura et al. 2016', 'logg_units':'log10[cm.s-2]'},
        # 'c':{'logg':3.2300188146519129,
        #     'logg_uperr':0.3, 'logg_lowerr':-0.3,
        #     'logg_ref':'Petigura et al. 2016', 'logg_units':'log10[cm.s-2]'}
        'b':{'logg':2.804767306962391,
             'logg_uperr':0.28047673069623913, 'logg_lowerr':-0.28,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'},
        'c':{'logg':2.4287470185879663,
             'logg_uperr':0.24287470185879662, 'logg_lowerr':-0.24287470185879662,
             'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'}
    }
    overwrite['K2-3'] = {
        # 'b':{
        #    'mass':0.0181,
        #    'mass_uperr':0.0077,
        #    'mass_lowerr':-0.0054,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':3.0730,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        # 'c':{
        #    'mass':0.0154,
        #    'mass_uperr':0.0066,
        #    'mass_lowerr':-0.0046,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':3.1462,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        # 'd':{
        #    'mass':0.0154,
        #    'mass_uperr':0.0066,
        #    'mass_lowerr':-0.0046,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':3.1462,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        'b':{
            'mass':0.02039,
            'mass_uperr':0.00311,
            'mass_lowerr':-0.00293,
            'mass_ref':'Kosiarek et al. 2019',
            'mass_units':'Jupiter mass',
            'logg':3.141655964019513,
            'logg_lowerr':-0.3141655964019513, 'logg_uperr':0.3141655964019513,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.00673,
            'mass_uperr':0.0034,
            'mass_lowerr':-0.00327,
            'mass_ref':'Kosiarek et al. 2019',
            'mass_units':'Jupiter mass',
            'logg':2.887685867464453,
            'logg_lowerr':-0.2887685867464453, 'logg_uperr':0.2887685867464453,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
        'd':{
            'mass':0.00881,
            'mass_uperr':0.0008810000000000001,
            'mass_lowerr':-0.0008810000000000001,
            'mass_ref':'Kosiarek et al. 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0046467116525237,
            'logg_lowerr':-0.3004646711652524, 'logg_uperr':0.3004646711652524,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        }
    }
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
             'mass':0.0258,
             'mass_uperr':0.0110,
             'mass_lowerr':-0.0077,
             'mass_ref':'Pearson 2019',
             'mass_units':'Jupiter mass',
             'logg':2.9791,
             'logg_lowerr':-0.1539, 'logg_uperr':0.1539,
             'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'}
    }
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
        'b':{
            'inc':87.6,
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
            'period_ref':'Siverd et al. 2012',
            "Spitzer_IRAC1_subarray": [
                0.3499475318779155,
                -0.13450119362315333,
                0.07098128685193948,
                -0.019248332190717504
            ],
            "Spitzer_IRAC2_subarray": [
                0.34079591311025204,
                -0.21763621595372798,
                0.1569303075862828,
                -0.048363772020055255
            ],
            "TESS": [
                0.4289436996072044,
                0.15082100841265006,
                0.18753067052587288,
                -0.15714785519864086
            ]
        }
    }

    overwrite['Kepler-16'] = {
        'R*':0.665924608009903, 'R*_uperr':0.0013, 'R*_lowerr':-0.0013,
        'R*_ref':'Oroz + GMR',
        # 'b':{'inc':89.7511397641686,
        #     'inc_uperr':0.0323, 'inc_lowerr':-0.04,
        #     'inc_ref':'Oroz + GMR',
        #     't0':2457914.235774330795,
        #     't0_uperr':0.004, 't0_lowerr':-0.004,
        #     't0_ref':'Oroz',
        #     'sma':0.7048,
        #     'sma_uperr':0.0011,
        #     'sma_lowerr':-0.0011,
        #     'sma_ref':'Doyle et al. 2011',
        #     'period':228.776,
        #     'period_ref':'Doyle et al. 2011'}
        'b':{'inc':90.032,
             'inc_uperr':0.002, 'inc_lowerr':-0.002,
             'inc_ref':'Doyle et al. 2011',
             't0':2455212.12316,
             't0_uperr':0.004, 't0_lowerr':-0.004,
             't0_ref':'Doyle et al. 2011',
             'sma':0.7048,
             'sma_uperr':0.0011,
             'sma_lowerr':-0.0011,
             'sma_ref':'Doyle et al. 2011',
             'period':228.776,
             'period_ref':'Doyle et al. 2011'}
    }
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
        'b':{
            'inc':82.5,
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
            'omega_ref':"Knutson et al. 2014",
            # limb darkening
            "Spitzer_IRAC1_subarray": [
                0.36885966190119,
                -0.148367404490232,
                0.07112997446947285,
                -0.014533906130047942
            ],
            "Spitzer_IRAC2_subarray": [
                0.33948631752691805,
                -0.19254408234857706,
                0.1277084571166541,
                -0.037068426815200436
            ],
            "TESS": [
                0.5502048381392664,
                -0.1915164225261399,
                0.5800130134071905,
                -0.29914817100785457
            ]
        }
    }
    overwrite['WASP-39'] = {
        'FEH*':-0.10, 'FEH*_uperr':0.1, 'FEH*_lowerr':-0.1,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Faedi et al. 2011'}
    overwrite['WASP-43'] = {
        'FEH*':-0.05, 'FEH*_uperr':0.17, 'FEH*_lowerr':-0.17,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':'Hellier et al. 2011',
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.5214015151262713,
                -0.116913722511716,
                -0.0025615252155260474,
                0.008679785618454554
            ],
            "Spitzer_IRAC2_subarray": [
                0.43762215323543396,
                -0.17305029863164503,
                0.09760807455104326,
                -0.029028877897651247
            ],
            "TESS": [
                0.4589036057499657,
                0.02833282271380786,
                0.436737566941398,
                -0.18583579059956945
            ]
        }
    }
    overwrite['WASP-6'] = {
        # 'b':{'inc':88.38,
        #     'inc_uperr':0.31, 'inc_lowerr':-0.31,
        #     'inc_ref':'Tregloan-Reed et al. 2015',
        #     't0':2456132.41081,
        #     't0_uperr':0.00015, 't0_lowerr':-0.00015,
        #     't0_ref':'Tregloan-Reed et al. 2015',
        #     'sma':0.0414,
        #     'sma_uperr':0.001,
        #     'sma_lowerr':-0.001,
        #     'sma_ref':'Tregloan-Reed et al. 2015',
        #     'period':3.361,
        #     'period_ref':'Tregloan-Reed et al. 2015'}
        'b':{'inc':88.47,
             'inc_uperr':0.56, 'inc_lowerr':-0.56,
             'inc_ref':'Stassun et al. 2017',
             't0':2455591.28967,
             't0_uperr':7e-05, 't0_lowerr':-7e-05,
             't0_ref':'Kokori et al. 2021',
             'sma':0.04217,
             'sma_uperr':0.00079,
             'sma_lowerr':-0.0012,
             'sma_ref':'Kokori et al. 2021',
             'period':3.36101,
             'period_ref':'Stassun et al. 2017'}
    }
    overwrite['XO-2'] = {
        # 'b':{'t0':2455565.54048,
        #     't0_uperr':0.005, 't0_lowerr':-0.005,
        #     't0_ref':'GMR'}
        'b':{'t0':2454508.73829,
             't0_uperr':0.00014, 't0_lowerr':-0.00016,
             't0_ref':'Crouzet et al. 2012'}
    }
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
            'omega_ref':"Stassun et al. 2017",
            "Spitzer_IRAC1_subarray": [
                0.4028945813566236,
                -0.1618193396025557,
                0.08312362942354319,
                -0.019766348298489313
            ],
            "Spitzer_IRAC2_subarray": [
                0.3712209668752866,
                -0.1996422788905644,
                0.12409504521199885,
                -0.033786702881953186
            ],
            "TESS": [
                0.5744071203410743,
                -0.20530021464676967,
                0.5968926436545263,
                -0.29448085413107394
            ]
        }
    }
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
            'omega_ref':"Bonomo et al. 2017",
            "Spitzer_IRAC1_subarray": [
                0.3556193331718539,
                -0.13491841927882636,
                0.06201863236774508,
                -0.012634699997427995
            ],
            "Spitzer_IRAC2_subarray": [
                0.3352914789225599,
                -0.1977755003834447,
                0.13543229121842332,
                -0.040489856045654665
            ],
            "TESS": [
                0.5070158474296776,
                -0.08317080742284876,
                0.46257441555844714,
                -0.25970830964110125
            ]
        }
    }
    overwrite['WASP-34'] = {
        'b':{
            'omega':215,
            'omega_lowerr':-150,
            'omega_uperr':77,
            'omega_ref':"Knutson et al. 2014",
            "Spitzer_IRAC1_subarray": [
                0.428983331524869,
                -0.18290950217251944,
                0.09885346596732751,
                -0.025116667946425204
            ],
            "Spitzer_IRAC2_subarray": [
                0.3809141367993901,
                -0.19189122283729515,
                0.11270592554391648,
                -0.02937059129121932
            ],
            "TESS": [
                0.5667281891657291,
                -0.16843841198851486,
                0.567918864270408,
                -0.2807821851882787
            ]
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
            'mass':0.137,
            'mass_uperr':0.008,
            'mass_lowerr':-0.01,
            'mass_ref':'Hadden & Lithwick et al. 2017',
            'mass_units':'Jupiter mass',
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
        # 'b':{
        #    'mass':0.0135,
        #    'mass_uperr':0.0058,
        #    'mass_lowerr':-0.0040,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':3.2305,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        # 'c':{
        #    'mass':0.0213,
        #    'mass_uperr':0.0091,
        #    'mass_lowerr':-0.0064,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':3.0195,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        # 'd':{
        #    'mass':0.0229,
        #    'mass_uperr':0.0097,
        #    'mass_lowerr':-0.0068,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':3.0025,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        # 'e':{
        #    'mass':0.0313,
        #    'mass_uperr':0.0133,
        #    'mass_lowerr':-0.0093,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':2.9538,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        # 'f':{
        #    'mass':0.0246,
        #    'mass_uperr':0.0105,
        #    'mass_lowerr':-0.0074,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':2.9871,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # }
        'b':{
            'mass':0.00975,
            'mass_uperr':0.0033,
            'mass_lowerr':-0.0033,
            'mass_ref':'Lopez et al. 2019',
            'mass_units':'Jupiter mass',
            'logg':3.122642551445353,
            'logg_lowerr':-0.31226425514453526, 'logg_uperr':0.31226425514453526,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.01985,
            'mass_uperr':0.00356,
            'mass_lowerr':-0.00356,
            'mass_ref':'Lopez et al. 2019',
            'mass_units':'Jupiter mass',
            'logg':3.0685582617244536,
            'logg_lowerr':-0.30685582617244533, 'logg_uperr':0.30685582617244533,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
        'd':{
            'mass':0.02492,
            'mass_uperr':0.00437,
            'mass_lowerr':-0.00425,
            'mass_ref':'Lopez et al. 2019',
            'mass_units':'Jupiter mass',
            'logg':3.134094303846485,
            'logg_lowerr':-0.3134094303846485, 'logg_uperr':0.3134094303846485,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
        'e':{
            'mass':0.04081,
            'mass_uperr':0.00623,
            'mass_lowerr':-0.00626,
            'mass_ref':'Lopez et al. 2019',
            'mass_units':'Jupiter mass',
            'logg':3.045058181595798,
            'logg_lowerr':-0.3045058181595798, 'logg_uperr':0.3045058181595798,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
        'f':{
            'mass':0.00513,
            'mass_uperr':0.00667,
            'mass_lowerr':-0.00371,
            'mass_ref':'Lopez et al. 2019',
            'mass_units':'Jupiter mass',
            'logg':2.277823309686141,
            'logg_lowerr':-0.22778233096861408, 'logg_uperr':0.22778233096861408,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
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
    overwrite['K2-32'] = {
        # 'b':{
        #    'mass':0.0717,
        #    'mass_uperr':0.0306,
        #    'mass_lowerr':-0.0215,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':2.9578,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        # 'c':{
        #    'mass':0.0238,
        #    'mass_uperr':0.0101,
        #    'mass_lowerr':-0.0071,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':2.9940,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        # 'd':{
        #    'mass':0.0310,
        #    'mass_uperr':0.0132,
        #    'mass_lowerr':-0.0092,
        #    'mass_ref':'Pearson 2019',
        #    'mass_units':'Jupiter mass',
        #    'logg':2.9548,
        #    'logg_lowerr':-0.15, 'logg_uperr':0.15,
        #    'logg_ref':'Pearson 2019', 'logg_units':'log10[cm.s-2]'
        # },
        'b':{
            'mass':0.0472,
            'mass_uperr':0.00566,
            'mass_lowerr':-0.00535,
            'mass_ref':'Lillo-Box et al. 2020',
            'mass_units':'Jupiter mass',
            'logg':2.718525189895293,
            'logg_lowerr':-0.2718525189895293, 'logg_uperr':0.2718525189895293,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
        'c':{
            'mass':0.02549,
            'mass_uperr':0.00755,
            'mass_lowerr':-0.00755,
            'mass_ref':'Lillo-Box et al. 2020',
            'mass_units':'Jupiter mass',
            'logg':2.9063592455216574,
            'logg_lowerr':-0.29063592455216575, 'logg_uperr':0.29063592455216575,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
        'd':{
            'mass':0.02108,
            'mass_uperr':0.00787,
            'mass_lowerr':-0.00787,
            'mass_ref':'Lillo-Box et al. 2020',
            'mass_units':'Jupiter mass',
            'logg':2.7326553012236623,
            'logg_lowerr':-0.2732655301223662, 'logg_uperr':0.2732655301223662,
            'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
        },
    }
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
    overwrite["CoRoT-2"]= {
        "b":{
            "Spitzer-IRAC-IR-45-SUB":{
                "rprs": 0.15417,
                "ars": 6.60677,
                "inc": 88.08,
                "ref": "KAP"
            },
            "Spitzer_IRAC1_subarray": [
                0.4423526389772671,
                -0.20200004648957037,
                0.11665312313321362,
                -0.03145249632862833
            ],
            "Spitzer_IRAC2_subarray": [
                0.38011095282800866,
                -0.18511089957904475,
                0.10671540314411156,
                -0.027341272754041506
            ],
            "TESS": [
                0.5676749293408454,
                -0.17061159554569968,
                0.5746825114940707,
                -0.2813825305235465
            ]
        }
    }
    # GEOFF UPDATE
    overwrite['GJ 1132'] = {
        'b':{
            # 'inc':88.41,
            'inc':86.58,
            # 'inc_uperr':2.6, 'inc_lowerr':-2.7,
            'inc_uperr':0.63, 'inc_lowerr':-0.63,
            # 'inc_ref':'KAP',
            'inc_ref':'Southworth et al. 2017',
            "Spitzer_IRAC1_subarray": [
                0.8808056407530139,
                -0.7457051451918199,
                0.4435989599088468,
                -0.11533694981224148
            ],
            "Spitzer_IRAC2_subarray": [
                0.8164503264173831,
                -0.7466319094521022,
                0.448251686664617,
                -0.11545411611119284
            ],
            "TESS": [
                1.3124310421082044,
                -0.6962246317249355,
                0.39036394089945214,
                -0.09647398152509991
            ]
        }
    }

    # Limb darkening coefficients computed with
    # https://github.com/ucl-exoplanets/ExoTETHyS
    overwrite["HD 189733"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.44354479528696034,
                -0.08947631097404696,
                -0.00435085810513761,
                0.00910594090013591
            ],
            "Spitzer_IRAC2_subarray": [
                0.38839253774457744,
                -0.17732423236450323,
                0.11810733772498544,
                -0.03657823168637474
            ],
            "TESS": [
                0.5285425956986032,
                -0.09651910670073637,
                0.5231848346221262,
                -0.2319323579788086
            ]
        }
    }

    overwrite["HD 209458"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.3801802812964466,
                -0.14959456473437277,
                0.08226460839494475,
                -0.02131689251855459
            ],
            "Spitzer_IRAC2_subarray": [
                0.36336971411871394,
                -0.21637839776809858,
                0.14792839620718698,
                -0.04286208270953803
            ],
            "TESS": [
                0.5208972672724171,
                -0.11501360812879736,
                0.5234666949470197,
                -0.2790487932759741
            ]
        }
    }

    overwrite["KELT-9"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.3313765755137107,
                -0.3324189051186633,
                0.2481428693159906,
                -0.0730038845221279
            ],
            "Spitzer_IRAC2_subarray": [
                0.3081800911324395,
                -0.32930034680921816,
                0.24236433024537915,
                -0.07019145797258527
            ],
            "TESS": [
                0.5917625640397705,
                -0.14708612614194713,
                0.10024661953421281,
                -0.034877320754916084
            ]
        }
    }

    # this one (WASP-33) has FEH* in the Exoplanet Archive, but no error bars
    #  there's three values in the table: 0.0,0.1,0.1 (no error bars for any of them)
    overwrite["WASP-33"] = {
        'FEH*':0.1, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':"Zhang et al. 2018",
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.3360838105569875,
                -0.20369556446757797,
                0.14180512020307806,
                -0.04279692505871632
            ],
            "Spitzer_IRAC2_subarray": [
                0.3250093115902963,
                -0.2634497438671309,
                0.19583740005736275,
                -0.05877816796715111
            ],
            "TESS": [
                0.21802280399616877,
                0.5962514686390606,
                -0.23031791722840977,
                -0.03585575616303472
            ]
        }
    }

    overwrite["WASP-103"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.3758384436095625,
                -0.1395975318171088,
                0.0693688736769638,
                -0.0162794345748232
            ],
            "Spitzer_IRAC2_subarray": [
                0.35938501700892445,
                -0.20947695473210462,
                0.14144809404384948,
                -0.040990804709028064
            ],
            "TESS": [
                0.5429130318486702,
                -0.156616144761292,
                0.5495530165809148,
                -0.28366840104598084
            ]
        }
    }

    overwrite["KELT-16"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.36916688544219783,
                -0.13957103936844534,
                0.07119044558535764,
                -0.018675120861031937
            ],
            "Spitzer_IRAC2_subarray": [
                0.3549546430076809,
                -0.2155295179664244,
                0.15043075368738368,
                -0.04514276133343067
            ],
            "TESS": [
                0.48135397145943876,
                0.043931390276899254,
                0.29719848335546284,
                -0.189371721999753
            ]
        }
    }

    overwrite["WASP-121"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.35343787303428653,
                -0.13444332321181401,
                0.06955169678670275,
                -0.018419427272667512
            ],
            "Spitzer_IRAC2_subarray": [
                0.34304917676671737,
                -0.21584682478514353,
                0.15457937580646092,
                -0.04742589029069567
            ],
            "TESS": [
                0.4444522032204701,
                0.1140605956202882,
                0.22720009715036546,
                -0.17020540487953517
            ]
        }
    }

    overwrite["KELT-20"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.3344300791273318,
                -0.28913882534855895,
                0.21010188872209157,
                -0.061790086627358104
            ],
            "Spitzer_IRAC2_subarray": [
                0.31807958384684176,
                -0.31082384355713244,
                0.22719653693837855,
                -0.06610531710958574
            ],
            "TESS": [
                0.40543237466351073,
                0.3436812246865186,
                -0.2906804877197937,
                0.06010577003234808
            ]
        }
    }

    overwrite["HAT-P-7"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.3637280943230625,
                -0.1424523111830543,
                0.06824894539731155,
                -0.014578311756816686
            ],
            "Spitzer_IRAC2_subarray": [
                0.3397626817587848,
                -0.1976647471694525,
                0.13403605366799531,
                -0.039618202725551235
            ],
            "TESS": [
                0.5246842605144799,
                -0.11953856525920781,
                0.4980149231109441,
                -0.26961464726248285
            ]
        }
    }

    overwrite["WASP-76"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.37508785151968,
                -0.15123541065822635,
                0.07733376118565834,
                -0.018551329687575616
            ],
            "Spitzer_IRAC2_subarray": [
                0.3503514529020057,
                -0.20455423732189046,
                0.13804368117965113,
                -0.040335740501177636
            ],
            "TESS": [
                0.526591056531864,
                -0.09631931826112265,
                0.46060203700341984,
                -0.2504015503095966
            ]
        }
    }

    overwrite["WASP-19"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.4485159019167513,
                -0.20853342549558768,
                0.12340401852800424,
                -0.034379873907955626
            ],
            "Spitzer_IRAC2_subarray": [
                0.3825821042080216,
                -0.18816485899811294,
                0.11078803979402557,
                -0.029013926574001703
            ],
            "TESS": [
                0.5721195104718484,
                -0.17820365872523525,
                0.5798089692388008,
                -0.2789522827254285
            ]
        }
    }

    overwrite["KELT-7"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.34264654838082775,
                -0.15263897274083513,
                0.09529544911293918,
                -0.028445262603159684
            ],
            "Spitzer_IRAC2_subarray": [
                0.3362025591248816,
                -0.23293403482921576,
                0.17083787539542783,
                -0.05207951610486718
            ],
            "TESS": [
                0.3610592308582314,
                0.2667423496103761,
                0.09821129170800148,
                -0.13712088235733524
            ]
        }
    }

    overwrite["KELT-14"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.4493742805668009,
                -0.24639842685467592,
                0.1640017017159755,
                -0.04777987900661082
            ],
            "Spitzer_IRAC2_subarray": [
                0.3941297227246355,
                -0.22099935114922328,
                0.13393719173432234,
                -0.03441830202500127
            ],
            "TESS": [
                0.6046731674770963,
                -0.2742435037232662,
                0.6696872069231958,
                -0.3137463600188404
            ]
        }
    }

    overwrite["WASP-74"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.3953736210736681,
                -0.16565075764571777,
                0.09318866035061182,
                -0.024247454399023875
            ],
            "Spitzer_IRAC2_subarray": [
                0.3696898156305795,
                -0.2125028155143125,
                0.13938377401131619,
                -0.03911691640241164
            ],
            "TESS": [
                0.5475220569094276,
                -0.16373562078107398,
                0.5668650786990014,
                -0.2898135162447064
            ]
        }
    }

    overwrite["HD 149026"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.367160938263667,
                -0.1305588325879479,
                0.06612034580484898,
                -0.018337084470844645
            ],
            "Spitzer_IRAC2_subarray": [
                0.3601039845595216,
                -0.22202760617949738,
                0.15729906194707344,
                -0.047668070962046706
            ],
            "TESS": [
                0.45979496744567355,
                0.10907781129247562,
                0.22771175767591864,
                -0.16374442687430096
            ]
        }
    }

    overwrite["TrES-3"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.43597266162376314,
                -0.19001215158398352,
                0.1056322815109545,
                -0.027744065630210032
            ],
            "Spitzer_IRAC2_subarray": [
                0.3810582548901189,
                -0.18972122146795323,
                0.1119886599627006,
                -0.029375587180729256
            ],
            "TESS": [
                0.5618074240304971,
                -0.15624585173162106,
                0.5619019121433206,
                -0.27886386958979975
            ]
        }
    }

    overwrite["WASP-77 A"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.44975696488393374,
                -0.17728194779592824,
                0.09158922569805375,
                -0.02479615071127561
            ],
            "Spitzer_IRAC2_subarray": [
                0.3906922322805967,
                -0.20039168705178179,
                0.13035218295191758,
                -0.03796619908753919
            ],
            "TESS": [
                0.5712236947886342,
                -0.16753044700656006,
                0.5618010283481705,
                -0.25682008121195343
            ]
        }
    }

    overwrite["WASP-95"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.413096705663071,
                -0.17163366883006448,
                0.09079314140373773,
                -0.022304168790776884
            ],
            "Spitzer_IRAC2_subarray": [
                0.3772265443367232,
                -0.2006526678014234,
                0.12252287674677063,
                -0.03283398178266608
            ],
            "TESS": [
                0.5768453770747085,
                -0.20310034971026414,
                0.5953732832218406,
                -0.2919313179831538
            ]
        }
    }

    overwrite["WASP-140"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.4294348108523915,
                -0.10313120787437623,
                0.016430633217998755,
                0.0008036783543789007
            ],
            "Spitzer_IRAC2_subarray": [
                0.39023842724510155,
                -0.19846841760266795,
                0.13665600099047498,
                -0.04254450064955062
            ],
            "TESS": [
                0.5442195480943604,
                -0.10646416484874582,
                0.5052815024147247,
                -0.2297014632374973
            ]
        }
    }

    overwrite["WASP-52"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.4542826787797213,
                -0.10475364767168102,
                0.01183804531437866,
                0.0029171050937958822
            ],
            "Spitzer_IRAC2_subarray": [
                0.3906750566059761,
                -0.17705502335732198,
                0.11703224188529365,
                -0.03571381970099965
            ],
            "TESS": [
                0.5292627627275839,
                -0.11149731065770283,
                0.554194374222394,
                -0.24527978805928258
            ]
        }
    }

    overwrite["GJ 1214"] = {
        "b": {
            "Spitzer_IRAC1_subarray": [
                0.9083242210542111,
                -0.7976911808204602,
                0.4698074336560188,
                -0.12001861589169728
            ],
            "Spitzer_IRAC2_subarray": [
                0.8239880988090422,
                -0.760781868877928,
                0.4513165756893245,
                -0.11497950826716168
            ],
            "TESS": [
                1.3498752247932537,
                -0.7092017952698113,
                0.3622420357029451,
                -0.08762538276594896
            ]
        }
    }

    overwrite['WASP-87'] = {
        'FEH*':0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[Fe/H]', 'FEH*_ref':"Default to solar metallicity"}

    # only one system (this one) is missing an H_mag
    #  make a guess at it based on V=16.56,I=15.30
    overwrite['OGLE-TR-056'] = {
        'Hmag':14, 'Hmag_uperr':1, 'Hmag_lowerr':-1,
        'Hmag_units':'[mag]', 'Hmag_ref':'Geoff guess'}

    return overwrite
# -------------------------------------------------------------------
