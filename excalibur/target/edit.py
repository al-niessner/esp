'''target task inputs and parameter overwrite'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie
import logging; log = logging.getLogger(__name__)
# ------------- ------------------------------------------------------
# -- ACTIVE FILTERS -- -----------------------------------------------
# FORMAT:
# OBSERVATORY/INSTRUMENT/DETECTOR/FILTER/MODE
# DO NOT MODIFY THIS, USE processme() RULES INSTEAD
def activefilters():
    '''
Spitzer-IRAC-IR-36-SUB
Spitzer-IRAC-IR-45-SUB
HST-WFC3-IR-G141-SCAN
HST-WFC3-IR-G102-SCAN
HST-STIS-CCD-G750L-STARE
HST-STIS-CCD-G430L-STARE
JWST-NIRISS-NIS-CLEAR-GR700XD
JWST-NIRCAM-NRCALONG-F322W2-GRISMR
JWST-NIRCAM-IMAGE-F210M-WLP8
JWST-NIRSPEC-NRS-F290LP-G395H
JWST-NIRSPEC-NRS-CLEAR-PRISM
    '''
    return
# ----------------------------- --------------------------------------
# -- PROCESS RULES -- ------------------------------------------------
def processme():
    '''
    - Empty list means everything goes through
    - Include/Exclude ['a', 'b'] will include/exclude filter/target including 'a' or 'b'
    -- FILTER: partial match
    -- TARGET: exact match
    '''
    out = {}
    out['FILTER'] = {}
    # out['FILTER']['include'] = []
    # out['FILTER']['include'] = ['NIRSPEC']
    out['FILTER']['include'] = ['HST', 'Ariel']
    out['FILTER']['exclude'] = []
    # out['FILTER']['exclude'] = ['STIS', 'JWST', 'Spitzer', 'Ariel']
    # out['FILTER']['exclude'] = ['HST', 'JWST', 'Spitzer']
    out['TARGET'] = {}
    # Best to use a function call that returns a specific list
    # if one wants to change the 'include' target content
    out['TARGET']['include'] = []
    out['TARGET']['exclude'] = dropouts()
    return out

def proceed(name, ext=None, verbose=False):
    '''
    GMR: High level filtering
    '''
    out = True
    rules = processme()
    filterkeys = [r for r in ['include', 'exclude'] if rules['FILTER'][r]]
    if ext:
        for thisrule in filterkeys:
            # partial name matching for filters, such that 'HST' matches all HST filters
            trout = any(itm in ext for itm in rules['FILTER'][thisrule])
            if 'exclude' in thisrule: trout = not trout
            if ext=='any filter': trout=True
            out = out and trout
            if verbose: log.warning('>---- FILTER %s: %s %s', ext, thisrule, out)
            pass
        pass
    namekeys = [r for r in ['include', 'exclude'] if rules['TARGET'][r]]
    for thisrule in namekeys:
        # exact name matching for targets, otherwise TOI-175 removes TOI-1759
        trout = any(itm==name for itm in rules['TARGET'][thisrule])
        if 'exclude' in thisrule: trout = not trout
        out = out and trout
        if verbose: log.warning('>---- TARGET %s: %s %s', name, thisrule, out)
        pass
    if verbose: log.warning('>-- PROCEED: %s', out)
    return out

def dropouts():
    '''
    GEOFF: Bad boys list
    '''
    out = ['HR 8799',
           'GJ 3193',  # doesn't exist in the Exoplanet Archive
           'HIP 41378',  # this is K2-93
           'HD 185603', 'HD 195689', 'HD 197481',
           'KIC 12266812', 'TIC 184892124',
           'LHS 6343',  # has G141, but doesn't exist in the Exoplanet Archive (listed as false positive)
           'TOI-175', 'TOI-193',
           'LHS 1140',  # alias for GJ 3053
           'LHS 1140 (taurex sim @TS)',
           '55 Cnc (taurex sim @TS)',  # drop taurex-sim targets; they waste CPU
           'AU Mic (taurex sim @TS)',
           'GJ 1132 (taurex sim @TS)',
           'GJ 1214 (taurex sim @TS)',
           'GJ 3053 (taurex sim @TS)',
           'GJ 3470 (taurex sim @TS)',
           'GJ 436 (taurex sim @TS)',
           'GJ 9827 (taurex sim @TS)',
           'HAT-P-1 (taurex sim @TS)',
           'HAT-P-11 (taurex sim @TS)',
           'HAT-P-12 (taurex sim @TS)',
           'HAT-P-17 (taurex sim @TS)',
           'HAT-P-18 (taurex sim @TS)',
           'HAT-P-2 (taurex sim @TS)',
           'HAT-P-24 (taurex sim @TS)',
           'HAT-P-26 (taurex sim @TS)',
           'HAT-P-3 (taurex sim @TS)',
           'HAT-P-32 (taurex sim @TS)',
           'HAT-P-38 (taurex sim @TS)',
           'HAT-P-41 (taurex sim @TS)',
           'HATS-7 (taurex sim @TS)',
           'HD 106315 (taurex sim @TS)',
           'HD 149026 (taurex sim @TS)',
           'HD 189733 (taurex sim @TS)',
           'HD 191939 (taurex sim @TS)',
           'HD 209458 (taurex sim @TS)',
           'HD 219666 (taurex sim @TS)',
           'HD 97658 (taurex sim @TS)',
           'K2-18 (taurex sim @TS)',
           'K2-24 (taurex sim @TS)',
           'K2-3 (taurex sim @TS)',
           'K2-33 (taurex sim @TS)',
           'K2-93 (taurex sim @TS)',
           'K2-96 (taurex sim @TS)',
           'KELT-1 (taurex sim @TS)',
           'KELT-11 (taurex sim @TS)',
           'KELT-20 (taurex sim @TS)',
           'KELT-7 (taurex sim @TS)',
           'Kepler-138 (taurex sim @TS)',
           'Kepler-16 (taurex sim @TS)',
           'Kepler-79 (taurex sim @TS)',
           'L 98-59 (taurex sim @TS)',
           'LTT 1445 A (taurex sim @TS)',
           'LTT 9779 (taurex sim @TS)',
           'TOI-1201 (taurex sim @TS)',
           'TOI-1231 (taurex sim @TS)',
           'TOI-270 (taurex sim @TS)',
           'TOI-674 (taurex sim @TS)',
           'TRAPPIST-1 (taurex sim @TS)',
           'V1298 Tau (taurex sim @TS)',
           'WASP-101 (taurex sim @TS)',
           'WASP-103 (taurex sim @TS)',
           'WASP-107 (taurex sim @TS)',
           'WASP-117 (taurex sim @TS)',
           'WASP-12 (taurex sim @TS)',
           'WASP-121 (taurex sim @TS)',
           'WASP-127 (taurex sim @TS)',
           'WASP-17 (taurex sim @TS)',
           'WASP-178 (taurex sim @TS)',
           'WASP-18 (taurex sim @TS)',
           'WASP-19 (taurex sim @TS)',
           'WASP-29 (taurex sim @TS)',
           'WASP-31 (taurex sim @TS)',
           'WASP-39 (taurex sim @TS)',
           'WASP-43 (taurex sim @TS)',
           'WASP-52 (taurex sim @TS)',
           'WASP-6 (taurex sim @TS)',
           'WASP-62 (taurex sim @TS)',
           'WASP-63 (taurex sim @TS)',
           'WASP-67 (taurex sim @TS)',
           'WASP-69 (taurex sim @TS)',
           'WASP-74 (taurex sim @TS)',
           'WASP-76 (taurex sim @TS)',
           'WASP-79 (taurex sim @TS)',
           'WASP-80 (taurex sim @TS)',
           'WASP-96 (taurex sim @TS)',
           'WASP-98 (taurex sim @TS)',
           'XO-1 (taurex sim @TS)',
           'XO-2 (taurex sim @TS)',
           'TOI-561 (taurex sim @TS)']
    return out
# ------------------- ------------------------------------------------
# -- CREATE -- -------------------------------------------------------
def createversion():
    '''
    1.2.0: Added Spitzer targets
    1.3.0: Added JWST-NIRISS-NIS-CLEAR-G700XD
    1.4.0: Added more targets (242 total)
    1.5.0: New limb darkening coefficients for spitzer targest
    1.6.0: Added a few FEH* values and one Hmag; removed some redundant settings
    1.7.0: Added confirmed planets in Ariel target list
    1.7.1: WFC3 targets 2023
    1.8.0: JWST filters
    '''
    return dawgie.VERSION(1,8,0)
# ------------ -------------------------------------------------------
# -- TARGET LIST -- --------------------------------------------------
# FIRST COL HAS TO BE SOLVABLE BY
# -- Obsolete https://archive.stsci.edu/hst/
# https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
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
HD 185603 :
HD 189733 :
HD 195689 :
HD 197481 :
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
Kepler-11 :
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
KIC 12266812 :
LHS 3844 :
OGLE-TR-056 :
OGLE-TR-10 :
Qatar-1 :
Qatar-2 :
TIC 184892124 :
TOI-175 :
TOI-193 :
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
AU Mic :
CoRoT-5 :
CoRoT-11 :
CoRoT-19 :
DS Tuc A :
EPIC 211945201 :
EPIC 246851721 :
G 9-40 :
GJ 3473 :
GJ 357 :
GJ 367 :
GJ 3929 :
GJ 486 :
GPX-1 :
HAT-P-14 :
HAT-P-21 :
HAT-P-24 :
HAT-P-25 :
HAT-P-27 :
HAT-P-28 :
HAT-P-29 :
HAT-P-31 :
HAT-P-35 :
HAT-P-36 :
HAT-P-37 :
HAT-P-39 :
HAT-P-42 :
HAT-P-43 :
HAT-P-44 :
HAT-P-45 :
HAT-P-46 :
HAT-P-49 :
HAT-P-50 :
HAT-P-51 :
HAT-P-52 :
HAT-P-53 :
HAT-P-54 :
HAT-P-55 :
HAT-P-56 :
HAT-P-57 :
HAT-P-58 :
HAT-P-59 :
HAT-P-60 :
HAT-P-61 :
HAT-P-62 :
HAT-P-64 :
HAT-P-65 :
HAT-P-66 :
HAT-P-67 :
HAT-P-68 :
HAT-P-69 :
HAT-P-70 :
HAT-P-9 :
HATS-1 :
HATS-11 :
HATS-13 :
HATS-18 :
HATS-2 :
HATS-23 :
HATS-24 :
HATS-25 :
HATS-26 :
HATS-27 :
HATS-29 :
HATS-30 :
HATS-31 :
HATS-33 :
HATS-34 :
HATS-35 :
HATS-37 A :
HATS-38 :
HATS-39 :
HATS-4 :
HATS-40 :
HATS-41 :
HATS-42 :
HATS-43 :
HATS-46 :
HATS-47 :
HATS-48 A :
HATS-5 :
HATS-50 :
HATS-51 :
HATS-52 :
HATS-53 :
HATS-56 :
HATS-57 :
HATS-58 A :
HATS-6 :
HATS-60 :
HATS-62 :
HATS-64 :
HATS-67 :
HATS-68 :
HATS-70 :
HATS-72 :
HATS-9 :
HD 108236 :
HD 110082 :
HD 110113 :
HD 136352 :
HD 1397 :
HD 152843 :
HD 15337 :
HD 183579 :
HD 191939 :
HD 202772 A :
HD 207897 :
HD 221416 :
HD 2685 :
HD 332231 :
HD 5278 :
HD 63433 :
HD 63935 :
HD 73583 :
HD 86226 :
HD 89345 :
HIP 65 A :
HIP 67522 :
K2-107 :
K2-116 :
K2-121 :
K2-129 :
K2-139 :
K2-140 :
K2-155 :
K2-198 :
K2-222 :
K2-232 :
K2-237 :
K2-238 :
K2-239 :
K2-260 :
K2-261 :
K2-266 :
K2-280 :
K2-284 :
K2-287 :
K2-29 :
K2-295 :
K2-329 :
K2-333 :
K2-334 :
K2-34 :
K2-353 :
K2-39 :
K2-403 :
K2-405 :
K2-406 :
KELT-10 :
KELT-12 :
KELT-15 :
KELT-17 :
KELT-18 :
KELT-19 A :
KELT-2 A :
KELT-21 :
KELT-23 A :
KELT-24 :
KELT-4 A :
KELT-6 :
KELT-8 :
KOI-13 :
KOI-94 :
KPS-1 :
Kepler-105 :
Kepler-108 :
Kepler-1314 :
Kepler-1513 :
Kepler-33 :
Kepler-396 :
Kepler-42 :
Kepler-435 :
Kepler-444 :
Kepler-447 :
Kepler-450 :
Kepler-468 :
Kepler-489 :
Kepler-76 :
Kepler-79 :
L 98-59 :
LHS 1478 :
LHS 1678 :
LP 714-47 :
LP 791-18 :
LTT 1445 A :
LTT 3780 :
LTT 9779 :
MASCARA-1 :
MASCARA-4 :
NGTS-10 :
NGTS-11 :
NGTS-12 :
NGTS-13 :
NGTS-2 :
NGTS-5 :
NGTS-6 :
NGTS-8 :
Qatar-10 :
Qatar-4 :
Qatar-5 :
Qatar-6 :
Qatar-7 :
Qatar-8 :
Qatar-9 :
TIC 257060897 :
TOI-1064 :
TOI-1075 :
TOI-1130 :
TOI-1201 :
TOI-122 :
TOI-1227 :
TOI-1231 :
TOI-125 :
TOI-1259 A :
TOI-1260 :
TOI-1266 :
TOI-1268 :
TOI-1296 :
TOI-1298 :
TOI-1333 :
TOI-1411 :
TOI-1431 :
TOI-1442 :
TOI-1478 :
TOI-150 :
TOI-1518 :
TOI-157 :
TOI-1601 :
TOI-163 :
TOI-1670 :
TOI-1685 :
TOI-169 :
TOI-1693 :
TOI-172 :
TOI-1728 :
TOI-1759 :
TOI-178 :
TOI-1789 :
TOI-1807 :
TOI-1842 :
TOI-1860 :
TOI-1899 :
TOI-201 :
TOI-2076 :
TOI-2109 :
TOI-216 :
TOI-2260 :
TOI-2337 :
TOI-237 :
TOI-2411 :
TOI-2427 :
TOI-257 :
TOI-2669 :
TOI-269 :
TOI-3362 :
TOI-421 :
TOI-431 :
TOI-4329 :
TOI-451 :
TOI-481 :
TOI-500 :
TOI-530 :
TOI-540 :
TOI-544 :
TOI-559 :
TOI-561 :
TOI-564 :
TOI-620 :
TOI-628 :
TOI-640 :
TOI-674 :
TOI-677 :
TOI-776 :
TOI-813 :
TOI-824 :
TOI-833 :
TOI-837 :
TOI-892 :
TOI-905 :
TOI-954 :
TrES-4 :
TrES-5 :
V1298 Tau :
WASP-105 :
WASP-106 :
WASP-110 :
WASP-113 :
WASP-114 :
WASP-117 :
WASP-118 :
WASP-119 :
WASP-120 :
WASP-123 :
WASP-124 :
WASP-126 :
WASP-132 :
WASP-133 :
WASP-135 :
WASP-136 :
WASP-138 :
WASP-139 :
WASP-141 :
WASP-142 :
WASP-145 A :
WASP-147 :
WASP-148 :
WASP-151 :
WASP-153 :
WASP-156 :
WASP-157 :
WASP-158 :
WASP-159 :
WASP-160 B :
WASP-161 :
WASP-163 :
WASP-164 :
WASP-165 :
WASP-166 :
WASP-167 :
WASP-168 :
WASP-169 :
WASP-170 :
WASP-172 :
WASP-173 A :
WASP-174 :
WASP-175 :
WASP-176 :
WASP-177 :
WASP-178 :
WASP-180 A :
WASP-181 :
WASP-182 :
WASP-183 :
WASP-184 :
WASP-185 :
WASP-186 :
WASP-187 :
WASP-189 :
WASP-190 :
WASP-192 :
WASP-20 :
WASP-22 :
WASP-23 :
WASP-25 :
WASP-41 :
WASP-42 :
WASP-44 :
WASP-45 :
WASP-47 :
WASP-53 :
WASP-54 :
WASP-55 :
WASP-56 :
WASP-57 :
WASP-58 :
WASP-61 :
WASP-66 :
WASP-68 :
WASP-70 A :
WASP-71 :
WASP-73 :
WASP-81 :
WASP-82 :
WASP-83 :
WASP-84 :
WASP-85 A :
WASP-88 :
WASP-89 :
WASP-90 :
WASP-91 :
WASP-92 :
WASP-93 :
WASP-98 :
WASP-99 :
Wolf 503 :
XO-6 :
XO-7 :
pi Men : HD 39091
HATS-22 :
K2-30 :
Kepler-1308 :
Qatar-3 :
WASP-129 :
WASP-144 :
Kepler-51 :
WD 1856 : WD 1856+534
GJ 4102 : LHS 475
HD 80606 :
GJ 4332 : L 168-9
LTT 5972 : TOI-836
    '''
# these two JWST targets are not yet listed in the Exoplanet Archive composite table:
# GJ 341 : HIP 45908
# GJ 1008 : HIP 1532  (TOI-260)
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
HD 185603 : HD185603
HD 189733 : HD189733
HD 195689 : HD195689
HD 197481 : HD197481
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
Kepler-11 : KEPLER11
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
KIC 12266812 : KIC12266812
LHS 3844 : LHS3844
OGLE-TR-056 : OGLETR056
OGLE-TR-10 : OGLETR10
Qatar-1 : QATAR1
Qatar-2 : QATAR2
TIC 184892124 : TIC184892124
TOI-175 : TOI175
TOI-193 : TOI193
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
AU Mic : AUMIC
CoRoT-5 : COROT5
CoRoT-11 : COROT11
CoRoT-19 : COROT19
DS Tuc A : DSTUCA
EPIC 211945201 : EPIC211945201
EPIC 246851721 : EPIC246851721
G 9-40 : G940
GJ 3473 : GJ3473
GJ 357 : GJ357
GJ 367 : GJ367
GJ 3929 : GJ3929
GJ 486 : GJ486
GPX-1 : GPX1
HAT-P-14 : HATP14
HAT-P-21 : HATP21
HAT-P-24 : HATP24
HAT-P-25 : HATP25
HAT-P-27 : HATP27
HAT-P-28 : HATP28
HAT-P-29 : HATP29
HAT-P-31 : HATP31
HAT-P-35 : HATP35
HAT-P-36 : HATP36
HAT-P-37 : HATP37
HAT-P-39 : HATP39
HAT-P-42 : HATP42
HAT-P-43 : HATP43
HAT-P-44 : HATP44
HAT-P-45 : HATP45
HAT-P-46 : HATP46
HAT-P-49 : HATP49
HAT-P-50 : HATP50
HAT-P-51 : HATP51
HAT-P-52 : HATP52
HAT-P-53 : HATP53
HAT-P-54 : HATP54
HAT-P-55 : HATP55
HAT-P-56 : HATP56
HAT-P-57 : HATP57
HAT-P-58 : HATP58
HAT-P-59 : HATP59
HAT-P-60 : HATP60
HAT-P-61 : HATP61
HAT-P-62 : HATP62
HAT-P-64 : HATP64
HAT-P-65 : HATP65
HAT-P-66 : HATP66
HAT-P-67 : HATP67
HAT-P-68 : HATP68
HAT-P-69 : HATP69
HAT-P-70 : HATP70
HAT-P-9 : HATP9
HATS-1 : HATS1
HATS-11 : HATS11
HATS-13 : HATS13
HATS-18 : HATS18
HATS-2 : HATS2
HATS-23 : HATS23
HATS-24 : HATS24
HATS-25 : HATS25
HATS-26 : HATS26
HATS-27 : HATS27
HATS-29 : HATS29
HATS-30 : HATS30
HATS-31 : HATS31
HATS-33 : HATS33
HATS-34 : HATS34
HATS-35 : HATS35
HATS-37 A : HATS37A
HATS-38 : HATS38
HATS-39 : HATS39
HATS-4 : HATS4
HATS-40 : HATS40
HATS-41 : HATS41
HATS-42 : HATS42
HATS-43 : HATS43
HATS-46 : HATS46
HATS-47 : HATS47
HATS-48 A : HATS48A
HATS-5 : HATS5
HATS-50 : HATS50
HATS-51 : HATS51
HATS-52 : HATS52
HATS-53 : HATS53
HATS-56 : HATS56
HATS-57 : HATS57
HATS-58 A : HATS58A
HATS-6 : HATS6
HATS-60 : HATS60
HATS-62 : HATS62
HATS-64 : HATS64
HATS-67 : HATS67
HATS-68 : HATS68
HATS-70 : HATS70
HATS-72 : HATS72
HATS-9 : HATS9
HD 108236 : HD108236
HD 110082 : HD110082
HD 110113 : HD110113
HD 136352 : HD136352
HD 1397 : HD1397
HD 152843 : HD152843
HD 15337 : HD15337
HD 183579 : HD183579
HD 191939 : HD191939
HD 202772 A : HD202772A
HD 207897 : HD207897
HD 221416 : HD221416
HD 2685 : HD2685
HD 332231 : HD332231
HD 5278 : HD5278
HD 63433 : HD63433
HD 63935 : HD63935
HD 73583 : HD73583
HD 86226 : HD86226
HD 89345 : HD89345
HIP 65 A : HIP65A
HIP 67522 : HIP67522
K2-107 : K2107
K2-116 : K2116
K2-121 : K2121
K2-129 : K2129
K2-139 : K2139
K2-140 : K2140
K2-155 : K2155
K2-198 : K2198
K2-222 : K2222
K2-232 : K2232
K2-237 : K2237
K2-238 : K2238
K2-239 : K2239
K2-260 : K2260
K2-261 : K2261
K2-266 : K2266
K2-280 : K2280
K2-284 : K2284
K2-287 : K2287
K2-29 : K229
K2-295 : K2295
K2-329 : K2329
K2-333 : K2333
K2-334 : K2334
K2-34 : K234
K2-353 : K2353
K2-39 : K239
K2-403 : K2403
K2-405 : K2405
K2-406 : K2406
KELT-10 : KELT10
KELT-12 : KELT12
KELT-15 : KELT15
KELT-17 : KELT17
KELT-18 : KELT18
KELT-19 A : KELT19A
KELT-2 A : KELT2A
KELT-21 : KELT21
KELT-23 A : KELT23A
KELT-24 : KELT24
KELT-4 A : KELT4A
KELT-6 : KELT6
KELT-8 : KELT8
KOI-13 : KOI13
KOI-94 : KOI94
KPS-1 : KPS1
Kepler-105 : KEPLER105
Kepler-108 : KEPLER108
Kepler-1314 : KEPLER1314
Kepler-1513 : KEPLER1513
Kepler-33 : KEPLER33
Kepler-396 : KEPLER396
Kepler-42 : KEPLER42
Kepler-435 : KEPLER435
Kepler-444 : KEPLER444
Kepler-447 : KEPLER447
Kepler-450 : KEPLER450
Kepler-468 : KEPLER468
Kepler-489 : KEPLER489
Kepler-76 : KEPLER76
Kepler-79 : KEPLER79
L 98-59 : L9859
LHS 1478 : LHS1478
LHS 1678 : LHS1678
LP 714-47 : LP71447
LP 791-18 : LP79118
LTT 1445 A : LTT1445A
LTT 3780 : LTT3780
LTT 9779 : LTT9779
MASCARA-1 : MASCARA1
MASCARA-4 : MASCARA4
NGTS-10 : NGTS10
NGTS-11 : NGTS11
NGTS-12 : NGTS12
NGTS-13 : NGTS13
NGTS-2 : NGTS2
NGTS-5 : NGTS5
NGTS-6 : NGTS6
NGTS-8 : NGTS8
Qatar-10 : QATAR10
Qatar-4 : QATAR4
Qatar-5 : QATAR5
Qatar-6 : QATAR6
Qatar-7 : QATAR7
Qatar-8 : QATAR8
Qatar-9 : QATAR9
TIC 257060897 : TIC257060897
TOI-1064 : TOI1064
TOI-1075 : TOI1075
TOI-1130 : TOI1130
TOI-1201 : TOI1201
TOI-122 : TOI122
TOI-1227 : TOI1227
TOI-1231 : TOI1231
TOI-125 : TOI125
TOI-1259 A : TOI1259A
TOI-1260 : TOI1260
TOI-1266 : TOI1266
TOI-1268 : TOI1268
TOI-1296 : TOI1296
TOI-1298 : TOI1298
TOI-1333 : TOI1333
TOI-1411 : TOI1411
TOI-1431 : TOI1431
TOI-1442 : TOI1442
TOI-1478 : TOI1478
TOI-150 : TOI150
TOI-1518 : TOI1518
TOI-157 : TOI157
TOI-1601 : TOI1601
TOI-163 : TOI163
TOI-1670 : TOI1670
TOI-1685 : TOI1685
TOI-169 : TOI169
TOI-1693 : TOI1693
TOI-172 : TOI172
TOI-1728 : TOI1728
TOI-1759 : TOI1759
TOI-178 : TOI178
TOI-1789 : TOI1789
TOI-1807 : TOI1807
TOI-1842 : TOI1842
TOI-1860 : TOI1860
TOI-1899 : TOI1899
TOI-201 : TOI201
TOI-2076 : TOI2076
TOI-2109 : TOI2109
TOI-216 : TOI216
TOI-2260 : TOI2260
TOI-2337 : TOI2337
TOI-237 : TOI237
TOI-2411 : TOI2411
TOI-2427 : TOI2427
TOI-257 : TOI257
TOI-2669 : TOI2669
TOI-269 : TOI269
TOI-3362 : TOI3362
TOI-421 : TOI421
TOI-431 : TOI431
TOI-4329 : TOI4329
TOI-451 : TOI451
TOI-481 : TOI481
TOI-500 : TOI500
TOI-530 : TOI530
TOI-540 : TOI540
TOI-544 : TOI544
TOI-559 : TOI559
TOI-561 : TOI561
TOI-564 : TOI564
TOI-620 : TOI620
TOI-628 : TOI628
TOI-640 : TOI640
TOI-674 : TOI674
TOI-677 : TOI677
TOI-776 : TOI776
TOI-813 : TOI813
TOI-824 : TOI824
TOI-833 : TOI833
TOI-837 : TOI837
TOI-892 : TOI892
TOI-905 : TOI905
TOI-954 : TOI954
TrES-4 : TRES4
TrES-5 : TRES5
V1298 Tau : V1298TAU
WASP-105 : WASP105
WASP-106 : WASP106
WASP-110 : WASP110
WASP-113 : WASP113
WASP-114 : WASP114
WASP-117 : WASP117
WASP-118 : WASP118
WASP-119 : WASP119
WASP-120 : WASP120
WASP-123 : WASP123
WASP-124 : WASP124
WASP-126 : WASP126
WASP-132 : WASP132
WASP-133 : WASP133
WASP-135 : WASP135
WASP-136 : WASP136
WASP-138 : WASP138
WASP-139 : WASP139
WASP-141 : WASP141
WASP-142 : WASP142
WASP-145 A : WASP145A
WASP-147 : WASP147
WASP-148 : WASP148
WASP-151 : WASP151
WASP-153 : WASP153
WASP-156 : WASP156
WASP-157 : WASP157
WASP-158 : WASP158
WASP-159 : WASP159
WASP-160 B : WASP160B
WASP-161 : WASP161
WASP-163 : WASP163
WASP-164 : WASP164
WASP-165 : WASP165
WASP-166 : WASP166
WASP-167 : WASP167
WASP-168 : WASP168
WASP-169 : WASP169
WASP-170 : WASP170
WASP-172 : WASP172
WASP-173 A : WASP173A
WASP-174 : WASP174
WASP-175 : WASP175
WASP-176 : WASP176
WASP-177 : WASP177
WASP-178 : WASP178
WASP-180 A : WASP180A
WASP-181 : WASP181
WASP-182 : WASP182
WASP-183 : WASP183
WASP-184 : WASP184
WASP-185 : WASP185
WASP-186 : WASP186
WASP-187 : WASP187
WASP-189 : WASP189
WASP-190 : WASP190
WASP-192 : WASP192
WASP-20 : WASP20
WASP-22 : WASP22
WASP-23 : WASP23
WASP-25 : WASP25
WASP-41 : WASP41
WASP-42 : WASP42
WASP-44 : WASP44
WASP-45 : WASP45
WASP-47 : WASP47
WASP-53 : WASP53
WASP-54 : WASP54
WASP-55 : WASP55
WASP-56 : WASP56
WASP-57 : WASP57
WASP-58 : WASP58
WASP-61 : WASP61
WASP-66 : WASP66
WASP-68 : WASP68
WASP-70 A : WASP70A
WASP-71 : WASP71
WASP-73 : WASP73
WASP-81 : WASP81
WASP-82 : WASP82
WASP-83 : WASP83
WASP-84 : WASP84
WASP-85 A : WASP85A
WASP-88 : WASP88
WASP-89 : WASP89
WASP-90 : WASP90
WASP-91 : WASP91
WASP-92 : WASP92
WASP-93 : WASP93
WASP-98 : WASP98
WASP-99 : WASP99
Wolf 503 : WOLF503
XO-6 : XO6
XO-7 : XO7
pi Men : PIMEN
HATS-22 : HATS22
K2-30 : K230
Kepler-1308 : KEPLER1308
Qatar-3 : QATAR3
WASP-129 : WASP129
WASP-144 : WASP144
Kepler-51 : KEPLER51
WD 1856 : WD1856
GJ 341 : GJ341
GJ 4102 : GJ4102
HD 80606 : HD80606
GJ 4332 : GJ4332
GJ 1008 : GJ1008
LTT 5972 : LTT5972
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
    'FEH*':[dex], 'FEH*_lowerr':[dex], 'FEH*_uperr':[dex], 'FEH*_ref':ref,
    'LOGG*':[dex CGS], 'LOGG*_lowerr':[dex CGS], 'LOGG*_uperr':[dex CGS], 'LOGG*_ref':ref,
    planet:
    {
    'inc':[degrees], 'inc_lowerr':[degrees], 'inc_uperr':[degrees], 'inc_ref':ref,
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
        'R*':0.637, 'R*_uperr':0.063, 'R*_lowerr':-0.063,
        'R*_ref':'Prieto-Arranz et al. 2018',
    }
    overwrite['HAT-P-11'] = {
        'R*':0.683, 'R*_uperr':0.009, 'R*_lowerr':-0.009,
        'R*_ref':'Yee et al. 2018',
        'b':{'period':4.88780244,
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
             'period_ref':'Howard et al. 2012'}
    }
    overwrite['HAT-P-3'] = {
        'b':{'sma':0.03878,
             'sma_uperr':0.00065, 'sma_lowerr':-0.00065,
             'sma_ref':'Kokori et al. 2021',
             't0':2455694.72623,
             't0_uperr':8e-05, 't0_lowerr':-8e-05,
             't0_ref':'Kokori et al. 2021'}
    }
    overwrite['HAT-P-41'] = {
        'R*':1.683, 'R*_uperr':0.058, 'R*_lowerr':-0.036,
        'R*_ref':'Hartman et al. 2012',
        'b':{'sma':0.04258,
             'sma_uperr':0.00047, 'sma_lowerr':-0.00048,
             'sma_ref':'Kokori et al. 2021'}
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
    }
    overwrite['HD 97658'] = {
        'b':{'sma':0.0805,
             'sma_uperr':0.001, 'sma_lowerr':-0.001,
             'sma_ref':'ExoFOP-TESS TOI'}
    }
    #  this logg is way off from what it's getting now (3.5). diff Rp I guess
    # overwrite['K2-3'] = {
    #    'd':{
    #        'logg':3.0046467116525237,
    #        'logg_lowerr':-0.3004646711652524, 'logg_uperr':0.3004646711652524,
    #        'logg_ref':'System Prior Auto Fill', 'logg_units':'log10[cm.s-2]'
    #    }
    # }
    overwrite['K2-33'] = {
        'FEH*':0.0, 'FEH*_uperr':0.13, 'FEH*_lowerr':-0.14,
        'FEH*_units':'[dex]', 'FEH*_ref':'Mann et al. 2016'}

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
        'FEH*_units':'[dex]', 'FEH*_ref':'Siverd et al. 2012',
        'b':{
            # hmm, these values are off a bit, even though its the same reference
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
        # the Triaud 2022 period (226 days, with no accompaning transit midtime) is no good
        #  none of the HST/G141 falls within the transit; data.timing is empty
        #  (in it's defense, that publication gives an errorbar of 1.7 days!)
        # the only other reference is the discovery paper with 228.776+-0.03
        #  no idea how they got such a small error bar; off by >100-sigma!
        # it's really hard to get the HST just right.  how did they schedule it?!
        #  HST is 12 orbits since the published T_0 in Jan.2020
        #   so a 0.01 error in period translates to a 3 hour shift in Jun.2017 (HST)
        # seems like t0=225.165 but it's still off a bit
        # let's just use the original params from a year ago:
        'b':{
            'inc':89.7511397641686,
            'inc_uperr':0.0323, 'inc_lowerr':-0.04,
            'inc_ref':'Oroz + GMR',
            't0':2457914.235774330795,
            't0_uperr':0.004, 't0_lowerr':-0.004,
            't0_ref':'Oroz',
            'period':228.776,
            'period_uperr':0.03,'period_lowerr':-0.03,
            'period_ref':'Doyle et al. 2011'
            }
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
        'FEH*_units':'[dex]', 'FEH*_ref':'Faedi et al. 2011',
        }
    overwrite['WASP-43'] = {
        'FEH*':-0.05, 'FEH*_uperr':0.17, 'FEH*_lowerr':-0.17,
        'FEH*_units':'[dex]', 'FEH*_ref':'Hellier et al. 2011',
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
        'b':{'t0':2455591.28967,
             't0_uperr':7e-05, 't0_lowerr':-7e-05,
             't0_ref':'Kokori et al. 2021',
             'sma':0.04217,
             'sma_uperr':0.00079,
             'sma_lowerr':-0.0012,
             'sma_ref':'Kokori et al. 2021'}
    }
    overwrite['XO-2'] = {
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
    overwrite['HAT-P-23'] = {
        'b':{
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
    overwrite['WASP-14'] = {
        'b':{
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
    overwrite['Kepler-9'] = {
        'b':{
            'mass':0.137,
            'mass_uperr':0.008,
            'mass_lowerr':-0.01,
            'mass_ref':'Hadden & Lithwick et al. 2017',
            'mass_units':'Jupiter mass',
        }
    }
    overwrite['LHS 3844'] = {
        'FEH*':0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':"Kyle's best guess",
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
        'FEH*_units':'[dex]', 'FEH*_ref':"Kyle's best guess",
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
    overwrite['GJ 1132'] = {
        'b':{
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

    overwrite["WASP-33"] = {
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
            # switch from the most recent ephemeris (Kokori 2022 = exoClock) back to the default
            'period':1.58040433,
            'period_uperr':1.3e-7, 'period_lowerr':-1.3e-7,
            'period_ref':'Cloutier et al. 2021',
            't0':2455701.413328,
            't0_uperr':0.000066, 't0_lowerr':-0.000059,
            't0_ref':'Cloutier et al. 2021',
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
        'FEH*_units':'[dex]', 'FEH*_ref':"Default to solar metallicity"}

    # only one system (this one) is missing an H_mag
    #  make a guess at it based on V=16.56,I=15.30
    overwrite['OGLE-TR-056'] = {
        'Hmag':14,
        'Hmag_uperr':1, 'Hmag_lowerr':-1,
        'Hmag_units':'[mag]', 'Hmag_ref':'Geoff guess'}

    # there's a bug in the Archive where this planet's radius is
    #  only given in Earth units, not our standard Jupiter units
    #  0.37+-0.18 REarth = 0.033+-0.16
    # mass is normally filled in via an assumed MRrelation; needs to be done here instead
    # logg is normally calculated from M+R; needs to be done here instead
    overwrite['Kepler-37'] = {
        'e':{'rp':0.033,
             'rp_uperr':0.016, 'rp_lowerr':-0.016,
             'rp_units':'[Jupiter radius]',
             'rp_ref':'Q1-Q8 KOI Table',
             'mass':0.0002,
             'mass_uperr':0.0002, 'mass_lowerr':-0.0001,
             'mass_units':'[Jupiter mass]',
             'mass_ref':'Assumed mass/radius relation',
             'logg':2.7,
             'logg_uperr':0.3, 'logg_lowerr':-0.2,
             'logg_units':'log10[cm.s-2]',
             'logg_ref':'Assumed mass/radius relation'}}

    # for the newly added comfirmed-planet Ariel targets, some metallicities are missing
    overwrite['AU Mic'] = {
        'FEH*':0.15, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Seli et al. 2002'}  # Neves 2013 has 0.32
    overwrite['HATS-50'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Chen et al. 2021'}
    overwrite['HATS-51'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Chen et al. 2021'}
    overwrite['HATS-52'] = {
        'FEH*':-0.09, 'FEH*_uperr':0.17, 'FEH*_lowerr':-0.17,
        'FEH*_units':'[dex]', 'FEH*_ref':'Magrini et al. 2022'}
    overwrite['HATS-53'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Chen et al. 2021'}
    overwrite['HATS-58 A'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Chen et al. 2021'}
    overwrite['K2-129'] = {
        'FEH*':0.105, 'FEH*_uperr':0.235, 'FEH*_lowerr':-0.235,
        'FEH*_units':'[dex]', 'FEH*_ref':'Hardagree-Ullman et al. 2020'}
    overwrite['LHS 1678'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    # Ammons 2006 has 0.76+-2.26, which is absurd
    overwrite['TIC 257060897'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-122'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-1227'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-1442'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-1693'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-237'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-2411'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-2427'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-451'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-540'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-544'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    overwrite['TOI-833'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    # this one is missing Hmag.  not sure why; it's in 2MASS/Simbad
    overwrite['K2-295'] = {
        'Hmag':11.135, 'Hmag_uperr':0.025, 'Hmag_lowerr':-0.025,
        'Hmag_units':'[mag]', 'Hmag_ref':'2MASS'}
    # this one is missing R* and M*.  That's unusual!
    # ah wait it does actually have a log-g measure of 4.1 (lower than Solar)
    # arg this one is really tricky.  planet semi-major axis is undefined without M*
    overwrite['WASP-110'] = {
        'R*':1.0, 'R*_uperr':0.25, 'R*_lowerr':-0.25,
        'R*_ref':'Default to solar radius',
        'M*':1.0, 'M*_uperr':0.25, 'M*_lowerr':-0.25,
        'M*_ref':'Default to solar mass',
        # RHO* derivation (from R* and M*) comes before this, so we have to set it here
        'RHO*':1.4, 'RHO*_uperr':0.25, 'RHO*_lowerr':-0.25,
        'RHO*_ref':'Default to solar density',
        # L* needed for teq (actually it's set below)
        'L*':1.0, 'L*_uperr':0.25, 'L*_lowerr':-0.25,
        'L*_ref':'Default to solar luminosity',
        # 'LOGG*':4.3, 'LOGG*_uperr':0.1, 'LOGG*_lowerr':-0.1,
        # 'LOGG*_ref':'Default to solar log(g)'}
        # Period is 3.87 days
        # teq derivation (from L* and sma) comes before this, so we have to set it here
        'b':{'sma':0.05, 'sma_uperr':0.01, 'sma_lowerr':-0.01,
             'sma_ref':'Assume solar mass',
             'teq':1245, 'teq_uperr':100, 'teq_lowerr:':-100,
             'teq_units':'[K]', 'teq_ref':'derived from L*,sma'}}

    # this one is weird. there's a metallicity value in the Archive for 'c' but not for 'b'
    overwrite['HD 63433'] = {
        'FEH*':0.05, 'FEH*_uperr':0.05, 'FEH*_lowerr':-0.05,
        'FEH*_units':'[dex]', 'FEH*_ref':'Dai et al. 2020'}

    overwrite['TOI-1411'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}

    # why isn't this in the archive?  non-hipparcos, but still..
    overwrite['TRAPPIST-1'] = {
        'dist':(1000./80.2123), 'dist_uperr':0.01, 'dist_lowerr':-0.01,
        'dist_units':'[pc]', 'dist_ref':'Gaia EDR3'}
    # had to use vizier for this one; not in simbad for some reason
    overwrite['NGTS-10'] = {
        'dist':(1000./3.8714), 'dist_uperr':12., 'dist_lowerr':-12.,
        'dist_units':'[pc]', 'dist_ref':'Gaia EDR3'}
    overwrite['Kepler-1314'] = {
        'dist':(1000./7.0083), 'dist_uperr':2., 'dist_lowerr':-2.,
        'dist_units':'[pc]', 'dist_ref':'Gaia EDR3'}
    # also had to use vizier for this one
    # it's in Gaia, but there's no parallax
    # wikipedia has it at 980pc from 2011 schneider site from buchhave 2011 discovery paper
    # the paper says it is from Girardi isochrone fitting
    overwrite['Kepler-14'] = {
        'dist':980., 'dist_uperr':100., 'dist_lowerr':-100.,
        'dist_units':'[pc]', 'dist_ref':'Buchhave et al. 2011'}

    # rather than filling in blank impact parameters here
    #  make it a mandatory parameter and then fill with default in system/core
    #  (same process as currently done for blank inclinations)
    # overwrite['TOI-201'] = {'b':{
    #    'impact':0., 'impact_uperr':0.5, 'impact_lowerr':-0.5,
    #    'impact_units':'[R*]', 'impact_ref':'default'}}
    # overwrite['WASP-110'] = {'b':{
    #    'impact':0., 'impact_uperr':0.5, 'impact_lowerr':-0.5,
    #    'impact_units':'[R*]', 'impact_ref':'default'}}

    # 11/10/23 period update to match G141 phase
    overwrite['HAT-P-26'] = {
        # 'b':{'period':4.234520,  # this is the default. decreasing it a bit
        'b':{'period':4.2345002,
             'period_uperr':7e-7, 'period_lowerr':-7e-7,
             'period_ref':'Kokori et al. 2022'}}

    # 11/12/23 period updates to match G141 phase
    overwrite['HAT-P-18'] = {
        # 'b':{'period':5.508023,  # this is the default. increasing it a bit
        # hmm, these are about the same.  what about t0?  308P+1.4min diff
        'b':{'period':5.5080287,
             'period_uperr':1.4e-6, 'period_lowerr':-1.4e-6,
             'period_ref':'Ivshina & Winn 2022'}}

    # some of the new JWST targets are missing mandatory parameters
    #  (without these system.finalize will crash)

    # not much in Vizier.  there's two StarHorse metallicities 0.0698 and -0.101483
    overwrite['GJ 4102'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}
    # even less in Vizier for this white dwarf.  e.g. C/He and Ca/He both blank
    overwrite['WD 1856'] = {
        'FEH*':0.0, 'FEH*_uperr':0.25, 'FEH*_lowerr':-0.25,
        'FEH*_units':'[dex]', 'FEH*_ref':'Default to solar metallicity'}

# HD 80606   data empty
# GJ 4332    data empty
# LTT 5972   data empty

    return overwrite
# -------------------------------------------------------------------
