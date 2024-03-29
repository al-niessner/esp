'''target core ds'''
# -- IMPORTS -- ------------------------------------------------------
import os
import numpy
import shutil
import re
import tempfile
import subprocess
import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.db

import excalibur.target as trg
import excalibur.target.edit as trgedit
import excalibur.system.core as syscore

import astropy.io.fits as pyfits
import urllib.error
import urllib.request as urlrequest
# ------------- ------------------------------------------------------
# -- URLTRICK -- -----------------------------------------------------
class urltrick():
    '''
    # GMR: with statement generates an attribute __enter__ error
    # without the with statement it doesnt pass CI checks
    # pulling out dirty tricks
    '''
    def __init__(self, thisurl):
        '''__init__ ds'''
        self.thisurl = thisurl
        self.req = None
        return

    def __enter__(self):
        '''__enter__ ds'''
        self.req = urlrequest.urlopen(self.thisurl)
        return self.req.read()

    def __exit__(self, exc_type, exc_value, traceback):
        '''__exit__ ds'''
        self.req.close()
        return
    pass
# -------------- -----------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    '''Checks for empty SV'''
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- TAP QUERY --  ---------------------------------------------------
def tap_query(base_url, query):
    '''table access protocol query'''
    # build url
    uri_full = base_url
    for k in query:
        if k != "format": uri_full+= f"{k} {query[k]} "
        pass
    uri_full = uri_full[:-1] + f"&format={query.get('format','csv')}"
    uri_full = uri_full.replace(' ','+')
    response = None
    with urltrick(uri_full) as test: response = test.decode('utf-8')
    return response
# ----------------- --------------------------------------------------
# -- SCRAPE IDS -- ---------------------------------------------------
def scrapeids(ds:dawgie.Dataset, out, web, genIDs=True):
    '''
    Parses table from ipac exoplanetarchive
    '''
    targets = trgedit.targetlist.__doc__
    targets = targets.split('\n')
    targets = [t.strip() for t in targets if t.replace(' ', '').__len__() > 0]
    tn = os.environ.get('TARGET_NAME', None)
    if tn is not None:
        found_target_list = None
        for target in targets:
            if tn == target.split(':')[0].strip(): found_target_list = target
            pass
        if found_target_list is None:
            # this is an ERROR.  the selected target should be in the target list
            # exit('ERROR: are you sure about that target?  it is not in the list')
            mssg = f'Obsolete target / Error in target name: {tn}'
            raise dawgie.NoValidOutputDataError(mssg)
        targets = [found_target_list]
        pass
    for target in targets:
        parsedstr = target.split(':')
        parsedstr = [t.strip() for t in parsedstr]
        out['starID'][parsedstr[0]] = {'planets':[], 'PID':[], 'aliases':[]}
        if parsedstr[1]:
            aliaslist = parsedstr[1].split(',')
            aliaslist = [a.strip() for a in aliaslist if a.strip()]
            out['starID'][parsedstr[0]]['aliases'].extend(aliaslist)
            pass
        if genIDs:
            # pylint: disable=protected-access
            dawgie.db.connect(trg.algorithms.create(), ds._bot(), parsedstr[0]).load()
            pass
        pass
    # new additions 4/14/23:
    #   pl_orblper (omega is used as prior in the transit fitting)
    #   pl_trandep (transit depth    is not used, but load it in, in case we want it later)
    #   pl_insol   (insolation       is not used, but load it in, in case we want it later)
    #   pl_trandur (transit duration is not used, but load it in, in case we want it later)
    #   pl_ratdor  (a/R*             is not used, but load it in, in case we want it later)
    #   pl_ratror  (Rp/R*            is not used, but load it in, in case we want it later)
    #   sy_dist    (same for distance; not used, but potentially useful)
    #  actually these additional references are only in the PSCompPars table, not the PS table
    # NO  sy_hmag_reflink (H magnitude reference can differ from the star param reference)
    # NO  sy_age_reflink  (age reference can differ from the star param reference)
    cols = "hostname,pl_letter,rowupdate,st_refname,pl_refname,sy_pnum,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,pl_bmassj,pl_bmassjerr1,pl_bmassjerr2,pl_radj,pl_radjerr1,pl_radjerr2,pl_dens,pl_denserr1,pl_denserr2,pl_eqt,pl_eqterr1,pl_eqterr2,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_imppar,pl_impparerr1,pl_impparerr2,st_teff,st_tefferr1,st_tefferr2,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2,st_lum,st_lumerr1,st_lumerr2,st_logg,st_loggerr1,st_loggerr2,st_dens,st_denserr1,st_denserr2,st_met,st_meterr1,st_meterr2,sy_hmag,sy_hmagerr1,sy_hmagerr2,st_age,st_ageerr1,st_ageerr2,pl_orblper,pl_orblpererr1,pl_orblpererr2,pl_trandep,pl_trandeperr1,pl_trandeperr2,pl_insol,pl_insolerr1,pl_insolerr2,pl_trandur,pl_trandurerr1,pl_trandurerr2,pl_ratdor,pl_ratdorerr1,pl_ratdorerr2,pl_ratror,pl_ratrorerr1,pl_ratrorerr2,sy_dist,sy_disterr1,sy_disterr2,st_spectype"
    uri_ipac_query = {
        "select": cols,
        "from": 'ps',
        "where": "tran_flag = 1 and default_flag = 1",
        "order by": "pl_name",
        "format": "csv"
    }
    # First Exoplanet Archive query gets default row for each target
    uri_ipac_query['where'] = 'tran_flag = 1 and default_flag = 1'
    response = tap_query(web, uri_ipac_query)
    for line in response.split('\n'): out['nexsciDefaults'].append(line)

    # Second Exoplanet Archive query gets all rows, not just the defaults
    uri_ipac_query['where'] = 'tran_flag = 1'
    response = tap_query(web, uri_ipac_query)
    for line in response.split('\n'): out['nexsciFulltable'].append(line)

    out['STATUS'].append(True)
    ds.update()
    return
# ---------------- ---------------------------------------------------
# -- CREATE FILTERS -- -----------------------------------------------
def createfltrs(out):
    '''
    Create filter name
    '''
    created = False
    filters = trgedit.activefilters.__doc__
    filters = filters.split('\n')
    filters = [t.strip() for t in filters if t.replace(' ', '')]
    out['activefilters']['TOTAL'] = len(filters)
    out['activefilters']['NAMES'] = filters
    if filters:
        for flt in filters:
            out['activefilters'][flt] = {'ROOTNAME':[], 'LOC':[], 'TOTAL':[]}
            pass
        created = True
        out['STATUS'].append(True)
        pass
    return created
# -------------------- -----------------------------------------------
# -- AUTOFILL -- -----------------------------------------------------
def autofillversion():
    '''
    1.1.1: manual trigger to autofill
    1.1.2: manual trigger for new spitzer priors
    1.1.3: manual trigger to fix new priors
    1.1.4: add stellar age integration
    1.1.5: added Hmag, L*; fixed FEH* problems
    1.2.0: new Exoplanet Archive tables and TAP protocol
    1.2.1: non-default parameters included; only two Exoplanet Archive queries; RHO* filled in by R*,M*
    '''
    return dawgie.VERSION(1,2,1)

def autofill(ident, thistarget, out,
             queryurl="https://archive.stsci.edu/hst/search.php?target=",
             action="&action=Search&resolver=SIMBAD&radius=3.0",
             outfmt="&outputformat=CSV&max_records=100000",
             opt="&sci_aec=S"):
    '''autofill ds'''

    out['starID'][thistarget] = ident['starID'][thistarget]
    targetID = [thistarget]

    # AUTOFILL WITH MAST QUERY ---------------------------------------
    solved = True
    querytarget = thistarget.replace(' ', '+')
    queryform = queryurl+querytarget+action+outfmt+opt
    failure = ['target not resolved, continue\n\n', 'no rows found\n\n',
               'target not resolved, continue', 'no rows found']
    with urltrick(queryform) as temp: framelist = temp.decode('utf-8')
    if framelist not in failure:
        framelist = framelist.split('\n')
        header = framelist[0].split(',')
        pidindex = header.index('Proposal ID')
        aliasindex = header.index('Target Name')
        pidlist = []
        aliaslist = []
        for frame in framelist[2:-1]:
            row = frame.split(',')
            pidlist.append(row[pidindex])
            aliaslist.append(row[aliasindex])
            pass
        pidlist = list(set(pidlist))
        aliaslist = list(set(aliaslist))
        out['starID'][thistarget]['aliases'].extend(aliaslist)
        out['starID'][thistarget]['PID'].extend(pidlist)
        solved = True
        out['STATUS'].append(True)
        pass
    solved = True  # target list autofill KP

    # AUTOFILL WITH NEXSCI EXOPLANET TABLE, DEFAULTS ONLY ---------------------------
    merged = False
    targetID.extend(ident['starID'][thistarget]['aliases'])
    response = ident['nexsciDefaults']
    header = response[0].split(',')
    # list all keys contained in csv
    # using tuples of form (a, b) where a is the key name
    # and then b are the other types of columns for that key
    # that are included
    # GMR: They are continuously changing the format: creating translatekeys()
    matchlist = translatekeys(header)
    banlist = ['star', 'planet', 'update', 'ref', 'ref_st', 'ref_pl', 'np']
    plist = ['period', 'period_uperr', 'period_lowerr', 'period_ref',
             'sma', 'sma_uperr', 'sma_lowerr', 'sma_ref',
             'ecc', 'ecc_uperr', 'ecc_lowerr', 'ecc_ref',
             'inc', 'inc_uperr', 'inc_lowerr', 'inc_ref',
             'mass', 'mass_uperr', 'mass_lowerr', 'mass_ref',
             'rp', 'rp_uperr', 'rp_lowerr', 'rp_ref',
             'rho', 'rho_uperr', 'rho_lowerr', 'rho_ref',
             'teq', 'teq_uperr', 'teq_lowerr', 'teq_ref',
             't0', 't0_uperr', 't0_lowerr', 't0_ref',
             'impact', 'impact_uperr', 'impact_lowerr', 'impact_ref',
             'omega','omega_uperr','omega_lowerr','omega_ref',
             'trandepth','trandepth_uperr','trandepth_lowerr','trandepth_ref',
             'trandur','trandur_uperr','trandur_lowerr','trandur_ref',
             'insol','insol_uperr','insol_lowerr','insol_ref',
             'ars','ars_uperr','ars_lowerr','ars_ref',
             'rprs','rprs_uperr','rprs_lowerr','rprs_ref']
    for line in response:
        elem = line.split(',')
        elem = clean_elems(elem)  # remove things such as bounding quotation marks
        trgt = elem[header.index('hostname')]
        if trgt in targetID:
            out['starID'][thistarget]['PLANETS #'] = elem[header.index('sy_pnum')]
            thisplanet = elem[header.index('pl_letter')]
            if thisplanet not in out['starID'][thistarget].keys():
                # GMR: Init planet dict
                out['starID'][thistarget][thisplanet] = {}
                if 'CREATED' not in out['starID'][thistarget][thisplanet].keys():
                    out['starID'][thistarget][thisplanet]['CREATED'] = True
                    # GMR: Inits with empty strings
                    null_keys = ['date', 'date_ref']
                    for key in null_keys:
                        out['starID'][thistarget][thisplanet][key] = ['']
                        pass
                    # GMR: Inits with empty lists
                    blank_keys = ['period_units', 'period_ref',
                                  'sma_units', 'sma_ref',
                                  'ecc_units', 'ecc_ref',
                                  'inc_units', 'inc_ref',
                                  'mass_units', 'mass_ref',
                                  'rp_units', 'rp_ref',
                                  'rho_units', 'rho_ref',
                                  'teq_units', 'teq_ref',
                                  't0_units', 't0_ref',
                                  'impact_units', 'impact_ref',
                                  'omega_units', 'omega_ref',
                                  'trandepth_units', 'trandepth_ref',
                                  'trandur_units', 'trandur_ref',
                                  'insol_units', 'insol_ref',
                                  'ars_units', 'ars_ref',
                                  'rprs_units', 'rprs_ref']
                    for key in blank_keys:
                        out['starID'][thistarget][thisplanet][key] = []
                        pass
                    pass
                # GMR: Init stellar dict
                if 'CREATED' not in out['starID'][thistarget].keys():
                    out['starID'][thistarget]['CREATED'] = True
                    out['starID'][thistarget]['date'] = ['']
                    out['starID'][thistarget]['date_ref'] = ['']
                    out['starID'][thistarget]['R*_units'] = []
                    out['starID'][thistarget]['R*_ref'] = []
                    out['starID'][thistarget]['M*_units'] = []
                    out['starID'][thistarget]['M*_ref'] = []
                    out['starID'][thistarget]['T*_units'] = []
                    out['starID'][thistarget]['T*_ref'] = []
                    out['starID'][thistarget]['L*_units'] = []
                    out['starID'][thistarget]['L*_ref'] = []
                    out['starID'][thistarget]['LOGG*_units'] = []
                    out['starID'][thistarget]['LOGG*_ref'] = []
                    out['starID'][thistarget]['RHO*_units'] = []
                    out['starID'][thistarget]['RHO*_ref'] = []
                    out['starID'][thistarget]['FEH*_ref'] = []
                    out['starID'][thistarget]['FEH*_units'] = ['[dex]']
                    out['starID'][thistarget]['AGE*'] = []
                    out['starID'][thistarget]['AGE*_uperr'] = []
                    out['starID'][thistarget]['AGE*_lowerr'] = []
                    out['starID'][thistarget]['AGE*_units'] = []
                    out['starID'][thistarget]['AGE*_ref'] = []
                    out['starID'][thistarget]['Hmag_units'] = []
                    out['starID'][thistarget]['Hmag_ref'] = []
                    out['starID'][thistarget]['dist_units'] = []
                    out['starID'][thistarget]['dist_ref'] = []
                    out['starID'][thistarget]['spTyp'] = []
                    out['starID'][thistarget]['spTyp_uperr'] = []
                    out['starID'][thistarget]['spTyp_lowerr'] = []
                    out['starID'][thistarget]['spTyp_units'] = []
                    out['starID'][thistarget]['spTyp_ref'] = []
                    pass
                ref_pl = elem[header.index('pl_refname')]
                ref_pl = ref_pl.split('</a>')[0]
                ref_pl = ref_pl.split('target=ref>')[-1]
                ref_pl = ref_pl.strip()
                ref_st = elem[header.index('st_refname')]
                ref_st = ref_st.split('</a>')[0]
                ref_st = ref_st.split('target=ref>')[-1]
                ref_st = ref_st.strip()
                pass
            for idx, keymatch in enumerate(matchlist):
                if keymatch not in banlist:
                    # GMR: Planet dict
                    if keymatch in plist: out_ref = out['starID'][thistarget][thisplanet]
                    # GMR: Stellar dict
                    else: out_ref = out['starID'][thistarget]
                    # Init
                    if keymatch not in out_ref: out_ref[keymatch] = []
                    out_ref[keymatch].append(elem[idx])
                    pass
                pass
            # GMR: Adding units to the planet dict
            out['starID'][thistarget][thisplanet]['period_units'].append('[days]')
            out['starID'][thistarget][thisplanet]['sma_units'].append('[AU]')
            out['starID'][thistarget][thisplanet]['ecc_units'].append('[]')
            out['starID'][thistarget][thisplanet]['inc_units'].append('[degree]')
            out['starID'][thistarget][thisplanet]['mass_units'].append('[Jupiter mass]')
            out['starID'][thistarget][thisplanet]['rp_units'].append('[Jupiter radius]')
            out['starID'][thistarget][thisplanet]['rho_units'].append('[g.cm-3]')
            out['starID'][thistarget][thisplanet]['teq_units'].append('[K]')
            out['starID'][thistarget][thisplanet]['t0_units'].append('[Julian Days]')
            out['starID'][thistarget][thisplanet]['impact_units'].append('[R*]')
            out['starID'][thistarget][thisplanet]['omega_units'].append('[degree]')
            out['starID'][thistarget][thisplanet]['trandepth_units'].append('[%]')
            out['starID'][thistarget][thisplanet]['trandur_units'].append('[hour]')
            out['starID'][thistarget][thisplanet]['insol_units'].append('[Earth flux]')
            out['starID'][thistarget][thisplanet]['ars_units'].append('[]')
            out['starID'][thistarget][thisplanet]['rprs_units'].append('[]')
            # GMR: Adding refs to the planet dict
            pkeys = [pk for pk in plist if '_' not in pk]
            for pk in pkeys:
                if out['starID'][thistarget][thisplanet][pk]: addme = ref_pl
                else: addme = ''
                out['starID'][thistarget][thisplanet][pk+'_ref'].append(addme)
                pass
            # GMR: Adding refs to the stellar dict
            skeys = [sk for sk in matchlist
                     if (sk not in banlist) and (sk not in plist) and ('_' not in sk)]
            for sk in skeys:
                if out['starID'][thistarget][sk]: addme = ref_st
                else: addme = ''
                if not out['starID'][thistarget][sk+'_ref']:
                    out['starID'][thistarget][sk+'_ref'].append(addme)
                    pass
                pass
            # GMR: Adding units to the stellar dict
            out['starID'][thistarget]['R*_units'].append('[Rsun]')
            out['starID'][thistarget]['M*_units'].append('[Msun]')
            out['starID'][thistarget]['T*_units'].append('[K]')
            out['starID'][thistarget]['L*_units'].append('[Lsun]')
            out['starID'][thistarget]['LOGG*_units'].append('log10[cm.s-2]')
            out['starID'][thistarget]['RHO*_units'].append('[g.cm-3]')
            out['starID'][thistarget]['AGE*_units'].append('[Gyr]')
            out['starID'][thistarget]['Hmag_units'].append('[mag]')
            out['starID'][thistarget]['dist_units'].append('[pc]')
            out['starID'][thistarget]['spTyp_units'].append('')
            # spectral type doesn't have uncertainties. fill in here by hand
            out['starID'][thistarget]['spTyp_uperr'].append('')
            out['starID'][thistarget]['spTyp_lowerr'].append('')
            merged = True
            pass
        pass

    # AUTOFILL WITH NEXSCI FULL TABLE, INCLUDING NON-DEFAULTS ----------------------------
    response = ident['nexsciFulltable']
    header = response[0].split(',')
    matchkey = translatekeys(header)
    for line in response:
        elem = line.split(',')
        elem = clean_elems(elem)
        if len(elem) != len(header):
            trgt = 'YOLO'
            thisplanet = 'CARPEDIEM'
            pass
        else:
            trgt = elem[header.index('hostname')]
            thisplanet = elem[header.index('pl_letter')]
            pass
        if trgt in targetID:
            ref_pl = elem[header.index('pl_refname')]
            ref_pl = ref_pl.split('</a>')[0]
            ref_pl = ref_pl.split('target=ref>')[-1]
            ref_pl = ref_pl.strip()
            ref_st = elem[header.index('st_refname')]
            ref_st = ref_st.split('</a>')[0]
            ref_st = ref_st.split('target=ref>')[-1]
            ref_st = ref_st.strip()
            numdate = elem[header.index('rowupdate')]
            strnum = ''
            for n in re.split('-|:| ', numdate): strnum = strnum+n.strip()
            numdate = float(strnum)
            for addme, key in zip(elem, matchkey):
                if key not in banlist:
                    refkey = key.split('_')[0]
                    if key in plist:
                        out['starID'][thistarget][thisplanet][key].append(addme)
                        if refkey==key:
                            out['starID'][thistarget][thisplanet][refkey+'_ref'].append(ref_pl)
                        pass
                    else:
                        refkey = key.split('_')[0]
                        out['starID'][thistarget][key].append(addme)
                        if refkey==key:
                            out['starID'][thistarget][refkey+'_ref'].append(ref_st)
                            pass
                        # again, fill in spectral type have uncertainties by hand
                        if key=='spTyp':
                            out['starID'][thistarget][key+'_uperr'].append('')
                            out['starID'][thistarget][key+'_lowerr'].append('')
                            pass
                        pass
                    pass
                pass
            merged = True
            pass
        pass

    # FINALIZE OUTPUT ------------------------------------------------
    if merged:
        candidates = ['b','c','d','e','f','g','h','i','j','k','l',
                      'm','n','o','p','q','r','s','t','u','v','w',
                      'x','y','z']
        plist = []
        for letter in candidates:
            if letter in out['starID'][thistarget].keys():
                plist.append(letter)
                pass
            pass
        out['starID'][thistarget]['planets'] = plist
        out['candidates'].extend(candidates)
        out['starkeys'].extend(sorted(skeys))
        out['planetkeys'].extend(sorted(pkeys))
        out['exts'].extend(['_uperr', '_lowerr', '_units', '_ref'])
        out['STATUS'].append(True)
        pass
    status = solved and merged
    return status

def clean_elem(elem):
    '''remove formatting on a single element from TAP API'''
    if not elem: return elem
    if elem[0] == '"' and elem[-1] == '"': return elem[1:-1]
    return elem

def clean_elems(elems):
    '''removes formatting symbols and prepares values for saving'''
    return [clean_elem(elem) for elem in elems]

def translatekeys(header):
    '''GMR: Translate NEXSCI keywords into Excalibur ones'''
    matchlist = []
    for key in header:
        thiskey = key
        if 'pl_hostname' == thiskey: xclbrkey = 'star'
        elif 'hostname' == thiskey: xclbrkey = 'star'
        elif 'pl_letter' == thiskey: xclbrkey = 'planet'
        elif 'pl_reflink' == thiskey: xclbrkey = 'ref_pl'
        elif 'rowupdate' == thiskey: xclbrkey = 'update'
        elif 'pl_def_refname' == thiskey: xclbrkey = 'ref_pl'
        elif 'pl_refname' == thiskey: xclbrkey = 'ref_pl'
        elif 'pl_pnum' == thiskey: xclbrkey = 'np'
        elif 'sy_pnum' == thiskey: xclbrkey = 'np'
        elif 'pl_orbper' == thiskey: xclbrkey = 'period'
        elif 'pl_orbpererr1' == thiskey: xclbrkey = 'period_uperr'
        elif 'pl_orbpererr2' == thiskey: xclbrkey = 'period_lowerr'
        elif 'pl_orbperreflink' == thiskey: xclbrkey = 'period_ref'
        elif 'pl_orbsmax' == thiskey: xclbrkey = 'sma'
        elif 'pl_smax' == thiskey: xclbrkey = 'sma'
        elif 'pl_orbsmaxerr1' == thiskey: xclbrkey = 'sma_uperr'
        elif 'pl_smaxerr1' == thiskey: xclbrkey = 'sma_uperr'
        elif 'pl_orbsmaxerr2' == thiskey: xclbrkey = 'sma_lowerr'
        elif 'pl_smaxerr2' == thiskey: xclbrkey = 'sma_lowerr'
        elif 'pl_smaxreflink' == thiskey: xclbrkey = 'sma_ref'
        elif 'pl_orbeccen' == thiskey: xclbrkey = 'ecc'
        elif 'pl_eccen' == thiskey: xclbrkey = 'ecc'
        elif 'pl_orbeccenerr1' == thiskey: xclbrkey = 'ecc_uperr'
        elif 'pl_eccenerr1' == thiskey: xclbrkey = 'ecc_uperr'
        elif 'pl_orbeccenerr2' == thiskey: xclbrkey = 'ecc_lowerr'
        elif 'pl_eccenerr2' == thiskey: xclbrkey = 'ecc_lowerr'
        elif 'pl_eccenreflink' == thiskey: xclbrkey = 'ecc_ref'
        elif 'pl_orbincl' == thiskey: xclbrkey = 'inc'
        elif 'pl_orbinclerr1' == thiskey: xclbrkey = 'inc_uperr'
        elif 'pl_orbinclerr2' == thiskey: xclbrkey = 'inc_lowerr'
        elif 'pl_bmassj' == thiskey: xclbrkey = 'mass'
        elif 'pl_bmassjerr1' == thiskey: xclbrkey = 'mass_uperr'
        elif 'pl_bmassjerr2' == thiskey: xclbrkey = 'mass_lowerr'
        elif 'pl_radj' == thiskey: xclbrkey = 'rp'
        elif 'pl_radjerr1' == thiskey: xclbrkey = 'rp_uperr'
        elif 'pl_radjerr2' == thiskey: xclbrkey = 'rp_lowerr'
        elif 'pl_radreflink' == thiskey: xclbrkey = 'rp_ref'
        elif 'pl_dens' == thiskey: xclbrkey = 'rho'
        elif 'pl_denserr1' == thiskey: xclbrkey = 'rho_uperr'
        elif 'pl_denserr2' == thiskey: xclbrkey = 'rho_lowerr'
        elif 'pl_eqt' == thiskey: xclbrkey = 'teq'
        elif 'pl_eqterr1' == thiskey: xclbrkey = 'teq_uperr'
        elif 'pl_eqterr2' == thiskey: xclbrkey = 'teq_lowerr'
        elif 'pl_eqtreflink' == thiskey: xclbrkey = 'teq_ref'
        elif 'pl_tranmid' == thiskey: xclbrkey = 't0'
        elif 'pl_tranmiderr1' == thiskey: xclbrkey = 't0_uperr'
        elif 'pl_tranmiderr2' == thiskey: xclbrkey = 't0_lowerr'
        elif 'pl_imppar' == thiskey: xclbrkey = 'impact'
        elif 'pl_impparerr1' == thiskey: xclbrkey = 'impact_uperr'
        elif 'pl_impparerr2' == thiskey: xclbrkey = 'impact_lowerr'
        elif 'pl_orblper' == thiskey: xclbrkey = 'omega'
        elif 'pl_orblpererr1' == thiskey: xclbrkey = 'omega_uperr'
        elif 'pl_orblpererr2' == thiskey: xclbrkey = 'omega_lowerr'
        elif 'pl_trandep' == thiskey: xclbrkey = 'trandepth'
        elif 'pl_trandeperr1' == thiskey: xclbrkey = 'trandepth_uperr'
        elif 'pl_trandeperr2' == thiskey: xclbrkey = 'trandepth_lowerr'
        elif 'pl_trandur' == thiskey: xclbrkey = 'trandur'
        elif 'pl_trandurerr1' == thiskey: xclbrkey = 'trandur_uperr'
        elif 'pl_trandurerr2' == thiskey: xclbrkey = 'trandur_lowerr'
        elif 'pl_insol' == thiskey: xclbrkey = 'insol'
        elif 'pl_insolerr1' == thiskey: xclbrkey = 'insol_uperr'
        elif 'pl_insolerr2' == thiskey: xclbrkey = 'insol_lowerr'
        elif 'pl_ratdor' == thiskey: xclbrkey = 'ars'
        elif 'pl_ratdorerr1' == thiskey: xclbrkey = 'ars_uperr'
        elif 'pl_ratdorerr2' == thiskey: xclbrkey = 'ars_lowerr'
        elif 'pl_ratror' == thiskey: xclbrkey = 'rprs'
        elif 'pl_ratrorerr1' == thiskey: xclbrkey = 'rprs_uperr'
        elif 'pl_ratrorerr2' == thiskey: xclbrkey = 'rprs_lowerr'
        elif 'st_teff' == thiskey: xclbrkey = 'T*'
        elif 'st_tefferr1' == thiskey: xclbrkey = 'T*_uperr'
        elif 'st_tefferr2' == thiskey: xclbrkey = 'T*_lowerr'
        elif 'st_teffreflink' == thiskey: xclbrkey = 'T*_ref'
        elif 'st_refname' == thiskey: xclbrkey = 'ref_st'
        elif 'st_mass' == thiskey: xclbrkey = 'M*'
        elif 'st_masserr1' == thiskey: xclbrkey = 'M*_uperr'
        elif 'st_masserr2' == thiskey: xclbrkey = 'M*_lowerr'
        elif 'st_rad' == thiskey: xclbrkey = 'R*'
        elif 'st_raderr1' == thiskey: xclbrkey = 'R*_uperr'
        elif 'st_raderr2' == thiskey: xclbrkey = 'R*_lowerr'
        elif 'st_radreflink' == thiskey: xclbrkey = 'R*_ref'
        elif 'st_lum' == thiskey: xclbrkey = 'L*'
        elif 'st_lumerr1' == thiskey: xclbrkey = 'L*_uperr'
        elif 'st_lumerr2' == thiskey: xclbrkey = 'L*_lowerr'
        elif 'st_logg' == thiskey: xclbrkey = 'LOGG*'
        elif 'st_loggerr1' == thiskey: xclbrkey = 'LOGG*_uperr'
        elif 'st_loggerr2' == thiskey: xclbrkey = 'LOGG*_lowerr'
        elif 'st_loggreflink' == thiskey: xclbrkey = 'LOGG*_ref'
        elif 'st_dens' == thiskey: xclbrkey = 'RHO*'
        elif 'st_denserr1' == thiskey: xclbrkey = 'RHO*_uperr'
        elif 'st_denserr2' == thiskey: xclbrkey = 'RHO*_lowerr'
        elif 'st_metfe' == thiskey: xclbrkey = 'FEH*'
        elif 'st_met' == thiskey: xclbrkey = 'FEH*'
        elif 'st_metfeerr1' == thiskey: xclbrkey = 'FEH*_uperr'
        elif 'st_meterr1' == thiskey: xclbrkey = 'FEH*_uperr'
        elif 'st_metfeerr2' == thiskey: xclbrkey = 'FEH*_lowerr'
        elif 'st_meterr2' == thiskey: xclbrkey = 'FEH*_lowerr'
        elif 'st_metratio' == thiskey: xclbrkey = 'FEH*_units'
        elif 'st_metreflink' == thiskey: xclbrkey = 'FEH*_ref'
        elif 'sy_hmag' == thiskey: xclbrkey = 'Hmag'
        elif 'sy_hmagerr1' == thiskey: xclbrkey = 'Hmag_uperr'
        elif 'sy_hmagerr2' == thiskey: xclbrkey = 'Hmag_lowerr'
        elif 'st_age' == thiskey: xclbrkey = 'AGE*'
        elif 'st_ageerr1' == thiskey: xclbrkey = 'AGE*_uperr'
        elif 'st_ageerr2' == thiskey: xclbrkey = 'AGE*_lowerr'
        elif 'sy_dist' == thiskey: xclbrkey = 'dist'
        elif 'sy_disterr1' == thiskey: xclbrkey = 'dist_uperr'
        elif 'sy_disterr2' == thiskey: xclbrkey = 'dist_lowerr'
        elif 'st_spectype' == thiskey: xclbrkey = 'spTyp'
        else: xclbrkey = None
        if xclbrkey is not None: matchlist.append(xclbrkey)
        pass
    if len(matchlist) != len(header):
        errstr = 'MISSING NEXSCI KEY MAPPING'
        log.warning('--< TARGET AUTOFILL: %s >--', errstr)
        pass
    return matchlist
# -------------- -----------------------------------------------------
# -- MAST -- ---------------------------------------------------------
def mast(selfstart, out, dbs, queryurl, mirror,
         alt=None,
         action='&action=Search&resolver=SIMBAD&radius=3.0',
         outfmt='&outputformat=CSV&max_records=100000',
         opt='&sci_aec=S'):
    '''Query and download from MAST'''
    found = False
    temploc = dawgie.context.data_stg
    targetID = list(selfstart['starID'].keys())
    querytarget = targetID[0].replace(' ', '+')
    targetID.extend(selfstart['starID'][targetID[0]]['aliases'])
    queryform = queryurl + querytarget + action + outfmt + opt
    with urlrequest.urlopen(queryform) as temp:
        framelist = temp.read().decode('utf-8').split('\n')
        pass
    namelist = []
    instlist = []
    for frame in framelist:
        row = frame.split(',')
        row = [v.strip() for v in row if v]
        if len(row) > 2:
            if row[1] in targetID:
                namelist.append(row[0].lower())
                instlist.append(row[8])
                found = True
                pass
            pass
        pass
    if found:
        tempdir = tempfile.mkdtemp(dir=temploc, prefix=targetID[0])
        ignname = []
        igninst = []
        for name, inst in zip(namelist, instlist):
            ext = '_flt.fits'
            if inst in ['WFC3', 'NICMOS']: ext = '_ima.fits'
            outfile = os.path.join(tempdir, name+ext)
            try: urlrequest.urlretrieve(mirror+name+ext, outfile)
            except(urllib.error.ContentTooShortError, urllib.error.URLError):
                log.warning('>-- Mirror1 %s %s %s', name, ext, 'NOT FOUND')
                try: urlrequest.urlretrieve(alt+name.upper()+ext.upper(), outfile)
                except (urllib.error.ContentTooShortError, urllib.error.HTTPError):
                    log.warning('>-- Mirror2 %s %s %s', name, ext, 'NOT FOUND')
                    ignname.append(name)
                    igninst.append(inst)
                    pass
                pass
            pass
        locations = [tempdir]
        new = dbscp(locations, dbs, out)
        shutil.rmtree(tempdir, True)
        pass
    ingested = found and new
    return ingested
# ---------- ---------------------------------------------------------
# -- DISK -- ---------------------------------------------------------
def disk(selfstart, out, diskloc, dbs):
    '''Query on disk data'''
    merge = False
    targetID = list(selfstart['starID'].keys())
    targets = trgedit.targetondisk.__doc__
    targets = targets.split('\n')
    targets = [t.strip() for t in targets if t.replace(' ', '')]
    locations = None
    for t in targets:
        parsedstr = t.split(':')
        parsedstr = [tt.strip() for tt in parsedstr]
        if parsedstr[0] in targetID:
            folders = parsedstr[1]
            folders = folders.split(',')
            folders = [f.strip() for f in folders]
            locations = [os.path.join(diskloc, fold) for fold in folders]
            pass
        pass

    if locations is None:
        log.warning('ADD data subdirectory name to list in target/edit.py!!')
        # exit('ADD it')
        pass

    # make sure that the data storage directory exists for this star
    # dang this gives 'permission denied' error for access to /proj/sdp
    # I guess I'll have to create all the new directories by command-line
    for loc in locations:
        if not os.path.exists(loc): os.makedirs(loc)
        pass

    if locations is not None: merge = dbscp(locations, dbs, out)
    return merge
# ---------- ---------------------------------------------------------
# -- DBS COPY -- -----------------------------------------------------
def dbscp(locations, dbs, out):
    '''Format data into SV'''
    copied = False
    imalist = None
    for loc in locations:
        if imalist is None:
            imalist = [os.path.join(loc, imafits) for imafits in os.listdir(loc)
                       if '.fits' in imafits]
            pass
        else:
            imalist.extend([os.path.join(loc, imafits) for imafits in os.listdir(loc)
                            if '.fits' in imafits])
            pass
        pass
    for fitsfile in imalist:
        filedict = {'md5':None, 'sha':None, 'loc':None, 'observatory':None,
                    'instrument':None, 'detector':None, 'filter':None, 'mode':None}
        m = subprocess.check_output(['md5sum', '-b', fitsfile])
        s = subprocess.check_output(['sha1sum', '-b', fitsfile])
        m = m.decode('utf-8').split(' ')
        s = s.decode('utf-8').split(' ')
        md5 = m[0]
        sha = s[0]
        filedict['md5'] = md5
        filedict['sha'] = sha
        filedict['loc'] = fitsfile
        with pyfits.open(fitsfile) as pf:
            mainheader = pf[0].header

            # HST
            if 'SCI' in mainheader.get('FILETYPE',''):
                keys = [k for k in mainheader.keys() if k != '']
                keys = list(set(keys))

                if 'TELESCOP' in keys: filedict['observatory'] = mainheader['TELESCOP']
                if 'INSTRUME' in keys: filedict['instrument'] = mainheader['INSTRUME']
                if 'DETECTOR' in keys:
                    filedict['detector'] = mainheader['DETECTOR'].replace('-', '.')
                if 'FILTER' in keys: filedict['filter'] = mainheader['FILTER']
                if ('SCAN_RAT' in keys) and (mainheader['SCAN_RAT'] > 0):
                    filedict['mode'] = 'SCAN'
                    pass
                else: filedict['mode'] = 'STARE'
                if 'FOCUS' in keys and filedict['detector'] is None:
                    filedict['detector'] = mainheader['FOCUS']
                    filedict['mode'] = 'IMAGE'
                    pass
                if filedict['instrument'] in ['STIS']:
                    filedict['filter'] = mainheader['OPT_ELEM']
                    pass

                out['name'][mainheader['ROOTNAME']] = filedict

            # Spitzer
            elif 'spitzer' in mainheader.get('TELESCOP').lower() and 'sci' in mainheader.get('EXPTYPE').lower():
                filedict['observatory'] = mainheader.get('TELESCOP')
                filedict['instrument'] = mainheader.get('INSTRUME')
                filedict['mode'] = mainheader.get('READMODE')
                filedict['detector'] = 'IR'
                if mainheader.get('CHNLNUM') == 1: filedict['filter'] = "36"
                elif mainheader.get('CHNLNUM') == 2: filedict['filter'] = "45"
                else: filedict['filter'] = mainheader.get('CHNLNUM')
                out['name'][str(mainheader.get('DPID'))] = filedict
                pass

            # JWST
            elif 'jwst' in mainheader.get('TELESCOP').lower():
                # Add filter for science frames only
                filedict['observatory'] = mainheader.get('TELESCOP').strip()
                filedict['instrument'] = mainheader.get('INSTRUME').strip()
                filedict['mode'] = mainheader.get('PUPIL').strip()  # maybe EXP_TYPE?
                filedict['detector'] = mainheader.get('DETECTOR').strip()
                filedict['filter'] = mainheader.get('FILTER').strip()
                out['name'][mainheader.get("FILENAME")] = filedict
                pass

        mastout = os.path.join(dbs, md5+'_'+sha)
        onmast = os.path.isfile(mastout)
        if not onmast:
            shutil.copyfile(fitsfile, mastout)
            os.chmod(mastout, int('0664', 8))
            pass
        copied = True
        out['STATUS'].append(True)
        pass

    return copied
# -------------- -----------------------------------------------------
