'''target core ds'''
# -- IMPORTS -- ------------------------------------------------------
import os
import shutil
import re
import tempfile
import subprocess
import logging; log = logging.getLogger(__name__)

import dawgie
import dawgie.db

import excalibur.target as trg
import excalibur.target.edit as trgedit

import astropy.io.fits as pyfits
import urllib.error
import urllib.request as urlrequest
# ------------- ------------------------------------------------------
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

    with urltrick(uri_full) as test: response = test.decode('utf-8')
    return response
# ----------------- --------------------------------------------------
# -- SCRAPE IDS -- ---------------------------------------------------
def scrapeids(ds:dawgie.Dataset, out, web, genIDs=True):
    '''Parses table from ipac exoplanetarchive'''
    targets = trgedit.targetlist.__doc__
    targets = targets.split('\n')
    targets = [t.strip() for t in targets if t.replace(' ', '').__len__() > 0]
    tn = os.environ.get('TARGET_NAME', None)
    if tn is not None:
        found_target_list = None
        for target in targets:
            if tn in target:
                found_target_list = target
        if found_target_list is None:
            # this is an ERROR.  the selected target should be in the target list
            # exit('ERROR: are you sure about that target?  it is not in the list')
            pass
        else:
            targets = [found_target_list]
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
    cols = "hostname,pl_letter,rowupdate,pl_refname,sy_pnum,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,pl_bmassj,pl_bmassjerr1,pl_bmassjerr2,pl_radj,pl_radjerr1,pl_radjerr2,pl_dens,pl_denserr1,pl_denserr2,pl_eqt,pl_eqterr1,pl_eqterr2,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_imppar,pl_impparerr1,pl_impparerr2,st_teff,st_tefferr1,st_tefferr2,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2,st_lum,st_lumerr1,st_lumerr2,st_logg,st_loggerr1,st_loggerr2,st_dens,st_denserr1,st_denserr2,st_met,st_meterr1,st_meterr2,sy_hmag,sy_hmagerr2,sy_hmagerr2"
    uri_ipac_query = {
        "select": cols,
        "from": 'ps',
        "where": "tran_flag = 1 and default_flag = 1",
        "order by": "pl_name",
        "format": "csv"
    }
    # First retrieval just gets default row for each target from table
    response = tap_query(web, uri_ipac_query)
    for line in response.split('\n'): out['nexscie'].append(line)
    # Second and third retrieval grab all rows for respective columns
    uri_ipac_query['select'] = "hostname,pl_letter,rowupdate,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_refname,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,pl_refname,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_refname,pl_radj,pl_radjerr1,pl_radjerr2,pl_refname,pl_eqt,pl_eqterr1,pl_eqterr2,pl_refname,st_teff,st_tefferr1,st_tefferr2,st_refname,st_rad,st_raderr1,st_raderr2,st_refname,st_lum,st_lumerr1,st_lumerr2,st_refname,st_logg,st_loggerr1,st_loggerr2,st_refname,st_met,st_meterr1,st_meterr2,st_refname,st_age,st_ageerr1,st_ageerr2,st_refname"
    uri_ipac_query['where'] = 'tran_flag = 1'
    response = tap_query(web, uri_ipac_query)
    for line in response.split('\n'): out['nexscic'].append(line)
    uri_ipac_query['select'] = "hostname,pl_letter,pl_refname,rowupdate,pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,st_logg,st_loggerr1,st_loggerr2,st_met,st_meterr1,st_meterr2"
    response = tap_query(web, uri_ipac_query)
    for line in response.split('\n'): out['nexscix'].append(line)
    out['STATUS'].append(True)
    ds.update()
    return
# ---------------- ---------------------------------------------------
# -- CREATE FILTERS -- -----------------------------------------------
def createfltrs(out):
    '''Create filter name'''
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
    '''
    return dawgie.VERSION(1,2,0)

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
    failure = ['target not resolved, continue\n\n', 'no rows found\n\n']
    with urlrequest.urlopen(queryform).read() as temp:
        framelist = temp.decode('utf-8')
        pass
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
    # AUTOFILL WITH NEXSCI EXOPLANET TABLE ---------------------------
    merged = False
    targetID.extend(ident['starID'][thistarget]['aliases'])
    response = ident['nexscie']
    header = response[0].split(',')
    # list all keys contained in csv
    # using tuples of form (a, b) where a is the key name
    # and then b are the other types of columns for that key
    # that are included
    matchlist = [('star', []), ('planet', []),
                 ('update', []), ('ref', []), ('np', []),
                 ('period', ['err']), ('sma', ['err']),
                 ('ecc', ['err']), ('inc', ['err']),
                 ('mass', ['err']), ('rp', ['err']),
                 ('rho', ['err']), ('teq', ['err']),
                 ('t0', ['err']), ('impact', ['err']),
                 ('T*', ['err']), ('M*', ['err']),
                 ('R*', ['err']), ('L*', ['err']),
                 ('LOGG*', ['err']),('RHO*', ['err']),
                 ('FEH*', ['err']),
                 ('Hmag', ['err'])]
    banlist = ['star', 'planet', 'update', 'ref', 'np']
    plist = ['period', 'period_uperr', 'period_lowerr', 'period_ref',
             'sma', 'sma_uperr', 'sma_lowerr', 'sma_ref',
             'ecc', 'ecc_uperr', 'ecc_lowerr', 'ecc_ref',
             'inc', 'inc_uperr', 'inc_lowerr', 'inc_ref',
             'mass', 'mass_uperr', 'mass_lowerr', 'mass_ref',
             'rp', 'rp_uperr', 'rp_lowerr', 'rp_ref',
             'rho', 'rho_uperr', 'rho_lowerr', 'rho_ref',
             'teq', 'teq_uperr', 'teq_lowerr', 'teq_ref',
             't0', 't0_uperr', 't0_lowerr', 't0_ref',
             'impact', 'impact_uperr', 'impact_lowerr', 'impact_ref']
    for line in response:
        elem = line.split(',')
        elem = clean_elems(elem)  # remove things such as bounding quotation marks
        if elem[0] in targetID:
            out['starID'][thistarget]['PLANETS #'] = elem[4]
            if elem[1] not in out['starID'][thistarget].keys():
                out['starID'][thistarget][elem[1]] = {}
                if 'CREATED' not in out['starID'][thistarget][elem[1]].keys():
                    out['starID'][thistarget][elem[1]]['CREATED'] = True
                    null_keys = ['date', 'date_ref']
                    for key in null_keys:
                        out['starID'][thistarget][elem[1]][key] = ['']
                    blank_keys = ['period_units', 'period_ref', 'sma_units',
                                  'sma_ref', 'ecc_units', 'ecc_ref', 'inc_units',
                                  'inc_ref', 'mass_units', 'mass_ref', 'rp_units',
                                  'rp_ref', 'rho_units', 'rho_ref', 'teq_units',
                                  'teq_ref', 't0_units', 't0_ref', 'impact_units',
                                  'impact_ref']
                    for key in blank_keys:
                        out['starID'][thistarget][elem[1]][key] = []
                    pass
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
                    out['starID'][thistarget]['FEH*_units'] = []
                    out['starID'][thistarget]['FEH*_ref'] = []
                    out['starID'][thistarget]['AGE*'] = ['']
                    out['starID'][thistarget]['AGE*_uperr'] = ['']
                    out['starID'][thistarget]['AGE*_lowerr'] = ['']
                    out['starID'][thistarget]['AGE*_units'] = ['']
                    out['starID'][thistarget]['AGE*_ref'] = ['']
                    out['starID'][thistarget]['Hmag_units'] = ['mag']
                    out['starID'][thistarget]['Hmag_ref'] = []
                    pass
                ref = elem[3]
                ref = ref.split('</a>')[0]
                ref = ref.split('target=ref>')[-1]
                ref = ref.strip()
                pass
            idx = 0
            for keymatch in matchlist:
                key = keymatch[0]
                match = keymatch[1]
                if key not in banlist:
                    if key in plist:
                        out_ref = out['starID'][thistarget][elem[1]]
                        pass
                    else:
                        out_ref = out['starID'][thistarget]
                        pass
                    pass
                    # initialize array if not already defined
                    keys = [key] + ([key+'_lowerr', key+'_uperr'] if 'err' in match else []) + \
                            ([key+'_units'] if 'units' in match else [])
                    for _key in keys:
                        if _key not in out_ref.keys():
                            out_ref[_key] = []
                    already_added = False
                    if out_ref[key] and out_ref[key][0]:
                        # ensure only first default value is added
                        already_added = True
                    # now add values in proper order
                    if not already_added:
                        out_ref[key].append(elem[idx])
                        if 'err' in match:
                            # ensure that system.validate will give +/-10% err
                            null_val = '' if key in plist else '0.0'
                            if elem[idx+1] or not elem[idx]:
                                out_ref[key+'_uperr'].append(elem[idx+1])
                            else: out_ref[key+'_uperr'].append(null_val)
                            if elem[idx+2] or not elem[idx]:
                                out_ref[key+'_lowerr'].append(elem[idx+2])
                            else: out_ref[key+'_lowerr'].append(null_val)
                        if 'units' in match:
                            units_idx = idx + (2 if 'err' in match else 0) + 1
                            out_ref[key+'_units'].append(elem[units_idx])
                pass
                # update to start of next key segment
                idx += 1 + (2 if 'err' in match else 0) + (1 if 'units' in match else 0)
            out['starID'][thistarget][elem[1]]['period_units'].append('[days]')
            out['starID'][thistarget][elem[1]]['sma_units'].append('[AU]')
            out['starID'][thistarget][elem[1]]['ecc_units'].append('[]')
            out['starID'][thistarget][elem[1]]['inc_units'].append('[degree]')
            out['starID'][thistarget][elem[1]]['mass_units'].append('[Jupiter mass]')
            out['starID'][thistarget][elem[1]]['rp_units'].append('[Jupiter radius]')
            out['starID'][thistarget][elem[1]]['rho_units'].append('[g.cm-3]')
            out['starID'][thistarget][elem[1]]['teq_units'].append('[K]')
            out['starID'][thistarget][elem[1]]['t0_units'].append('[Julian Days]')
            out['starID'][thistarget][elem[1]]['impact_units'].append('[R*]')
            pkeys = [pk for pk in plist if '_' not in pk]
            for pk in pkeys:
                if out['starID'][thistarget][elem[1]][pk][-1]: addme = ref
                else: addme = ''
                out['starID'][thistarget][elem[1]][pk+'_ref'].append(addme)
                pass
            skeys = [sk[0] for sk in matchlist
                     if sk[0] not in banlist and sk[0] not in plist]
            for sk in skeys:
                if out['starID'][thistarget][sk][-1]: addme = ref
                else: addme = ''
                if out['starID'][thistarget][sk+'_ref'] and \
                        out['starID'][thistarget][sk+'_ref'][0]:
                    # ensure only first defined row is added
                    continue
                out['starID'][thistarget][sk+'_ref'].append(addme)
                pass
            out['starID'][thistarget]['R*_units'].append('[Rsun]')
            out['starID'][thistarget]['M*_units'].append('[Msun]')
            out['starID'][thistarget]['T*_units'].append('[K]')
            out['starID'][thistarget]['L*_units'].append('[Lsun]')
            out['starID'][thistarget]['LOGG*_units'].append('log10[cm.s-2]')
            out['starID'][thistarget]['RHO*_units'].append('[g.cm-3]')
            out['starID'][thistarget]['AGE*_units'].append('[Gyr]')
            out['starID'][thistarget]['FEH*_units'].append('[Fe/H]')
            merged = True
            pass
        pass
    # AUTOFILL WITH NEXSCI EXTENDED TABLE ----------------------------
    response = ident['nexscix']
    header = response[0].split(',')
    matchkey = ['star', 'planet', 'ref', 'date', 'inc', 'inc_uperr', 'inc_lowerr',
                't0', 't0_uperr', 't0_lowerr',
                'LOGG*', 'LOGG*_uperr', 'LOGG*_lowerr',
                'FEH*', 'FEH*_uperr', 'FEH*_lowerr']
    for line in response:
        elem = line.split(',')
        elem = clean_elems(elem)
        if elem[0] in targetID:
            ref = elem[2]
            ref = ref.split('</a>')[0]
            ref = ref.split('target=ref>')[-1]
            ref = ref.strip()
            numdate = elem[3]
            strnum = ''
            for n in re.split('-|:| ', numdate): strnum = strnum+n.strip()
            numdate = float(strnum)
            for addme,key in zip(elem,matchkey):
                if key not in banlist and addme:
                    if key in plist:
                        test = out['starID'][thistarget][elem[1]][key]
                        refkey = key.split('_')[0]
                        tref = out['starID'][thistarget][elem[1]][refkey+'_ref']
                        updt = (key+'updt') in out['starID'][thistarget][elem[1]].keys()
                        if updt:
                            lupdt = out['starID'][thistarget][elem[1]][key+'updt']
                            if numdate > lupdt: test.append('')
                            pass
                        if len(test[-1]) < 1 and not test[0]:
                            test.append(addme)
                            tref.append(ref)
                            out['starID'][thistarget][elem[1]][key+'updt'] = numdate
                            pass
                        pass
                    else:
                        test = out['starID'][thistarget][key]
                        refkey = key.split('_')[0]
                        tref = out['starID'][thistarget][refkey+'_ref']
                        updt = (key+'updt') in out['starID'][thistarget]
                        if updt:
                            lupdt = out['starID'][thistarget][key+'updt']
                            if numdate > lupdt: test.append('')
                            pass
                        if len(test[-1]) < 1 and not test[0]:
                            test.append(addme)
                            tref.append(ref)
                            out['starID'][thistarget][key+'updt'] = numdate
                            pass
                        pass
                    pass
                pass
            merged = True
            pass
        pass
    # AUTOFILL WITH NEXSCI COMPOSITE TABLE ---------------------------
    response = ident['nexscic']
    header = response[0].split(',')
    matchkey = ['star', 'planet', 'date',
                'period', 'period_uperr', 'period_lowerr',
                'period_ref',
                'sma', 'sma_uperr', 'sma_lowerr', 'sma_ref',
                'ecc', 'ecc_uperr', 'ecc_lowerr', 'ecc_ref',
                'rp', 'rp_uperr', 'rp_lowerr', 'rp_ref',
                'teq', 'teq_uperr', 'teq_lowerr', 'teq_ref',
                'T*', 'T*_uperr', 'T*_lowerr', 'T*_ref',
                'R*', 'R*_uperr', 'R*_lowerr', 'R*_ref',
                'L*', 'L*_uperr', 'L*_lowerr', 'L*_ref',
                'LOGG*', 'LOGG*_uperr', 'LOGG*_lowerr', 'LOGG*_ref',
                'FEH*', 'FEH*_uperr', 'FEH*_lowerr','FEH*_ref',
                'AGE*', 'AGE*_uperr', 'AGE*_lowerr', 'AGE*_ref']
    for line in response:
        elem = line.split(',')
        elem = clean_elems(elem)
        if elem[0] in targetID:
            numdate = elem[2]
            strnum = ''
            for n in re.split('-|:| ', numdate): strnum = strnum+n.strip()
            numdate = float(strnum)
            for addme,key in zip(elem,matchkey):
                if '_ref' in key and addme:
                    addme = addme.split('</a>')[0]
                    addme = addme.split('target=ref>')[-1]
                    addme = addme.strip()
                    pass
                if 'target=_blank>Calculated Value' in addme:
                    addme = 'NEXSCI'
                    pass
                if key not in banlist and addme:
                    if key in plist:
                        test = out['starID'][thistarget][elem[1]][key]
                        updt = (key+'updt') in out['starID'][thistarget][elem[1]].keys()
                        if updt:
                            lupdt = out['starID'][thistarget][elem[1]][key+'updt']
                            if numdate > lupdt: test.append('')
                            pass
                        if len(test[-1]) < 1 and not test[0]:
                            test.append(addme)
                            out['starID'][thistarget][elem[1]][key+'updt'] = numdate
                            pass
                        pass
                    else:
                        test = out['starID'][thistarget][key]
                        updt = (key+'updt') in out['starID'][thistarget]
                        if updt:
                            lupdt = out['starID'][thistarget][key+'updt']
                            if numdate > lupdt: test.append('')
                            pass
                        if len(test[-1]) < 1 and not test[0]:
                            test.append(addme)
                            out['starID'][thistarget][key+'updt'] = numdate
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
    if not elem:
        return elem
    if elem[0] == '"' and elem[-1] == '"':
        return elem[1:-1]
    return elem

def clean_elems(elems):
    '''removes formatting symbols and prepares values for saving'''
    return [clean_elem(elem) for elem in elems]

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
