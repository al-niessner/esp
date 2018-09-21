# -- IMPORTS -- ------------------------------------------------------
import os
import shutil
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
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- SCRAPE IDS -- ---------------------------------------------------
def scrapeids(ds:dawgie.Dataset, out, web, genIDs=True):
    targets = trgedit.targetlist.__doc__
    targets = targets.split('\n')
    targets = [t.strip() for t in targets if t.replace(' ', '').__len__() > 0]
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
    table = "table=exoplanets"
    cols = "&select=pl_hostname,pl_letter,rowupdate,pl_def_refname,pl_pnum,pl_orbper,pl_orbpererr1,pl_orbpererr2,pl_orbsmax,pl_orbsmaxerr1,pl_orbsmaxerr2,pl_orbeccen,pl_orbeccenerr1,pl_orbeccenerr2,pl_orbincl,pl_orbinclerr1,pl_orbinclerr2,pl_bmassj,pl_bmassjerr1,pl_bmassjerr2,pl_radj,pl_radjerr1,pl_radjerr2,pl_dens,pl_denserr1,pl_denserr2,pl_eqt,pl_eqterr1,pl_eqterr2,pl_tranmid,pl_tranmiderr1,pl_tranmiderr2,pl_imppar,pl_impparerr1,pl_impparerr2,st_teff,st_tefferr1,st_tefferr2,st_mass,st_masserr1,st_masserr2,st_rad,st_raderr1,st_raderr2,st_logg,st_loggerr1,st_loggerr2,st_dens,st_denserr1,st_denserr2,st_metfe,st_metfeerr1,st_metfeerr2,st_metratio"
    queryform = web+table+cols
    response = (urlrequest.urlopen(queryform)).read().decode('utf-8')
    for line in response.split('\n'): out['nexscie'].append(line)
    table2 = "table=compositepars"
    cols2 = "&select=fpl_hostname,fpl_letter,fpl_orbper,fpl_orbpererr1,fpl_orbpererr2,fpl_orbperreflink,fpl_smax,fpl_smaxerr1,fpl_smaxerr2,fpl_smaxreflink,fpl_eccen,fpl_eccenerr1,fpl_eccenerr2,fpl_eccenreflink,fpl_radj,fpl_radjerr1,fpl_radjerr2,fpl_radreflink,fpl_eqt,fpl_eqterr1,fpl_eqterr2,fpl_eqtreflink,fst_teff,fst_tefferr1,fst_tefferr2,fst_teffreflink,fst_rad,fst_raderr1,fst_raderr2,fst_radreflink,fst_logg,fst_loggerr1,fst_loggerr2,fst_loggreflink,fst_met,fst_meterr1,fst_meterr2,fst_metratio,fst_metreflink"
    queryform = web+table2+cols2
    response = (urlrequest.urlopen(queryform)).read().decode('utf-8')
    for line in response.split('\n'): out['nexscic'].append(line)
    table3 = "table=exomultpars"
    cols3 = "&select=mpl_hostname,mpl_letter,mpl_reflink,rowupdate,mpl_orbincl,mpl_orbinclerr1,mpl_orbinclerr2,mpl_tranmid,mpl_tranmiderr1,mpl_tranmiderr2,mst_logg,mst_loggerr1,mst_loggerr2,mst_metfe,mst_metfeerr1,mst_metfeerr2,mst_metratio"
    queryform = web+table3+cols3
    response = (urlrequest.urlopen(queryform)).read().decode('utf-8')
    for line in response.split('\n'): out['nexscix'].append(line)
    out['STATUS'].append(True)
    ds.update()
    return
# ---------------- ---------------------------------------------------
# -- CREATE FILTERS -- -----------------------------------------------
def createfltrs(out):
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
def autofill(ident, thistarget, out,
             queryurl="https://archive.stsci.edu/hst/search.php?target=",
             action="&action=Search&resolver=SIMBAD&radius=3.0",
             outfmt="&outputformat=CSV&max_records=100000",
             opt="&sci_aec=S"):
    out['starID'][thistarget] = ident['starID'][thistarget]
    targetID = [thistarget]
    # AUTOFILL WITH MAST QUERY ---------------------------------------
    solved = False
    querytarget = thistarget.replace(' ', '+')
    queryform = queryurl+querytarget+action+outfmt+opt
    failure = ['target not resolved, continue\n\n', 'no rows found\n\n']
    framelist = urlrequest.urlopen(queryform).read().decode('utf-8')
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
        pidlist = [pid for pid in set(pidlist)]
        aliaslist = [alias for alias in set(aliaslist)]
        out['starID'][thistarget]['aliases'].extend(aliaslist)
        out['starID'][thistarget]['PID'].extend(pidlist)
        solved = True
        out['STATUS'].append(True)
        pass
    # AUTOFILL WITH NEXSCI EXOPLANET TABLE ---------------------------
    merged = False
    targetID.extend(ident['starID'][thistarget]['aliases'])
    response = ident['nexscie']
    header = response[0].split(',')
    matchkey = ['star', 'planet', 'update','ref', 'np',
                'period', 'period_uperr', 'period_lowerr',
                'sma', 'sma_uperr', 'sma_lowerr',
                'ecc', 'ecc_uperr', 'ecc_lowerr',
                'inc', 'inc_uperr', 'inc_lowerr',
                'mass', 'mass_uperr', 'mass_lowerr',
                'rp', 'rp_uperr', 'rp_lowerr',
                'rho', 'rho_uperr', 'rho_lowerr',
                'teq', 'teq_uperr', 'teq_lowerr',
                't0', 't0_uperr', 't0_lowerr',
                'impact', 'impact_uperr', 'impact_lowerr',
                'T*', 'T*_uperr', 'T*_lowerr',
                'M*', 'M*_uperr', 'M*_lowerr',
                'R*', 'R*_uperr', 'R*_lowerr',
                'LOGG*', 'LOGG*_uperr', 'LOGG*_lowerr',
                'RHO*', 'RHO*_uperr', 'RHO*_lowerr',
                'FEH*', 'FEH*_uperr', 'FEH*_lowerr','FEH*_units']
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
        if elem[0] in targetID:
            out['starID'][thistarget]['PLANETS #'] = elem[4]
            if elem[1] not in out['starID'][thistarget].keys():
                out['starID'][thistarget][elem[1]] = {}
                if 'CREATED' not in out['starID'][thistarget][elem[1]].keys():
                    out['starID'][thistarget][elem[1]]['CREATED'] = True
                    out['starID'][thistarget][elem[1]]['date'] = ['']
                    out['starID'][thistarget][elem[1]]['date_ref'] = ['']
                    out['starID'][thistarget][elem[1]]['period_units'] = []
                    out['starID'][thistarget][elem[1]]['period_ref'] = []
                    out['starID'][thistarget][elem[1]]['sma_units'] = []
                    out['starID'][thistarget][elem[1]]['sma_ref'] = []
                    out['starID'][thistarget][elem[1]]['ecc_units'] = []
                    out['starID'][thistarget][elem[1]]['ecc_ref'] = []
                    out['starID'][thistarget][elem[1]]['inc_units'] = []
                    out['starID'][thistarget][elem[1]]['inc_ref'] = []
                    out['starID'][thistarget][elem[1]]['mass_units'] = []
                    out['starID'][thistarget][elem[1]]['mass_ref'] = []
                    out['starID'][thistarget][elem[1]]['rp_units'] = []
                    out['starID'][thistarget][elem[1]]['rp_ref'] = []
                    out['starID'][thistarget][elem[1]]['rho_units'] = []
                    out['starID'][thistarget][elem[1]]['rho_ref'] = []
                    out['starID'][thistarget][elem[1]]['teq_units'] = []
                    out['starID'][thistarget][elem[1]]['teq_ref'] = []
                    out['starID'][thistarget][elem[1]]['t0_units'] = []
                    out['starID'][thistarget][elem[1]]['t0_ref'] = []
                    out['starID'][thistarget][elem[1]]['impact_units'] = []
                    out['starID'][thistarget][elem[1]]['impact_ref'] = []
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
                    out['starID'][thistarget]['LOGG*_units'] = []
                    out['starID'][thistarget]['LOGG*_ref'] = []
                    out['starID'][thistarget]['RHO*_units'] = []
                    out['starID'][thistarget]['RHO*_ref'] = []
                    out['starID'][thistarget]['FEH*_ref'] = []
                    pass
                ref = elem[3]
                pass
            for i,key in enumerate(matchkey):
                if key not in banlist:
                    if key in plist:
                        if key not in out['starID'][thistarget][elem[1]].keys():
                            out['starID'][thistarget][elem[1]][key] = []
                            pass
                        out['starID'][thistarget][elem[1]][key].append(elem[i])
                        pass
                    else:
                        if key not in out['starID'][thistarget].keys():
                            out['starID'][thistarget][key] = []
                            pass
                        out['starID'][thistarget][key].append(elem[i])
                        pass
                    pass
                pass
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
            skeys = [sk for sk in matchkey
                     if sk not in banlist and sk not in plist and '_' not in sk]
            for sk in skeys:
                if out['starID'][thistarget][sk][-1]: addme = ref
                else: addme = ''
                out['starID'][thistarget][sk+'_ref'].append(addme)
                pass
            out['starID'][thistarget]['R*_units'].append('[Rsun]')
            out['starID'][thistarget]['M*_units'].append('[Msun]')
            out['starID'][thistarget]['T*_units'].append('[K]')
            out['starID'][thistarget]['LOGG*_units'].append('log10[cm.s-2]')
            out['starID'][thistarget]['RHO*_units'].append('[g.cm-3]')
            merged = True
            pass
        pass
    # AUTOFILL WITH NEXSCI EXTENDED TABLE ----------------------------
    response = ident['nexscix']
    header = response[0].split(',')
    matchkey = ['star', 'planet', 'ref', 'date',
                'inc', 'inc_uperr', 'inc_lowerr',
                't0', 't0_uperr', 't0_lowerr',
                'LOGG*', 'LOGG*_uperr', 'LOGG*_lowerr',
                'FEH*', 'FEH*_uperr', 'FEH*_lowerr','FEH*_units']
    for line in response:
        elem = line.split(',')
        if elem[0] in targetID:
            ref = elem[2]
            ref = ref.split('</a>')[0]
            ref = ref.split('target=ref>')[-1]
            numdate = elem[3]
            strnum = ''
            for n in numdate.split('-'): strnum = strnum+n.strip()
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
                        if len(test[-1]) < 1:
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
                        if len(test[-1]) < 1:
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
    matchkey = ['star', 'planet',
                'period', 'period_uperr', 'period_lowerr',
                'period_ref',
                'sma', 'sma_uperr', 'sma_lowerr', 'sma_ref',
                'ecc', 'ecc_uperr', 'ecc_lowerr', 'ecc_ref',
                'rp', 'rp_uperr', 'rp_lowerr', 'rp_ref',
                'teq', 'teq_uperr', 'teq_lowerr', 'teq_ref',
                'T*', 'T*_uperr', 'T*_lowerr', 'T*_ref',
                'R*', 'R*_uperr', 'R*_lowerr', 'R*_ref',
                'LOGG*', 'LOGG*_uperr', 'LOGG*_lowerr', 'LOGG*_ref',
                'FEH*', 'FEH*_uperr', 'FEH*_lowerr','FEH*_units',
                'FEH*_ref']
    for line in response:
        elem = line.split(',')
        if elem[0] in targetID:
            for addme,key in zip(elem,matchkey):
                if '_ref' in key and addme:
                    addme = addme.split('</a>')[0]
                    addme = addme.split('target=ref>')[-1]
                    pass
                if 'target=_blank>Calculated Value' in addme:
                    addme = 'NEXSCI'
                    pass
                if key not in banlist and addme:
                    if key in plist:
                        out['starID'][thistarget][elem[1]][key].append(addme)
                        pass
                    else: out['starID'][thistarget][key].append(addme)
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
# -------------- -----------------------------------------------------
# -- MAST -- ---------------------------------------------------------
def mast(selfstart, out, dbs, queryurl, mirror,
         alt=None,
         action='&action=Search&resolver=SIMBAD&radius=3.0',
         outfmt='&outputformat=CSV&max_records=100000',
         opt='&sci_aec=S'):
    found = False
    temploc = dawgie.context.data_stg
    targetID = [t for t in selfstart['starID'].keys()]
    querytarget = targetID[0].replace(' ', '+')
    targetID.extend(selfstart['starID'][targetID[0]]['aliases'])
    queryform = queryurl + querytarget + action + outfmt + opt
    framelist = urlrequest.urlopen(queryform)
    framelist = framelist.read().decode('utf-8').split('\n')
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
                log.log(31, '>-- Mirror1 %s %s %s', name, ext, 'NOT FOUND')
                try: urlrequest.urlretrieve(alt+name.upper()+ext.upper(), outfile)
                except (urllib.error.ContentTooShortError, urllib.error.HTTPError):
                    log.log(31, '>-- Mirror2 %s %s %s', name, ext, 'NOT FOUND')
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
    merge = False
    targetID = [t for t in selfstart['starID'].keys()]
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
            if 'SCI' in mainheader['FILETYPE']:
                keys = [k for k in mainheader.keys() if k != '']
                keys = [k for k in set(keys)]
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
                pass
            pass
        out['name'][mainheader['ROOTNAME']] = filedict
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
