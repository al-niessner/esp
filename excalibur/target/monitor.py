'''target.monitor ds'''

# -- IMPORTS -- ------------------------------------------------------
import email.message
import logging; log = logging.getLogger(__name__)
import numpy
import smtplib
import math
# ------------- ------------------------------------------------------

care_about = ['t0']

def _diff (vl):
    '''_diff ds'''
    if 1 < len (vl):
        try:
            d = numpy.nan
            a = float (vl[0])
            d = 0
            b = float (vl[1])
            d = (a - b) / b * 100
        except ValueError: pass
    else: d = 0
    return d

def _outlier (vl):
    '''Finds whether first element of vl is within 5 sigma of other elems'''
    # 1 or more element of vl is an empty string there '' and that triggers bad things
    if 1 < len (vl):
        # Cheap fix attempt with try except
        try:
            vl = [float(v) for v in vl]

            if math.isnan(vl[0]): is_outlier = False
            else:
                vl_prev = numpy.array(vl[1:])
                vl_prev = vl_prev[~numpy.isnan(vl_prev)]  # clear all nans
                if len(vl_prev) > 1:
                    mean = numpy.mean(vl_prev)
                    std = numpy.std(vl_prev)
                    is_outlier = abs(vl[0]-mean)>5*std
                else: is_outlier = False
                pass
            pass
        except ValueError: is_outlier = False
    else: is_outlier = False  # only 1 or 0 elems; no outlier can exist
    return is_outlier

def alert (asp:{str:{str:{str:object}}}, known:[], table:[])->([],[],[]):
    '''alert ds'''
    changes,kwn,tab = [],[],[]
    for target in asp:
        kwn.append (target)
        tab.append (asp[target]['target.variations_of.parameters']['last'])

        if target in known:
            index = known.index (target)
            for pk in tab[-1].keys():
                if pk in table[index]:
                    lp_isnan = numpy.isnan (tab[-1][pk])
                    tb_isnan = numpy.isnan (table[index][pk])

                    if any ([lp_isnan and not tb_isnan,
                             not lp_isnan and tb_isnan,
                             (tab[-1][pk] != table[index][pk] and not
                              (lp_isnan and tb_isnan))]):
                        pks = pk.split ('_')
                        remark = (str (target) + '::' + str(pks[0]) +
                                  ' ' + '_'.join (pks[1:]) +
                                  ' has transitiond from {0} to {1}')
                        remark = remark.format (str(table[index][pk]),
                                                str(tab[-1][pk]))
                        changes.append (remark)
                        pass
                    pass
                pass
            pass
        pass
    changes.sort()

    if changes:
        # simplest for myriad of possibilities, pylint: disable=bare-except
        try:
            msg = email.message.EmailMessage()
            msg.set_content ('\n'.join (changes))
            msg['Subject'] = 'Alert: target parameter changes detected'
            msg['From'] = 'do-not-reply@excalibur.jpl.nasa.gov'
            msg['To'] = 'sdp@jpl.nasa.gov'
            s = smtplib.SMTP ('localhost')
            s.send_message (msg)
        except: log.exception ('Could not send alert email')
        pass
    return changes, kwn, tab

def regress (planet:{},rids:[],tl:{str:{str:{str:object}}})->({str:float},{str:[]},[]):
    '''regress ds'''
    for i,rid in enumerate (tl):
        if rid in rids: break

        rids.insert (i,rid)
        svv = list(tl[rid]['target.autofill.parameters']['starID'].values())
        svv = svv[0]
        for p in svv['planets']:
            for ca in care_about:
                k = '_'.join ([p,ca])
                if k in planet: planet[k].insert (i,svv[p][ca][0])
                else: planet[k] = [svv[p][ca][0]]
                pass
            pass
        pass
    # last = dict([(pp, _diff (vl)) for pp,vl in planet.items()])
    # I believe this might have been a debug thingy, commenting the print statement
    # print (planet.items())
    last = {pp:_diff (vl) for pp, vl in planet.items()}
    # outliers = dict([(pp, _outlier (vl)) for pp,vl in planet.items()])
    outliers = {pp:_outlier (vl) for pp, vl in planet.items()}
    return last,outliers
