
import email.message
import logging; log = logging.getLogger(__name__)
import numpy
import smtplib

care_about = ['t0']

def _diff (vl):
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

def alert (asp:[(str,{str:{}})], known:[], table:[])->([],[],[]):
    changes,kwn,tab = [],[],[]
    for target,svs in sorted (asp, key=lambda t:t[0]):
        kwn.append (target)
        tab.append (svs['target.variations_of.parameters']['last'])

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
        # pylint: disable=bare-except
        try:
            msg = email.message.EmailMessage()
            msg.set_content ('\n'.join (changes))
            msg['Subject'] = 'Alert: target parameter changes detected'
            msg['From'] = 'do-not-reply@mentor.jpl.nasa.gov'
            msg['To'] = 'sdp@jpl.nasa.gov'
            s = smtplib.SMTP ('localhost')
            s.send_message (msg)
        except: log.exception ('Could not send alert email')
        pass
    return changes, kwn, tab

def regress (planet:{},rids:[],tl:[(int,{str:{}})])->({str:float},{str:[]},[]):
    for i,(rid,svs) in enumerate (tl):
        if rid in rids: break

        rids.insert (i,rid)
        svv = [t for t in svs['target.autofill.parameters']['starID'].values()][0]
        for p in svv['planets']:
            for ca in care_about:
                k = '_'.join ([p,ca])
                if k in planet: planet[k].insert (i,svv[p][ca][0])
                else: planet[k] = [svv[p][ca][0]]
                pass
            pass
        pass
    last = dict([(pp, _diff (vl)) for pp,vl in planet.items()])
    return last
