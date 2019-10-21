
import numpy

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
    last = dict([(pp, _diff (vl)) for pp,vl in planet.items()])
    return last
