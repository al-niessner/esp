# -- IMPORTS -- ------------------------------------------------------
import excalibur.data.core as datcore
import excalibur.system.core as syscore

import pymc as pm
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d as itp
from ldtk.ldmodel import *
from ldtk import LDPSetCreator, BoxcarFilter
from pymc.distributions import Normal as pmnd
from pymc.distributions import Uniform as pmud
from pymc.distributions import TruncatedNormal as pmtnd
# ------------- ------------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def checksv(sv):
    valid = False
    errstring = None
    if sv['STATUS'][-1]: valid = True
    else: errstring = sv.name()+' IS EMPTY'
    return valid, errstring
# ----------------- --------------------------------------------------
# -- NORMALIZATION -- ------------------------------------------------
def norm(cal, tme, fin, ext, out, selftype, verbose=False, debug=False):
    normed = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    spectra = cal['data']['SPECTRUM']
    wave = cal['data']['WAVE']
    time = np.array(cal['data']['TIME'])
    disp = np.array(cal['data']['DISPERSION'])
    scanlen = np.array(cal['data']['SCANLENGTH'])
    vrange = cal['data']['VRANGE']
    arcsec2pix = datcore.dps(ext)
    scanlen = np.floor(scanlen/arcsec2pix)
    if len(tme[selftype]) > 0:
        for p in tme['data'].keys():
            out['data'][p] = {}
            rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
            ignore = np.array(tme['data'][p]['ignore'])
            nrmignore = ignore.copy()
            orbits = tme['data'][p]['orbits']
            dvisits = tme['data'][p]['dvisits']
            visits = tme['data'][p]['visits']
            phase = tme['data'][p]['phase']
            z = tme['data'][p]['z']
            zoot = z.copy()
            out['data'][p]['vrange'] = vrange
            out['data'][p]['visits'] = []
            out['data'][p]['dvisnum'] = []
            out['data'][p]['nspec'] = []
            out['data'][p]['wave'] = []
            out['data'][p]['time'] = []
            out['data'][p]['orbits'] = []
            out['data'][p]['dispersion'] = []
            out['data'][p]['z'] = []
            out['data'][p]['phase'] = []
            out['data'][p]['spect'] = []
            out['data'][p]['wavet'] = []
            out['data'][p]['photnoise'] = []
            out['data'][p]['trial'] = []
            out['data'][p]['vignore'] = []
            for v in tme['data'][p][selftype]:
                selv = (dvisits == v)
                # ORBIT REJECTION ------------------------------------
                if selftype in ['transit', 'phasecurve']:
                    select = (phase[selv] > 0.25) | (phase[selv] < -0.25)
                    vzoot = zoot[selv]
                    vzoot[select] = vzoot[select] + 666
                    zoot[selv] = vzoot
                    pass
                if selftype in ['eclipse']:
                    vzoot = zoot[selv]
                    select = (phase[selv] < 0.25) & (phase[selv] > -0.25)
                    vzoot[select] = vzoot[select] + 666
                    zoot[selv] = vzoot
                    pass
                ootplus = []
                ootpv = []
                ootminus = []
                ootmv = []
                for o in set(orbits[selv]):
                    selo = (orbits[selv] == o)
                    zorb = zoot[selv][selo]
                    if (min(abs(zorb)) > (1 + rpors)):
                        if np.median(zorb) < 0:
                            ootminus.append(o)
                            ootmv.append(np.median(zorb))
                            pass
                        else:
                            ootplus.append(o)
                            ootpv.append(np.median(zorb))
                            pass
                        pass
                    pass
                trash = []
                if len(ootplus) > 1:
                    keep = ootplus[ootpv.index(min(ootpv))]
                    trash.extend([i for i in ootplus if i != keep])
                    pass
                if len(ootminus) > 1:
                    keep = ootminus[ootmv.index(max(ootmv))]
                    trash.extend([i for i in ootminus if i != keep])
                    pass
                if len(trash) > 0:
                    for o in trash:
                        select = orbits[selv] == o
                        if selftype in ['transit', 'eclipse']:
                            vignore = ignore[selv]
                            vignore[select] = True
                            ignore[selv] = vignore
                            pass
                        vnrmignore = nrmignore[selv]
                        vnrmignore[select] = True
                        nrmignore[selv] = vnrmignore
                        pass
                    pass
                # OUT OF TRANSIT SELECTION ---------------------------
                selv = selv & (~ignore.astype(bool))
                selvoot = selv & (abs(z) > (1e0 + rpors)) & (~nrmignore)
                voots = [s for s in np.array(spectra)[selvoot]]
                vootw = [w for w in np.array(wave)[selvoot]]
                viss = [s for s in np.array(spectra)[selv]]
                visw = [w for w in np.array(wave)[selv]]
                if np.sum(selvoot) > 6:
                    wt, te = tplbuild(voots, vootw, vrange, disp[selvoot]*1e-4)
                    selfin = np.isfinite(te)
                    wt = np.array(wt)
                    te = np.array(te)
                    if np.sum(~selfin) > 0:
                        wt = wt[selfin]
                        te = te[selfin]
                        pass
                    ti = itp(wt, te, kind='linear', bounds_error=False, fill_value=np.nan)
                    nspectra = [s/ti(w) for s,w in zip(viss, visw)]
                    photnoise = [np.sqrt(s)/ti(w)/np.sqrt(l)
                                 for s,w,l in zip(viss, visw, scanlen[selv])]
                    check = np.array([np.nanstd(s) < 27e0*np.nanmedian(p)
                                      for s, p in zip(nspectra, photnoise)])
                    if np.sum(check) > 9:
                        vnspec = np.array(nspectra)[check]
                        nphotn = np.array(photnoise)[check]
                        visw = np.array(visw)
                        out['data'][p]['visits'].append(v)
                        out['data'][p]['dvisnum'].append(set(visits[selv]))
                        out['data'][p]['wavet'].append(wt)
                        out['data'][p]['spect'].append(te)
                        out['data'][p]['nspec'].append(vnspec)
                        out['data'][p]['wave'].append(visw[check])
                        out['data'][p]['time'].append(time[selv][check])
                        out['data'][p]['orbits'].append(orbits[selv][check])
                        out['data'][p]['dispersion'].append(disp[selv][check])
                        out['data'][p]['z'].append(z[selv][check])
                        out['data'][p]['phase'].append(phase[selv][check])
                        out['data'][p]['photnoise'].append(nphotn)
                        if debug:
                            plt.figure()
                            for w,s in zip(visw[check], vnspec):
                                select = (w > np.min(vrange)) & (w < np.max(vrange))
                                plt.plot(w[select], s[select], 'o')
                                pass
                            plt.ylabel('Normalized Spectra')
                            plt.xlabel('Wavelength [$\mu$m]')
                            plt.xlim(np.min(vrange), np.max(vrange))
                            plt.show()
                            pass
                        pass
                    else:
                        out['data'][p]['trial'].append('Variance Excess')
                        out['data'][p]['vignore'].append(v)
                        pass
                    pass
                else:
                    out['data'][p]['trial'].append('N/Visit < 7')
                    out['data'][p]['vignore'].append(v)
                    pass
                pass
            if verbose:
                for v, m in zip(out['data'][p]['vignore'], out['data'][p]['trial']):
                    print(v, m)
                    pass
                pass
            if len(out['data'][p]['visits']) > 0:
                normed = True
                out['STATUS'].append(True)
                pass
            pass
        pass
    return normed
# ------------------- ------------------------------------------------
# -- TEMPLATE BUILDER -- ---------------------------------------------
'''
Builds a spectrum template according to the peak in population
density per wavelength bins
'''
def tplbuild(spectra, wave, vrange, disp,
             verbose=False, debug=False):
    allspec = []
    for s in spectra.copy(): allspec.extend(s)
    allwave = []
    for w in wave.copy(): allwave.extend(w)
    allspec = np.array(allspec)
    allwave = np.array(allwave)
    vdisp = np.mean(disp)
    wavet = []
    template = []
    guess = [np.min(vrange) - vdisp/2e0]
    while guess[-1] < (max(vrange) + vdisp/2e0):
        dist = abs(allwave - guess[-1])
        selwave = list(allwave[dist < vdisp])
        selspec = list(allspec[dist < vdisp])
        seldist = list(dist[dist < vdisp])
        cluster = [selwave[seldist.index(min(seldist))]]
        cloud = [selspec[seldist.index(min(seldist))]]
        selwave.pop(seldist.index(min(seldist)))
        selspec.pop(seldist.index(min(seldist)))
        while len(cluster) < disp.size:
            seldist = list(abs(np.array(selwave) - np.mean(cluster)))
            cluster.append(selwave[seldist.index(min(seldist))])
            cloud.append(selspec[seldist.index(min(seldist))])
            selwave.pop(seldist.index(min(seldist)))
            selspec.pop(seldist.index(min(seldist)))
            pass
        wavet.append(np.mean(cluster))
        template.append(np.mean(cloud))
        guess.append(np.mean(cluster) + vdisp)
        pass
    if verbose or debug: pass
    return wavet, template
# ---------------------- ---------------------------------------------
# -- WHITE LIGHT CURVE -- --------------------------------------------
def whitelight(nrm, fin, out, selftype,
               chainlen=int(8e4), verbose=False, debug=False):
    wl = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = [p for p in nrm['data'].keys() if (len(nrm['data'][p]['visits']) > 0)]
    for p in planetloop:
        rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
        visits = nrm['data'][p]['visits']
        orbits = nrm['data'][p]['orbits']
        time = nrm['data'][p]['time']
        vrange = nrm['data'][p]['vrange']
        wave = nrm['data'][p]['wave']
        nspec = nrm['data'][p]['nspec']
        sep = nrm['data'][p]['z']
        phase = nrm['data'][p]['phase']
        wavet = nrm['data'][p]['wavet']
        disp = nrm['data'][p]['dispersion']
        photnoise = nrm['data'][p]['photnoise']
        out['data'][p] = {}
        out['data'][p]['nspec'] = nspec
        out['data'][p]['wave'] = wave
        out['data'][p]['visits'] = visits
        out['data'][p]['orbits'] = orbits
        allwhite = []
        allerrwhite = []
        flatminww = []
        flatmaxww = []
        for index, v in enumerate(visits):
            vdisp = np.mean(disp[index])*1e-4
            vwavet = np.array(wavet[index])
            white = []
            errwhite = []
            for w, s, e in zip(wave[index], nspec[index], photnoise[index]):
                select = ((w > (min(vwavet) + vdisp/2e0)) &
                          (w < (max(vwavet) - vdisp/2e0)))
                if True in select:
                    white.append(np.nanmedian(s[select]))
                    errwhite.append(np.nanmedian(e[select])/np.sqrt(np.nansum(select)))
                    pass
                else:
                    white.append(np.nan)
                    errwhite.append(np.nan)
                    pass
                flatminww.append(min(w[select]))
                flatmaxww.append(max(w[select]))
                pass
            allwhite.append(white)
            allerrwhite.append(errwhite)
            pass
        flaterrwhite = []
        for r in allerrwhite: flaterrwhite.extend(r)
        flaterrwhite = np.array(flaterrwhite)
        flatwhite = []
        for w in allwhite: flatwhite.extend(w)
        flatwhite = np.array(flatwhite)
        flatz = []
        for z in sep: flatz.extend(z)
        flatz = np.array(flatz)
        flatphase = []
        for ph in phase: flatphase.extend(ph)
        flatphase = np.array(flatphase)
        allwwmin = min(flatminww)
        allwwmax = max(flatmaxww)
        out['data'][p]['allwhite'] = allwhite
        out['data'][p]['phase'] = phase
        out['data'][p]['flatphase'] = flatphase
        # LIMB DARKENING ---------------------------------------------
        if selftype == 'transit':
            whiteld = createldgrid([allwwmin], [allwwmax], priors,
                                   segmentation=int(10), verbose=verbose, debug=debug)
            g1, g2, g3, g4 = whiteld['LD']
            wlmod = tldlc(abs(flatz), rpors, g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0])
            pass
        else:
            wlmod = tldlc(abs(flatz), rpors)
            whiteld = {}
            pass
        out['data'][p]['whiteld'] = whiteld
        # PLOT -------------------------------------------------------
        if verbose:
            plt.figure(figsize=(10, 6))
            for v in visits:
                index = visits.index(v)
                plt.plot(phase[index], allwhite[index], 'o', label=str(v))
                pass
            if len(visits) > 14: ncol = 2
            else: ncol = 1
            plt.plot(flatphase, wlmod, '^', label='M')
            plt.xlabel('Orbital Phase')
            plt.ylabel('Normalized Raw White Light Curve')
            plt.legend(bbox_to_anchor=(1 + 0.1*(ncol - 0.5), 0.5),
                       loc=5, ncol=ncol, mode='expand', numpoints=1,
                       borderaxespad=0., frameon=False)
            plt.tight_layout(rect=[0,0,(1 - 0.1*ncol),1])
            plt.show()
            pass
        # TTV --------------------------------------------------------
        ttv = []
        for index, v in enumerate(visits):
            select = (abs(sep[index]) < (1e0+rpors)) & (abs(sep[index]) > (1e0-rpors))
            if True in select: ttv.append(v)
            pass
        # PRIORS -----------------------------------------------------
        tmjd = priors[p]['t0']
        if tmjd > 2400000.5: tmjd -= 2400000.5
        period = priors[p]['period']
        ecc = priors[p]['ecc']
        inc = priors[p]['inc']
        smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
        flatoot = abs(flatz) > 1e0 + rpors
        ootstd = np.nanstd(flatwhite[flatoot])
        taurprs = 1e0/(rpors*1e-2)**2
        rprs = pmtnd('rprs', rpors, taurprs, rpors/2e0, 2e0*rpors)
        nodes = [rprs]
        ttrdur = np.arcsin((1e0+rpors)/smaors)
        trdura = priors[p]['period']*ttrdur/np.pi
        alltknot = np.empty(len(ttv), dtype=object)
        mintscale = []
        maxtscale = []
        for i, tvs in enumerate(time):
            mintscale.append(np.min(abs(np.diff(np.sort(tvs)))))
            for o in set(orbits[i]):
                maxtscale.append(np.max(tvs[orbits[i] == o]) -
                                 np.min(tvs[orbits[i] == o]))
            pass
        for i in range(len(ttv)):
            tautknot = 1e0/(3e0*np.min(mintscale))**2
            tknotmin = tmjd - np.max(maxtscale)/2e0
            tknotmax = tmjd + np.max(maxtscale)/2e0
            alltknot[i] = pmtnd('dtk%i' % ttv[i], tmjd, tautknot, tknotmin, tknotmax)
            pass
        nodes.extend(alltknot)
        if priors[p]['inc'] != 9e1:
            if priors[p]['inc'] > 9e1:
                lowinc = 9e1
                upinc = 9e1 + 18e1*np.arcsin(1e0/smaors)/np.pi
                pass
            if priors[p]['inc'] < 9e1:
                lowinc = 9e1 - 18e1*np.arcsin(1e0/smaors)/np.pi
                upinc = 9e1
                pass
            tauinc = 1e0/(priors[p]['inc']*1e-2)**2
            inc = pmtnd('inc', priors[p]['inc'], tauinc, lowinc, upinc)
            nodes.append(inc)
            pass
        # OOT ORBITS
        ootorbits = []
        ootoindex = []
        totalindex = 0
        for i in range(len(visits)):
            oot2append = []
            ooti2append = []
            for j in set(np.sort(orbits[i])):
                select = orbits[i] == j
                if abs(np.median(sep[i][select])) > (1e0 + rpors):
                    oot2append.append(j)
                    ooti2append.append(totalindex)
                    totalindex += 1
                    pass
                pass
            ootorbits.append(oot2append)
            ootoindex.append(ooti2append)
            pass
        # INSTRUMENT MODEL PRIORS ------------------------------------
        minoscale = np.log10(np.min(mintscale)*36e2*24)
        maxoscale = np.log10(np.max(maxtscale)*36e2*24)
        maxdelay = np.log10(5e0*np.max(maxtscale)*36e2*24)
        tauvs = 1e0/((1e-2/trdura)**2)
        tauvi = 1e0/(ootstd**2)
        allvslope = np.empty(len(visits), dtype=object)
        allvitcp = np.empty(len(visits), dtype=object)
        if len(visits) > 20:
            alloslope = np.empty(len(visits), dtype=object)
            allologtau = np.empty(len(visits), dtype=object)
            allologdelay = np.empty(len(visits), dtype=object)
            commonoim = True
            pass
        else:
            alloslope = np.empty(totalindex, dtype=object)
            allologtau = np.empty(totalindex, dtype=object)
            allologdelay = np.empty(totalindex, dtype=object)
            commonoim = False
            pass
        for i, vi in enumerate(visits):
            allvslope[i] = pmtnd('vslope%i' % vi, 0e0, tauvs, -3e-2/trdura, 3e-2/trdura)
            allvitcp[i] = pmtnd('vitcp%i' % vi, 1e0, tauvi,
                                1e0 - 3e0*ootstd, 1e0 + 3e0*ootstd)
            if commonoim:
                alloslope[i] = pmnd('oslope%i' % vi, 0e0, tauvs)
                allologtau[i] = pmud('ologtau%i' % vi, minoscale, maxoscale)
                allologdelay[i] = pmud('ologdelay%i' % vi, minoscale, maxdelay)
                pass
            pass
        if not commonoim:
            for i in range(totalindex):
                alloslope[i] = pmtnd('oslope%i' % i, 0e0, tauvs,
                                     -3e-2/trdura, 3e-2/trdura)
                allologtau[i] = pmud('ologtau%i' % i, minoscale, maxoscale)
                allologdelay[i] = pmud('ologdelay%i' % i, minoscale, maxdelay)
                pass
            pass
        nodes.extend(allvslope)
        nodes.extend(allvitcp)
        nodes.extend(alloslope)
        nodes.extend(allologtau)
        nodes.extend(allologdelay)
        selectfit = np.isfinite(flatwhite)
        # ORBITAL MODEL ----------------------------------------------
        @pm.deterministic
        def orbital(r=rprs, icln=inc, atk=alltknot,
                    avs=allvslope, avi=allvitcp,
                    aos=alloslope, aolt=allologtau, aold=allologdelay):
            out = []
            for v in visits:
                omt = time[visits.index(v)]
                if v in ttv: omtk = float(atk[ttv.index(v)])
                else: omtk = tmjd
                omz, pmph = datcore.time2z(omt, float(icln), omtk, smaors, period, ecc)
                lcout = tldlc(abs(omz), float(r), g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0])
                if commonoim:
                    imout = timlc(omt, orbits[visits.index(v)],
                                  vslope=float(avs[visits.index(v)]),
                                  vitcp=float(avi[visits.index(v)]),
                                  oslope=float(aos[visits.index(v)]),
                                  ologtau=float(aolt[visits.index(v)]),
                                  ologdelay=float(aold[visits.index(v)]))
                    pass
                else:
                    ooti = ootoindex[visits.index(v)]
                    oslopetable = [float(aos[i]) for i in ooti]
                    ologtautable = [float(aolt[i]) for i in ooti]
                    ologdelaytable = [float(aold[i]) for i in ooti]
                    imout = timlc(omt, orbits[visits.index(v)],
                                  vslope=float(avs[visits.index(v)]),
                                  vitcp=float(avi[visits.index(v)]),
                                  oslope=oslopetable,
                                  ologtau=ologtautable,
                                  ologdelay=ologdelaytable,
                                  ooto=ootorbits[visits.index(v)])
                    pass
                out.extend(lcout*imout)
                pass
            return np.array(out)[selectfit]
        tauwhite = 1e0/((np.median(flaterrwhite))**2)
        whitedata = pmnd('whitedata', mu=orbital, tau=tauwhite,
                         value=flatwhite[selectfit], observed=True)
        nodes.append(whitedata)
        allnodes = [n.__name__ for n in nodes if not n.observed]
        if verbose: print('MCMC nodes:', allnodes)
        model = pm.Model(nodes)
        mcmc = pm.MCMC(model)
        burnin = int(chainlen/2)
        mcmc.sample(chainlen, burn=burnin, progress_bar=verbose)
        if verbose: print('')
        mcpost = mcmc.stats()
        mctrace = {}
        for key in allnodes: mctrace[key] = mcmc.trace(key)[:]
        postlc = []
        postim = []
        postsep = []
        postphase = []
        postflatphase = []
        for v in visits:
            postt = time[visits.index(v)]
            if v in ttv: posttk = mcpost['dtk%i' % v]['quantiles'][50]
            else: posttk = tmjd
            if 'inc' in allnodes:
                postinc = mcpost['inc']['quantiles'][50]
                postz, postph = datcore.time2z(postt, postinc, posttk,
                                               smaors, period, ecc)
                pass
            else: postz, postph = datcore.time2z(postt, inc, posttk, smaors, period, ecc)
            postsep.extend(postz)
            postphase.append(postph)
            postflatphase.extend(postph)
            postlc.extend(tldlc(abs(postz), mcpost['rprs']['quantiles'][50],
                                g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0]))
            if commonoim:
                postim.append(timlc(postt, orbits[visits.index(v)],
                                    vslope=mcpost['vslope%i' % v]['quantiles'][50],
                                    vitcp=mcpost['vitcp%i' % v]['quantiles'][50],
                                    oslope=mcpost['oslope%i' % v]['quantiles'][50],
                                    ologtau=mcpost['ologtau%i' % v]['quantiles'][50],
                                    ologdelay=mcpost['ologdelay%i' % v]['quantiles'][50]))
                pass
            else:
                ooti = ootoindex[visits.index(v)]
                oslopetable = [mcpost['oslope%i' % i]['quantiles'][50]
                               for i in ooti]
                ologtautable = [mcpost['ologtau%i' % i]['quantiles'][50]
                                for i in ooti]
                ologdelaytable = [mcpost['ologdelay%i' % i]['quantiles'][50]
                                  for i in ooti]
                postim.append(timlc(postt, orbits[visits.index(v)],
                                    vslope=mcpost['vslope%i' % v]['quantiles'][50],
                                    vitcp=mcpost['vitcp%i' % v]['quantiles'][50],
                                    oslope=oslopetable,
                                    ologtau=ologtautable,
                                    ologdelay=ologdelaytable,
                                    ooto=ootorbits[visits.index(v)]))
                pass
            pass
        out['data'][p]['postlc'] = postlc
        out['data'][p]['postim'] = postim
        out['data'][p]['postsep'] = postsep
        out['data'][p]['postphase'] = postphase
        out['data'][p]['postflatphase'] = postflatphase
        out['data'][p]['mcpost'] = mcpost
        out['data'][p]['mctrace'] = mctrace
        out['data'][p]['ootorbits'] = ootorbits
        out['data'][p]['ootoindex'] = ootoindex
        out['data'][p]['commonoim'] = commonoim
        out['data'][p]['totalindex'] = totalindex
        out['STATUS'].append(True)
        wl = True
        if verbose:
            plt.figure(figsize=(10, 6))
            for iv, v in enumerate(visits):
                plt.plot(phase[iv], allwhite[iv], 'k+')
                plt.plot(postphase[iv], allwhite[iv]/postim[iv], 'o', label=str(v))
                pass
            if len(visits) > 14: ncol = 2
            else: ncol = 1
            plt.plot(postflatphase, postlc, '^', label='M')
            plt.xlabel('Orbital Phase')
            plt.ylabel('Normalized Post White Light Curve')
            plt.legend(bbox_to_anchor=(1 + 0.1*(ncol - 0.5), 0.5), loc=5, ncol=ncol,
                       mode='expand', numpoints=1, borderaxespad=0., frameon=False)
            plt.tight_layout(rect=[0,0,(1 - 0.1*ncol),1])
            plt.show()
            pass
        pass
    return wl
# ----------------------- --------------------------------------------
# -- TRANSIT LIMB DARKENED LIGHT CURVE -- ----------------------------
def tldlc(z, rprs, g1=0, g2=0, g3=0, g4=0,
          nint=int(8**2), verbose=False, debug=False):
    ldlc = np.zeros(z.size)
    xin = z.copy() - rprs
    xin[xin < 0e0] = 0e0
    xout = z.copy() + rprs
    xout[xout > 1e0] = 1e0
    select = xin > 1e0
    if (True in select): ldlc[select] = 1e0
    inldlc = []
    xint = np.linspace(1e0, 0e0, nint)
    znot = z.copy()[~select]
    xinnot = np.arccos(xin[~select])
    xoutnot = np.arccos(xout[~select])
    xrs = np.array([xint]).T*(xinnot - xoutnot) + xoutnot
    xrs = np.cos(xrs)
    diffxrs = np.diff(xrs, axis=0)
    extxrs = np.zeros((xrs.shape[0]+1, xrs.shape[1]))
    extxrs[1:-1, :] = xrs[1:,:] - diffxrs/2.
    extxrs[0, :] = xrs[0, :] - diffxrs[0]/2.
    extxrs[-1, :] = xrs[-1, :] + diffxrs[-1]/2.
    occulted = vecoccs(znot, extxrs, rprs)
    diffocc = np.diff(occulted, axis=0)
    si = vecistar(xrs, g1, g2, g3, g4)
    drop = np.sum(diffocc*si, axis=0)
    inldlc = 1. - drop
    ldlc[~select] = np.array(inldlc)
    return ldlc
# --------------------------------------- ----------------------------
# -- STELLAR EXTINCTION LAW -- ---------------------------------------
def vecistar(xrs, g1, g2, g3, g4):
    norm = (-g1/10e0 - g2/6e0 - 3e0*g3/14e0 - g4/4e0 + 5e-1)*2e0*np.pi
    select = xrs < 1e0
    mu = np.zeros(xrs.shape)
    mu[select] = (1e0 - xrs[select]**2)**(1e0/4e0)
    s1 = g1*(1e0 - mu)
    s2 = g2*(1e0 - mu**2)
    s3 = g3*(1e0 - mu**3)
    s4 = g4*(1e0 - mu**4)
    outld = (1e0 - (s1+s2+s3+s4))/norm
    return outld
# ---------------------------- ---------------------------------------
# -- STELLAR SURFACE OCCULTATION -- ----------------------------------
def vecoccs(z, xrs, rprs):
    out = np.zeros(xrs.shape)
    vecxrs = xrs.copy()
    selx = vecxrs > 0e0
    veczsel = np.array([z.copy()]*xrs.shape[0])
    veczsel[veczsel < 0e0] = 0e0
    select1 = (vecxrs <= rprs - veczsel) & selx
    select2 = (vecxrs >= rprs + veczsel) & selx
    select = (~select1) & (~select2) & selx
    zzero = veczsel == 0e0
    if (True in select1 & zzero):
        out[select1 & zzero] = np.pi*(np.square(vecxrs[select1 & zzero]))
    if (True in select2 & zzero):
        out[select2 & zzero] = np.pi*(rprs**2)
    if (True in select & zzero):
        out[select & zzero] = np.pi*(rprs**2)
    if (True in select1 & ~zzero):
        out[select1 & ~zzero] = np.pi*(np.square(vecxrs[select1 & ~zzero]))
    if (True in select2):
        out[select2 & ~zzero] = np.pi*(rprs**2)
    if (True in select & ~zzero):
        redxrs = vecxrs[select & ~zzero]
        redz = veczsel[select & ~zzero]
        s1 = ((np.square(redz) + np.square(redxrs) - rprs**2)/(2e0*redz*redxrs))
        s1[s1 > 1e0] = 1e0
        s2 = ((np.square(redz) + rprs**2 - np.square(redxrs))/(2e0*redz*rprs))
        s2[s2 > 1e0] = 1e0
        s3 = ((-redz + redxrs + rprs)*(redz + redxrs - rprs)*
              (redz - redxrs + rprs)*(redz + redxrs + rprs))
        zselect = s3 < 0e0
        if (True in zselect): s3[zselect] = 0e0
        out[select & ~zzero] = (np.square(redxrs)*np.arccos(s1) +
                                (rprs**2)*np.arccos(s2) - (5e-1)*np.sqrt(s3))
        pass
    return out
# --------------------------------- ----------------------------------
# -- CREATE LD GRID -- -----------------------------------------------
def createldgrid(minmu, maxmu, orbp,
                 ldmodel='nonlinear', phoenixmin=1e-2,
                 segmentation=int(10),
                 verbose=False, debug=False):
    tstar = orbp['T*']
    terr = np.sqrt(abs(orbp['T*_uperr']*orbp['T*_lowerr']))
    if terr < (3e0*tstar/1e2): terr = 3e0*tstar/1e2
    fehstar = orbp['FEH*']
    feherr = np.sqrt(abs(orbp['FEH*_uperr']*orbp['FEH*_lowerr']))
    loggstar = orbp['LOGG*']
    loggerr = np.sqrt(abs(orbp['LOGG*_uperr']*orbp['LOGG*_lowerr']))
    if verbose:
        print('Temperature', tstar, terr)
        print('Metallicity', fehstar, feherr)
        print('Surface Gravity', loggstar, loggerr)
        pass
    niter = int(len(minmu)/segmentation) + 1
    allcl = None
    allel = None
    out = {}
    avmu = [np.mean([mm, xm]) for mm, xm in zip(minmu, maxmu)]
    for i in np.arange(niter):
        if debug: print(str(i)+'/'+str(niter-1))
        loweri = i*segmentation
        upperi = (i+1)*segmentation
        if (i == (niter-1)): upperi = len(avmu)
        munm = 1e3*np.array(avmu[loweri:upperi])
        munmmin = 1e3*np.array(minmu[loweri:upperi])
        munmmax = 1e3*np.array(maxmu[loweri:upperi])
        filters = [BoxcarFilter(str(mue), mun, mux)
                   for mue, mun, mux in zip(munm, munmmin, munmmax)]
        sc = LDPSetCreator(teff=(tstar, terr),
                           logg=(loggstar, loggerr),
                           z=(fehstar, feherr),
                           filters=filters)
        ps = sc.create_profiles(nsamples=int(1e4))
        cl, el = ldx(ps._mu, ps._mean, ps._std,
                     mumin=phoenixmin, debug=verbose, model=ldmodel)
        if allcl is None: allcl = cl
        else: allcl = np.concatenate((allcl, cl), axis=0)
        if allel is None: allel = el
        else: allel = np.concatenate((allel, el), axis=0)
        pass
    allel[allel > 1.] = 0.
    allel[~np.isfinite(allel)] = 0.
    out['MU'] = avmu
    out['LD'] = allcl.T
    out['ERR'] = allel.T
    if verbose:
        plt.figure(figsize=(10, 6))
        for i in np.arange((allcl.T).shape[0]):
            thiscl = allcl.T[i]
            thisel = allel.T[i]
            plt.errorbar(avmu, thiscl, yerr=thisel)
            plt.plot(avmu, thiscl, '*', label='$\gamma$ %i' % i)
            pass
        plt.xlabel('Wavelength [$\mu$m]')
        plt.legend(bbox_to_anchor=(1, 0.5),
                   loc=5, ncol=1, mode='expand', numpoints=1,
                   borderaxespad=0., frameon=False)
        plt.tight_layout(rect=[0,0,0.9,1])
        plt.show()
        pass
    return out
# -------------------- -----------------------------------------------
# -- LDX -- ----------------------------------------------------------
def ldx(psmu, psmean, psstd,
        mumin=1e-6, debug=False, model='nonlinear'):
    mup = np.array(psmu).copy()
    prfs = np.array(psmean).copy()
    sprfs = np.array(psstd).copy()
    nwave = prfs.shape[0]
    select = (mup > mumin)
    fitmup = mup[select]
    fitprfs = prfs[:, select]
    fitsprfs = sprfs[:, select]
    cl = []
    el = []
    params = lm.Parameters()
    params.add('gamma1', value=1e-1)
    params.add('gamma2', value=5e-1)
    params.add('gamma3', value=1e-1)
    params.add('gamma4', expr='1 - gamma1 - gamma2 - gamma3')
    if debug: plt.figure()
    for iwave in np.arange(nwave):
        if model == 'linear':
            out = lm.minimize(lnldx, params, args=(fitmup,
                                                   fitprfs[iwave],
                                                   fitsprfs[iwave]))
            cl.append([out.params['gamma1'].value])
            el.append([out.params['gamma1'].stderr])
            pass
        if model == 'quadratic':
            out = lm.minimize(qdldx, params, args=(fitmup,
                                                   fitprfs[iwave],
                                                   fitsprfs[iwave]))
            cl.append([out.params['gamma1'].value,
                       out.params['gamma2'].value])
            el.append([out.params['gamma1'].stderr,
                       out.params['gamma2'].stderr])
            pass
        if model == 'nonlinear':
            out = lm.minimize(nlldx, params, args=(fitmup,
                                                   fitprfs[iwave],
                                                   fitsprfs[iwave]))
            cl.append([out.params['gamma1'].value,
                       out.params['gamma2'].value,
                       out.params['gamma3'].value,
                       out.params['gamma4'].value])
            el.append([out.params['gamma1'].stderr,
                       out.params['gamma2'].stderr,
                       out.params['gamma3'].stderr,
                       out.params['gamma4'].stderr])
            pass
        if debug:
            plt.plot(mup, prfs[iwave], 'k^')
            plt.errorbar(fitmup, fitprfs[iwave],
                         yerr=fitsprfs[iwave], ls='None')
            if model == 'linear':
                plt.plot(fitmup, lnldx(out.params, fitmup))
                pass
            if model == 'quadratic':
                plt.plot(fitmup, qdldx(out.params, fitmup))
                pass
            if model == 'nonlinear':
                plt.plot(fitmup, nlldx(out.params, fitmup))
                pass
            pass
        pass
    if debug:
        plt.ylabel('$I(\mu)$')
        plt.xlabel('$\mu$')
        plt.title(model)
        plt.show()
        pass
    return np.array(cl), np.array(el)
# --------- ----------------------------------------------------------
# -- LNLDX -- --------------------------------------------------------
def lnldx(params, x, data=None, weights=None):
    gamma1 = params['gamma1'].value
    model = LinearModel.evaluate(x, [gamma1])
    if data is None: return model
    if weights is None: return data - model
    return (data - model)/weights
# ----------- --------------------------------------------------------
# -- QDLDX -- --------------------------------------------------------
def qdldx(params, x, data=None, weights=None):
    gamma1 = params['gamma1'].value
    gamma2 = params['gamma2'].value
    model = QuadraticModel.evaluate(x, [gamma1, gamma2])
    if data is None: return model
    if weights is None: return data - model
    return (data - model)/weights
# ----------- --------------------------------------------------------
# -- NLLDX -- --------------------------------------------------------
def nlldx(params, x, data=None, weights=None):
    gamma1 = params['gamma1'].value
    gamma2 = params['gamma2'].value
    gamma3 = params['gamma3'].value
    gamma4 = params['gamma4'].value
    model = NonlinearModel.evaluate(x, [gamma1, gamma2, gamma3, gamma4])
    if data is None: return model
    if weights is None: return data - model
    return (data - model)/weights
# ----------- --------------------------------------------------------
# -- INSTRUMENT MODEL -- ---------------------------------------------
def timlc(vtime, orbits,
          vslope=0, vitcp=1e0, oslope=0, ologtau=0, ologdelay=0, ooto=None):
    xout = np.array(vtime) - np.mean(vtime)
    vout = vslope*xout + vitcp
    oout = np.ones(vout.size)
    orbinc = 0
    for o in set(np.sort(orbits)):
        select = orbits == o
        otime = xout[select] - np.mean(xout[select])
        if ooto is None:
            olin = oslope*otime + 1e0
            otimerp = (xout[select] - min(xout[select]))*(36e2)*(24e0)  # SECONDS
            orbramp = (1e0 - np.exp(-(otimerp + (1e1)**ologdelay)/((1e1)**ologtau)))
            oout[select] = orbramp*olin
            pass
        else:
            if o in ooto:
                olin = oslope[orbinc]*otime + 1e0
                otimerp = (xout[select] - min(xout[select]))*(36e2)*(24e0)  # SECONDS
                orbramp = (1e0 - np.exp(-(otimerp + (1e1)**ologdelay[orbinc])/
                                        ((1e1)**ologtau[orbinc])))
                oout[select] = orbramp*olin
                orbinc += 1
                pass
            else:
                olin = oslope[-1]*otime + 1e0
                otimerp = (xout[select] - min(xout[select]))*(36e2)*(24e0)  # SECONDS
                orbramp = (1e0 - np.exp(-(otimerp + (1e1)**ologdelay[-1])/
                                        ((1e1)**ologtau[-1])))
                oout[select] = orbramp*olin
                pass
            pass
        pass
    return vout*oout
# ---------------------- ---------------------------------------------
# -- SPECTRUM -- -----------------------------------------------------
def spectrum(fin, nrm, wht, out, selftype,
             chainlen=int(2e4), verbose=False, debug=False):
    exospec = False
    priors = fin['priors'].copy()
    ssc = syscore.ssconstants()
    planetloop = [p for p in nrm['data'].keys() if (len(nrm['data'][p]['visits']) > 0)]
    for p in planetloop:
        out['data'][p] = {'LD':[]}
        rpors = priors[p]['rp']/priors['R*']*ssc['Rjup/Rsun']
        smaors = priors[p]['sma']/priors['R*']/ssc['Rsun/AU']
        ttrdur = np.arcsin((1e0+rpors)/smaors)
        trdura = priors[p]['period']*ttrdur/np.pi
        wave = nrm['data'][p]['wavet']
        nspec = nrm['data'][p]['nspec']
        photnoise = nrm['data'][p]['photnoise']
        time = nrm['data'][p]['time']
        visits = nrm['data'][p]['visits']
        orbits = nrm['data'][p]['orbits']
        im = wht['data'][p]['postim']
        allz = wht['data'][p]['postsep']
        commonoim = wht['data'][p]['commonoim']
        ootorbits = wht['data'][p]['ootorbits']
        ootoindex = wht['data'][p]['ootoindex']
        totalindex = wht['data'][p]['totalindex']
        whiterprs = wht['data'][p]['mcpost']['rprs']['quantiles'][50]
        allwave = []
        allspec = []
        allim = []
        allpnoise = []
        for w, s, i, n in zip(nrm['data'][p]['wave'], nspec, im, photnoise):
            allwave.extend(w)
            allspec.extend(s)
            allim.extend(i)
            allpnoise.extend(n)
            pass
        allim = np.array(allim)
        allz = np.array(allz)
        disp = np.median([np.median(np.diff(w)) for w in wave])
        nbin = np.min([len(w) for w in wave])
        wavel = [np.min(w) for w in wave]
        wavec = np.arange(nbin)*disp + np.mean([np.max(wavel), np.min(wavel)])
        lwavec = wavec - disp/2e0
        hwavec = wavec + disp/2e0
        # LOOP OVER WAVELENGTH BINS ----------------------------------
        out['data'][p]['WB'] = []
        out['data'][p]['WBlow'] = []
        out['data'][p]['WBup'] = []
        out['data'][p]['ES'] = []
        out['data'][p]['ESerr'] = []
        out['data'][p]['LD'] = []
        out['data'][p]['MCPOST'] = []
        for wl, wh in zip(lwavec[1:-1], hwavec[1:-1]):
            out['data'][p]['WBlow'].append(wl)
            out['data'][p]['WBup'].append(wh)
            out['data'][p]['WB'].append(np.mean([wl, wh]))
            select = [(w > wl) & (w < wh) for w in allwave]
            data = np.array([np.median(d[s]) for d, s in zip(allspec, select)])
            dnoise = np.array([np.median(n[s]) for n, s in zip(allpnoise, select)])
            valid = np.isfinite(data)
            if selftype == 'transit':
                bld = createldgrid([wl], [wh], priors,
                                   segmentation=int(10), verbose=debug, debug=debug)
                g1, g2, g3, g4 = bld['LD']
                out['data'][p]['LD'].append([g1[0], g2[0], g3[0], g4[0]])
                model = tldlc(abs(allz), whiterprs,
                              g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0])
                pass
            else: model = tldlc(abs(allz), whiterprs)
            if debug:
                plt.figure()
                plt.title(str(int(1e3*np.mean([wl, wh])))+' nm')
                plt.plot(allz[valid], data[valid]/allim[valid], 'o')
                plt.plot(allz[valid], model[valid], '^')
                plt.xlabel('Separation [R*]')
                plt.show()
                pass
            # PRIORS -------------------------------------------------
            oot = (abs(allz) > (1e0 + whiterprs)) & valid
            iot = ~oot
            ootstd = np.std(data[oot])
            rprs = pmud('rprs', 0, 2e0*whiterprs)
            nodes = [rprs]
            tauvs = 1e0/((1e-2/trdura)**2)
            tauvi = 1e0/(ootstd**2)
            allvslope = np.empty(len(visits), dtype=object)
            allvitcp = np.empty(len(visits), dtype=object)
            if commonoim: alloslope = np.empty(len(visits), dtype=object)
            else: alloslope = np.empty(totalindex, dtype=object)
            allologtau = []
            allologdelay = []
            for i, v in enumerate(visits):
                allvslope[i] = pmtnd('vslope%i' % v, 0e0, tauvs,
                                     -3e-2/trdura, 3e-2/trdura)
                allvitcp[i] = pmtnd('vitcp%i' % v, 1e0, tauvi,
                                    1e0 - 3e0*ootstd, 1e0 + 3e0*ootstd)
                if commonoim:
                    alloslope[i] = pmnd('oslope%i' % v, 0e0, tauvs)
                    allologtau.append(wht['data'][p]['mcpost']['ologtau%i' % v]['quantiles'][50])
                    allologdelay.append(wht['data'][p]['mcpost']['ologdelay%i' % v]['quantiles'][50])
                    pass
                pass
            if not commonoim:
                for i in range(totalindex):
                    alloslope[i] = pmtnd('oslope%i' % i, 0e0, tauvs,
                                         -3e-2/trdura, 3e-2/trdura)
                    allologtau.append(wht['data'][p]['mcpost']['ologtau%i' % i]['quantiles'][50])
                    allologdelay.append(wht['data'][p]['mcpost']['ologdelay%i' % i]['quantiles'][50])
                    pass
                pass
            nodes.extend(allvslope)
            nodes.extend(allvitcp)
            nodes.extend(alloslope)
            # LIGHT CURVE MODEL --------------------------------------
            @pm.deterministic
            def lcmodel(r=rprs, avs=allvslope, avi=allvitcp, aos=alloslope):
                allimout = []
                for iv, v in enumerate(visits):
                    if commonoim:
                        imout = timlc(time[iv], orbits[iv], vslope=float(avs[iv]),
                                      vitcp=float(avi[iv]), oslope=float(aos[iv]),
                                      ologtau=float(allologtau[iv]),
                                      ologdelay=float(allologdelay[iv]))
                        pass
                    else:
                        ooti = ootoindex[iv]
                        oslopetable = [float(aos[i]) for i in ooti]
                        ologtautable = [float(allologtau[i]) for i in ooti]
                        ologdelaytable = [float(allologdelay[i]) for i in ooti]
                        imout = timlc(time[iv], orbits[iv],
                                      vslope=float(avs[iv]), vitcp=float(avi[iv]),
                                      oslope=oslopetable, ologtau=ologtautable,
                                      ologdelay=ologdelaytable, ooto=ootorbits[iv])
                        pass
                    allimout.extend(imout)
                    pass
                if selftype == 'transit':
                    out = tldlc(abs(allz), float(r),
                                g1=g1[0], g2=g2[0], g3=g3[0], g4=g4[0])
                    pass
                else: out = tldlc(abs(allz), float(r))
                out = out*np.array(imout)
                return out[valid]

            tauwbdata = 1e0/dnoise**2
            wbdata = pmnd('wbdata', mu=lcmodel,
                          tau=tauwbdata[valid], value=data[valid], observed=True)
            nodes.append(wbdata)
            allnodes = [n.__name__ for n in nodes if not n.observed]
            model = pm.Model(nodes)
            mcmc = pm.MCMC(model)
            burnin = int(chainlen/2)
            mcmc.sample(chainlen, burn=burnin, progress_bar=verbose)
            if verbose: print('')
            mcpost = mcmc.stats()
            out['data'][p]['ES'].append(mcpost['rprs']['quantiles'][50])
            out['data'][p]['ESerr'].append(mcpost['rprs']['standard deviation'])
            out['data'][p]['MCPOST'].append(mcpost)
            pass
        exospec = True
        out['STATUS'].append(True)
        pass
    if verbose:
        for p in out['data'].keys():
            vspectrum = np.array(out['data'][p]['ES'])
            specerr = np.array(out['data'][p]['ESerr'])
            specwave = np.array(out['data'][p]['WB'])
            specerr = abs(vspectrum**2 - (vspectrum + specerr)**2)
            vspectrum = vspectrum**2
            plt.figure(figsize=(8,6))
            plt.title(p)
            plt.errorbar(specwave, 1e2*vspectrum, fmt='.', yerr=1e2*specerr)
            plt.xlabel(str('Wavelength [$\mu m$]'))
            plt.ylabel(str('$(R_p/R_*)^2$ [%]'))
            plt.show()
            pass
        pass
    return exospec
# ----------- -- -----------------------------------------------------
