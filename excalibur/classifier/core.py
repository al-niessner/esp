import os
import dawgie
import joblib
import numpy as np
import excalibur
import excalibur.system.core as syscore

def predversion():
    return dawgie.VERSION(2,0,2)

def predict(transit_whitelight, transit_spectrum, priors, out):
    '''
    V. Timmaraju and Hamsa S. Venkataram:
    Predicting the science plausibility of an exoplanet's lightcurve
    '''
    magicdir = os.path.join(excalibur.context['data_dir'], 'cls_models')
    # pr_thresh = 50
    rsdpn_thresh = 3.5
    for planet in transit_whitelight['data']:
        try:
            test = transit_whitelight['data'][planet].keys()
            z = np.array(transit_whitelight['data'][planet]['postsep'])
            classd = transit_whitelight['data'][planet]['allwhite']
            classim = transit_whitelight['data'][planet]['postim']
            model = np.array(transit_whitelight['data'][planet]['postlc'])
            pass
        except AttributeError:
            z = []
            classd = []
            classim = []
            model = []
            for p in transit_whitelight['data'][planet]:
                z.extend(p['postsep'])
                classd.append(p['allwhite'])
                classim.append(np.ones(p['allwhite'].size))
                model.extend(p['postlc'])
                pass
            z = np.array(z)
            model = np.array(model)
            pass
        # need to loop through len(transit_whitelight['data'][planet]) for each event in Spitzer
        ssc = syscore.ssconstants()
        rpors = priors[planet]['rp']/priors['R*']*ssc['Rjup/Rsun']
        in_transit = np.sum(abs(z) <= (1e0 + rpors))
        planet_spectrum = np.array(transit_spectrum['data'][planet]['ES'])
        vspectrum = np.array(planet_spectrum)
        vspectrum = vspectrum**2
        perc_rejected = len([i for i in vspectrum if np.isnan(i)])/len(vspectrum)*100
        if 'LCFIT' in transit_spectrum['data'][planet]:
            spec_rsdpn = [np.nanstd(i['residuals'])/i['dnoise'] for i in
                          transit_spectrum['data'][planet]['LCFIT']]
            rsdpn = np.mean(spec_rsdpn)
        else:
            rsdpn = rsdpn_thresh
        out_transit_mask = abs(z) > (1e0 + rpors)
        all_data = []
        for d, im in zip(classd, classim): all_data.extend(d/im)
        out_transit_data = np.array(all_data)
        out_transit_data = out_transit_data[out_transit_mask]
        # std = np.nanstd(out_transit_data) GMR: UNUSED
        # model = np.array(transit_whitelight['data'][planet]['postlc'])

        # GMR: Quick fix, please make sure that is what you wanted
        X_t = np.array([int(in_transit)])
        test = X_t.reshape(1,-1)
        if np.all(~np.isfinite(all_data)): fin_pred = ['All NAN input']
        else:
            # scaler = joblib.load(magicdir+'/cls_scaler.save')
            pca = joblib.load(magicdir+'/cls_pca.pkl')
            X_test = np.array(pca.transform(test))
            clstrain = joblib.load(magicdir+'/cls_rf.pkl')

            def sp_filter(pr, rsd):
                if pr <= 50 and rsd <= 3.5: val = 1
                else: val = -1
                return val

            sp_pred = sp_filter(perc_rejected, rsdpn)
            cls_pred = clstrain.predict(X_test)

            def pred(sp_p, c_p):
                if c_p == 1 and sp_p == 1: val = ['Scientifically Plausible']
                elif c_p == 1 and sp_p == -1: val = ['Caution']
                elif c_p == 0: val = ['Scientifically Implausible']
                return val

            fin_pred = pred(sp_pred, cls_pred)
            pass
        out['data'][planet] = {}
        out['data'][planet]['prediction'] = fin_pred[0]
        out['data'][planet]['allwhite'] = all_data
        out['data'][planet]['postlc'] = model
        out['data'][planet]['postsep'] = z
        pass
    out['STATUS'].append(True)
    return True
