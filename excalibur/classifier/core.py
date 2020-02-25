import os
import dawgie
import joblib
import numpy as np
import excalibur
import excalibur.system.core as syscore

def predversion():
    return dawgie.VERSION(1,0,0)

def predict(transit_whitelight, priors, out):
    '''
    V. Timmaraju and Hamsa S. Venkataram:
    Predicting the science plausibility of an exoplanet's lightcurve
    '''
    magicdir = os.path.join(excalibur.context['data_dir'], 'cls_models')

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

        out_transit_mask = abs(z) > (1e0 + rpors)
        all_data = []
        for d, im in zip(classd, classim): all_data.extend(d/im)
        out_transit_data = np.array(all_data)
        out_transit_data = out_transit_data[out_transit_mask]
        # std = np.nanstd(out_transit_data) GMR: UNUSED
        # model = np.array(transit_whitelight['data'][planet]['postlc'])

        outliers = abs(all_data - model)  # removed the threshold > 2e0*std
        # outliers_num = np.sum(outliers) GMR: UNUSED
        X_t = np.mean(outliers.tolist())
        test = X_t.reshape(1,-1)

        # import pdb; pdb.set_trace()
        # REMOVE ME

        if np.all(~np.isfinite(all_data)): rf_pred = ['All NAN input']
        else:
            clstrain = joblib.load(magicdir+'/cls_rf.pkl')
            rf_pred = list(clstrain.predict(test))
            rf_pred = ['Scientifically Plausible' if i==0 else 'Scientifically Implausible' for i in rf_pred]
            pass
        out['data'][planet] = {}
        out['data'][planet]['prediction'] = rf_pred[0]
        out['data'][planet]['allwhite'] = all_data
        out['data'][planet]['postlc'] = model
        out['data'][planet]['postsep'] = z
        pass
    out['STATUS'].append(True)
    return True
