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
        z = np.array(transit_whitelight['data'][planet]['postsep'])
        ssc = syscore.ssconstants()
        rpors = priors[planet]['rp']/priors['R*']*ssc['Rjup/Rsun']

        out_transit_mask = abs(z) > (1e0 + rpors)
        all_data = []
        for d in transit_whitelight['data'][planet]['allwhite']: all_data.extend(d)
        out_transit_data = np.array(all_data)
        out_transit_data = out_transit_data[out_transit_mask]
        # std = np.nanstd(out_transit_data) GMR: UNUSED
        model = np.array(transit_whitelight['data'][planet]['postlc'])

        outliers = abs(all_data - model)  # removed the threshold > 2e0*std
        # outliers_num = np.sum(outliers) GMR UNUSED
        X_t = np.mean(outliers.tolist())
        test = X_t.reshape(1,-1)

        # import pdb; pdb.set_trace()
        # REMOVE ME

        clstrain = joblib.load(magicdir+'/cls_rf.pkl')
        rf_pred = list(clstrain.predict(test))
        out['data'][planet] = rf_pred
        pass
    out['STATUS'].append(True)
    return True
