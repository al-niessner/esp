'''classifier core ds'''
# -- IMPORTS -- ------------------------------------------------------
import os
import dawgie
import joblib
import numpy as np
import math

import excalibur
import excalibur.system.core as syscore
# ------------- ------------------------------------------------------
def predversion():
    '''predversion ds'''
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
        perc_rejected_val = len([i for i in vspectrum if np.isnan(i)])/len(vspectrum)*100
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

            sp_pred = sp_filter(perc_rejected_val, rsdpn)
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

def cpwlversion():
    '''cpwlversion ds'''
    return dawgie.VERSION(1,0,0)

# cpwl stands for count points whitelight.
def cpwl(transit_whitelight, priors, out):
    '''
    K. Mccarthy
    Counts points in transit curve
    '''

    for planet in transit_whitelight['data']:
        try:
            # test = transit_whitelight['data'][planet].keys()
            sep = (transit_whitelight['data'][planet]['postsep'])  # alternatively, could use orbital phase
            whitelight = (np.array(transit_whitelight['data'][planet]['allwhite']))
            pass
        except AttributeError:
            sep = []
            whitelight = []
            for p in transit_whitelight['data'][planet]:
                sep.extend(p['postsep'])
                whitelight.append(p['allwhite'])
                pass
            pass

        wl_flat = np.concatenate(whitelight).ravel().tolist()

        # each point is in [sep, wl] format.
        points = np.column_stack((np.array(sep), np.array(wl_flat)))
        transit_points_tot = []  # between contact points 1 and 4
        transit_points_full = []  # between contact points 2 and 3

        # get variables required for z equations
        r_p = priors[planet]['rp']  # planet radius (Jupiter radii)
        r_s = priors['R*']  # star radius (solar radii)

        sma = priors[planet]['sma']  # semi major axis (in AU) of planet's orbit
        inc = priors[planet]['inc']  # inclination (in degrees) of planet's orbit

        # get constants for unit conversions
        ssc = syscore.ssconstants()
        b = (((math.cos(math.radians(inc)))*sma)/r_s)*ssc['Rsun/AU']  # impact parameter
        radius_ratio = r_p/r_s*ssc['Rjup/Rsun']

        # use z equations to find postsep vals at the 4 contact points:
        # Note that z_1 = -1 * z_3, and z_2 = -1 * z_4.
        # Therefore, we can use abs(t) to find if a value t is between contact points 1 and 4 or between 2 and 3.
        z_3 = math.sqrt(((1 - radius_ratio)**2) - (b ** 2))  # postsep val at 3rd contact point
        z_4 = math.sqrt(((1 + radius_ratio)**2) - (b ** 2))  # postsep val at 4th contact point

        # count the number of points inside the transit
        for p in points:

            # between 1st and 4th contact point
            if abs(p[0]) <= z_4:
                transit_points_tot.append(p)

            # between 2nd and 3rd contact point
            if abs(p[0]) <= z_3:
                transit_points_full.append(p)

        # flag depending on number of transit points

        # color key
        flags = {
          0: 'green',
          1: 'yellow',
          2: 'red'
        }

        flag_val = 0
        flag_descrip = ""

        # No points between 1st and 4th contact.
        if len(transit_points_tot) == 0:
            flag_val = max(flag_val, 2)
            flag_descrip += "Insufficient points between 1st and 4th contact points."

        # Not many points between 1st and 4th contact.
        elif len(transit_points_tot) < 6:
            flag_val = max(flag_val, 1)
            flag_descrip += "Insufficient points between 1st and 4th contact points."

        else:
            flag_descrip += "Sufficient points between 1st and 4th contact points."

        # No points between 2nd and 3rd contact.
        if len(transit_points_full) == 0:

            # Not many points between 1st and 4th contact, either.
            if len(transit_points_tot) < 6:
                flag_val = max(flag_val, 2)
                flag_descrip += " No points between 2nd and 3rd contact points."

            # At least 6 points between 1st and 4th contact, so mark yellow instead of red.
            else:
                flag_val = max(flag_val, 1)
                flag_descrip += " No points between 2nd and 3rd contact points, but at least 6 points between 1st and 4th contact points."

        # Not many points between 2nd and 3rd contact.
        elif len(transit_points_full) < 5:
            flag_val = max(flag_val, 1)
            flag_descrip += " Insufficient points between 2nd and 3rd contact points."

        else:
            flag_descrip += " Sufficient points between 2nd and 3rd contact points."

        flag_color = flags[flag_val]

        # avoid overwriting existing data
        try:
            out['data'][planet]['count_points_wl'] = {}
        except KeyError:
            out['data'][planet] = {}
            out['data'][planet]['count_points_wl'] = {}

        out['data'][planet]['count_points_wl']['flag_color'] = flag_color
        out['data'][planet]['count_points_wl']['flag_descrip'] = flag_descrip
        out['data'][planet]['count_points_wl']['total'] = len(transit_points_tot)
        out['data'][planet]['count_points_wl']['full'] = len(transit_points_full)

        pass
    out['STATUS'].append(True)
    return True

def symwlversion():
    '''symwlversion ds'''
    return dawgie.VERSION(1,0,0)

# symwl stands for symmetry whitelight.
def symwl(transit_whitelight, priors, out):
    '''
    K. McCarthy
    Check symmetry of light curve
    '''

    for planet in transit_whitelight['data']:
        try:
            sep = (transit_whitelight['data'][planet]['postsep'])  # (alternatively, could use orbital phase)
            whitelight = (np.array(transit_whitelight['data'][planet]['allwhite']))
            pass
        except AttributeError:
            sep = []
            whitelight = []
            for p in transit_whitelight['data'][planet]:
                sep.extend(p['postsep'])
                whitelight.append(p['allwhite'])
                pass
            pass

        wl_flat = np.concatenate(whitelight).ravel().tolist()

        # each point is in [sep, wl] format.
        points = np.column_stack((np.array(sep), np.array(wl_flat)))
        transit_points_before_1 = []  # before contact point 1
        transit_points_after_4 = []  # after contact point 4

        # get variables required for z equations
        r_p = priors[planet]['rp']  # planet radius (Jupiter radii)
        r_s = priors['R*']  # star radius (solar radii)

        sma = priors[planet]['sma']  # semi major axis (in AU) of planet's orbit
        inc = priors[planet]['inc']  # inclination (in degrees) of planet's orbit

        # get constants for unit conversions
        ssc = syscore.ssconstants()
        b = (((math.cos(math.radians(inc)))*sma)/r_s)*ssc['Rsun/AU']  # impact parameter
        radius_ratio = r_p/r_s*ssc['Rjup/Rsun']

        # use z equations to find postsep vals at the 4 contact points:
        # Note that z_1 = -1 * z_3, and z_2 = -1 * z_4.
        # Therefore, we can use abs(t) to find if a value t is between contact points 1 and 4 or between 2 and 3.
        z_4 = math.sqrt(((1 + radius_ratio)**2) - (b ** 2))  # postsep val at 4th contact point
        z_1 = -1 * z_4

        # count the number of points outside of the transit on both sides
        for p in points:

            # outside 1st or 4th contact point
            if p[0] > z_4:
                transit_points_after_4.append(p)
            if p[0] < z_1:
                transit_points_before_1.append(p)

        # flag depending on number of pre- and post-transit points

        # color key
        flags = {
          0: 'green',
          1: 'yellow',
          2: 'red'
        }

        flag_val = 0
        flag_descrip = ""

        if len(transit_points_before_1) == 0 and len(transit_points_after_4) == 0:
            flag_val = max(flag_val, 2)
            flag_descrip = "Insufficient points before 1st contact point and insufficient points after 4th contact point."
        elif len(transit_points_before_1) == 0:
            flag_val = max(flag_val, 1)
            flag_descrip = "Insufficient points before 1st contact point."
        elif len(transit_points_after_4) == 0:
            flag_val = max(flag_val, 1)
            flag_descrip = "Insufficient points after 4th contact point."
        else:
            flag_val = max(flag_val, 0)
            flag_descrip = "Sufficient points before 1st contact point and after 4th contact point."

        flag_color = flags[flag_val]

        # try/except to avoid overwriting any existing data in out['data'][planet].
        try:
            out['data'][planet]['symmetry_wl'] = {}
        except KeyError:
            out['data'][planet] = {}
            out['data'][planet]['symmetry_wl'] = {}

        out['data'][planet]['symmetry_wl']['flag_color'] = flag_color
        out['data'][planet]['symmetry_wl']['flag_descrip'] = flag_descrip
        out['data'][planet]['symmetry_wl']['left'] = len(transit_points_before_1)
        out['data'][planet]['symmetry_wl']['right'] = len(transit_points_after_4)

        pass

    out['STATUS'].append(True)
    return True

def rsdmversion():
    '''RSDMversion ds'''
    return dawgie.VERSION(1,0,0)

def rsdm(transit_spectrum, out):
    '''
    K. McCarthy
    Calculate RSDM (in transit.spectrum) and flag target accordingly
    '''

    for planet in transit_spectrum['data']:

        try:
            spectrum_data = transit_spectrum['data'][planet]
            pass

        except AttributeError:
            pass

        if 'LCFIT' in spectrum_data:
            spec_rsdpn = [np.nanstd(i['residuals'])/i['dnoise']
                              for i in spectrum_data['LCFIT']]
            avg_rsdpn = np.nanmean(spec_rsdpn)

            # flag depending on mean RSDM
            # color key
            flags = {
              0: 'green',
              1: 'yellow',
              2: 'red'
            }

            flag_val = 0
            flag_descrip = ""

            green_upper_bound = 8.804
            yellow_upper_bound = 13.818

            if avg_rsdpn >= yellow_upper_bound:
                flag_val = 2
                flag_descrip = "Mean RSDM above threshold."
            if avg_rsdpn >= green_upper_bound:
                flag_val = 1
                flag_descrip = "Mean RSDM above threshold."
            else:
                flag_val = 0
                flag_descrip = "Mean RSDM is below threshold."

            flag_color = flags[flag_val]

            # avoid overwriting existing planet data
            try:
                out['data'][planet]['rsdm'] = {}
            except KeyError:
                out['data'][planet] = {}
                out['data'][planet]['rsdm'] = {}

            out['data'][planet]['rsdm']['mean_rsdm'] = avg_rsdpn
            out['data'][planet]['rsdm']['flag_color'] = flag_color
            out['data'][planet]['rsdm']['flag_descrip'] = flag_descrip

    out['STATUS'].append(True)
    return True

def perc_rejected_version():
    '''perc_rejected_version ds'''
    return dawgie.VERSION(1,0,0)

def perc_rejected(transit_spectrum, out):
    '''
    K. McCarthy
    Calculate the percentage of spectral channels rejected in cumulative spectrum distribution (in transit.spectrum) and flag target accordingly
    '''

    for planet in transit_spectrum['data']:

        try:
            spectrum_data = transit_spectrum['data'][planet]
            pass

        except AttributeError:
            pass

        # % spectral channels rejected in cumulative spectrum distribution
        planet_spectrum = np.array(spectrum_data['ES'])
        vspectrum = np.array(planet_spectrum)
        vspectrum = vspectrum**2
        perc_rejected_value = len([i for i in vspectrum if np.isnan(i)])/len(vspectrum)*100

        # flag depending on percent rejected
        # color key
        flags = {
          0: 'green',
          1: 'yellow',
          2: 'red'
        }

        flag_val = 0
        flag_descrip = ""

        green_upper_bound = 31.844
        yellow_upper_bound = 52.166

        if perc_rejected_value >= yellow_upper_bound:
            flag_val = 2
            flag_descrip = "Percent Rejected in Cumulative Spectrum Distribution is very high."
        elif perc_rejected_value >= green_upper_bound:
            flag_val = 1
            flag_descrip = "Percent Rejected in Cumulative Spectrum Distribution is high."
        else:
            flag_val = 0
            flag_descrip = "Percent Rejected in Cumulative Spectrum Distribution is below threshold."

        flag_color = flags[flag_val]

        # avoid overwriting existing planet data
        try:
            out['data'][planet]['perc_rejected'] = {}
        except KeyError:
            out['data'][planet] = {}
            out['data'][planet]['perc_rejected'] = {}

        out['data'][planet]['perc_rejected']['percent_rejected_value'] = perc_rejected_value
        out['data'][planet]['perc_rejected']['flag_color'] = flag_color
        out['data'][planet]['perc_rejected']['flag_descrip'] = flag_descrip

    out['STATUS'].append(True)
    return True

def median_error_version():
    '''median_error_version ds'''
    return dawgie.VERSION(1,0,0)

def median_error(data_calibration, out):
    '''
    K. McCarthy
    Calculate median error in data.calibration and flag target accordingly
    '''

    try:
        # get error distribution (adapted from esp/excalibur/data/core.py)
        data = data_calibration['data']
        spec = np.array([d for d,i in zip(data['SPECTRUM'], data['IGNORED']) if not i])
        wave = np.array([d for d,i in zip(data['WAVE'], data['IGNORED']) if not i])
        errspec = np.array([d for d,i in zip(data['SPECERR'], data['IGNORED']) if not i])
        vrange = data['VRANGE']
        allerr = []
        for s, e, w in zip(spec, errspec, wave):
            select = (w > vrange[0]) & (w < vrange[1])
            allerr.extend(e[select]/np.sqrt(s[select]))
            pass
        allerr = np.array(allerr)
        select = np.isfinite(allerr)
        allerr = allerr[select]
        allerr = allerr[allerr > 0.9]

        median_err = np.median(allerr)

        # flag depending on error distribution
        # color key
        flags = {
          0: 'green',
          1: 'yellow',
          2: 'red'
        }

        flag_val = 0
        flag_descrip = ""

        green_upper_bound = 2.5

        if median_err >= green_upper_bound:
            flag_val = 1
            flag_descrip = "Median error in data.calibration is above threshold."
        else:
            flag_val = 0
            flag_descrip = "Median error in data.calibration is below threshold."

        flag_color = flags[flag_val]

        out['data']['median_error'] = {}
        out['data']['median_error']['median_error_value'] = median_err
        out['data']['median_error']['flag_color'] = flag_color
        out['data']['median_error']['flag_descrip'] = flag_descrip

    except KeyError:
        pass

    out['STATUS'].append(True)
    return True
