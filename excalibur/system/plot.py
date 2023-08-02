'''ancillary plot dc'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import io
import matplotlib.pyplot as plt
import numpy as np
# ------------- ------------------------------------------------------
# -- PLOTTING FUNCTIONS-- --------------------------------------------
def rendertable(data, params, visitor:dawgie.Visitor)->None:
    '''
    Helper function to render a table using data corresponding to
    the passed parameters
    '''
    clabels = ['name', 'estimate', 'units', 'description', 'ref']
    table = visitor.add_table(clabels=clabels, rows=len(params))
    # display stellar estimates
    for i, param in enumerate(params):
        table.get_cell(i, 0).add_primitive(param)
        table.get_cell(i, 1).add_primitive(data[param])
        param_proc = [('_units', 'N/A'), ('_descr', 'No Description'),
                      ('_ref', 'N/A')]
        for idx, (suffix, msg) in enumerate(param_proc):
            if param+suffix in data:
                table.get_cell(i, 2+idx).add_primitive(data[param+suffix])
                pass
            else: table.get_cell(i, 2+idx).add_primitive(msg)
            pass
        pass
    return

def barplot(title, categories, counts, categories2, counts2, visitor):
    '''barplot ds'''
    myfig = plt.figure()
    plt.title(title.replace('log(','').replace(')',''), fontsize=18)
    plt.bar(categories, counts, color='khaki', zorder=1, label='everything')
    plt.bar(categories2, counts2, color='olive', zorder=2, label='Roudier et al. 2021')
    plt.ylabel('# of planets', fontsize=14)
    plt.xlabel('Spectral Type', fontsize=14)
    plt.legend()
    buf = io.BytesIO()
    myfig.savefig(buf, format='png')
    visitor.add_image('...', ' ', buf.getvalue())
    plt.close(myfig)
    return

def distrplot(paramName, values, values2, visitor, units=None):
    '''distrplot ds'''

    # clean up the values so that it's just an array of floats; no string error messages
    cleanvalues = []
    for value in values:
        if len(str(value)) > 0:
            if str(value)[0].isdigit() or str(value)[0]=='-':
                cleanvalues.append(value)
    cleanvalues = np.array(cleanvalues, dtype=float)
    cleanvalues2 = []
    for value2 in values2:
        if len(str(value2)) > 0:
            if str(value2)[0].isdigit() or str(value2)[0]=='-':
                cleanvalues2.append(value2)
    cleanvalues2 = np.array(cleanvalues2, dtype=float)

    # most histograms are better on a log scale
    if paramName=='luminosity':
        cleanvalues = np.log10(cleanvalues)
        cleanvalues2 = np.log10(cleanvalues2)
        paramName = 'log(L*)'
    elif paramName in ['M*', 'RHO*', 'L*',
                       'mass', 'sma', 'period',
                       'density','insolation','H','H_max',
                       'modulation','modulation_max','ZFOM','ZFOM_max',
                       'v_wind','rho_wind',
                       'M_loss_rate_wind','M_loss_rate_evap','Beta_rad']:
        cleanvalues = np.log10(cleanvalues)
        cleanvalues2 = np.log10(cleanvalues2)
        paramName = 'log('+paramName+')'

    # not sure why this is necessary.  why are some fields and entire params blank?
    # I guess it's the spTyp field, which seems to be missing from the resulting histograms
    if len(cleanvalues)==0: return
    if len(cleanvalues2)==0: return

    myfig = plt.figure()
    plt.title(paramName.replace('log(','').replace(')',''), fontsize=18)
    outlier_aware_hist(cleanvalues, cleanvalues2, *calculate_bounds(cleanvalues))
    plt.ylabel('# of planets', fontsize=14)
    # if units is None: plt.xlabel('Estimate')
    # else: plt.xlabel(f'Estimate [{units}]')
    if units is None: plt.xlabel(paramName, fontsize=14)
    else: plt.xlabel(paramName+f' [{units}]', fontsize=14)
    buf = io.BytesIO()
    myfig.savefig(buf, format='png')
    visitor.add_image('...', ' ', buf.getvalue())
    plt.close(myfig)
    return

def mad(data):
    '''mad ds'''
    median = np.nanmedian(data)
    diff = np.abs(data - median)
    mad_est = np.nanmedian(diff)
    return mad_est

def calculate_bounds(data, z_thresh=3.5):
    '''computes outlier cutoffs'''
    MAD = mad(data)
    median = np.nanmedian(data)
    const = z_thresh * MAD / 0.6745
    return (median - const, median + const)

def outlier_aware_hist(data, data2, lower=None, upper=None):
    '''note: code is originally from
    https://stackoverflow.com/questions/15837810/making-pyplot-hist-first-and-last-bins-include-outliers'''
    if not lower or lower < data.min():
        lower = data.min()
        lower_outliers = False
    else: lower_outliers = True

    if not upper or upper > data.max():
        upper = data.max()
        upper_outliers = False
    else: upper_outliers = True

    _, _, patches = plt.hist(data, range=(lower, upper), bins=15,
                             color='khaki', zorder=1, label='everything')

    plt.hist(data2, range=(lower, upper), bins=15,
             color='olive', zorder=2, label='Roudier et al. 2021')

    if lower_outliers:
        n_lower_outliers = (data < lower).sum()
        patches[0].set_height(patches[0].get_height() + n_lower_outliers)
        patches[0].set_facecolor('gold')
        patches[0].set_label(f'Lower outliers: ({data.min():.2f}, {lower:.2f})')

    if upper_outliers:
        n_upper_outliers = (data > upper).sum()
        patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
        patches[-1].set_facecolor('yellowgreen')
        patches[-1].set_label(f'Upper outliers: ({upper:.2f}, {data.max():.2f})')

    plt.legend()
    return
