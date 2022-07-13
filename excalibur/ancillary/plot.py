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

def barplot(title, categories, counts, visitor):
    '''barplot ds'''
    myfig = plt.figure()
    plt.title(title)
    plt.bar(categories, counts)
    plt.ylabel('Count')
    plt.xlabel('Type')
    buf = io.BytesIO()
    myfig.savefig(buf, format='png')
    visitor.add_image('...', ' ', buf.getvalue())
    plt.close(myfig)
    return

def distrplot(title, values, visitor, units=None):
    '''distrplot ds'''
    myfig = plt.figure()
    plt.title(title)
    outlier_aware_hist(np.array(values), *calculate_bounds(values))
    plt.ylabel('Count')
    if units is None: plt.xlabel('Estimate')
    else: plt.xlabel(f'Estimate [{units}]')
    buf = io.BytesIO()
    myfig.savefig(buf, format='png')
    visitor.add_image('...', ' ', buf.getvalue())
    plt.close(myfig)
    return

def mad(data):
    '''mad ds'''
    median = np.median(data)
    diff = np.abs(data - median)
    mad_est = np.median(diff)
    return mad_est

def calculate_bounds(data, z_thresh=3.5):
    '''computes outlier cutoffs'''
    MAD = mad(data)
    median = np.median(data)
    const = z_thresh * MAD / 0.6745
    return (median - const, median + const)

def outlier_aware_hist(data, lower=None, upper=None):
    '''note: code is taken with little modification from
    https://stackoverflow.com/questions/15837810/making-pyplot-hist-first-and-last-bins-include-outliers'''
    if not lower or lower < data.min():
        lower = data.min()
        lower_outliers = False
    else: lower_outliers = True

    if not upper or upper > data.max():
        upper = data.max()
        upper_outliers = False
    else: upper_outliers = True

    _, _, patches = plt.hist(data, range=(lower, upper), bins='auto')

    if lower_outliers:
        n_lower_outliers = (data < lower).sum()
        patches[0].set_height(patches[0].get_height() + n_lower_outliers)
        patches[0].set_facecolor('c')
        patches[0].set_label(f'Lower outliers: ({data.min():.2f}, {lower:.2f})')
        pass

    if upper_outliers:
        n_upper_outliers = (data > upper).sum()
        patches[-1].set_height(patches[-1].get_height() + n_upper_outliers)
        patches[-1].set_facecolor('m')
        patches[-1].set_label(f'Upper outliers: ({upper:.2f}, {data.max():.2f})')
        pass

    if lower_outliers or upper_outliers: plt.legend()
    return
