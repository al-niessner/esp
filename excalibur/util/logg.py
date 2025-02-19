'''util logg ds'''

# -- IMPORTS -- ------------------------------------------------------
import numpy


# -------------- -----------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def calculate_logg(M, R, sscmks, units='solar'):
    '''calculate log(g).  units should be solar for a star, Jupiter for a planet'''

    if units == 'solar':
        mass = float(M) * sscmks['Msun']
        radius = float(R) * sscmks['Rsun']
    else:
        mass = float(M) * sscmks['Mjup']
        radius = float(R) * sscmks['Rjup']

    g = sscmks['G'] * mass / radius**2
    logg = numpy.log10(g)

    return logg
