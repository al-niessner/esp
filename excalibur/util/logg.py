'''util logg ds'''

# -- IMPORTS -- ------------------------------------------------------
import numpy


# -------------- -----------------------------------------------------
# -- SV VALIDITY -- --------------------------------------------------
def calculate_logg(mass, radius, sscmks, units='solar'):
    '''calculate log(g).  units should be solar for a star, Jupiter for a planet'''

    if units == 'solar':
        mass = float(mass) * sscmks['Msun']
        radius = float(radius) * sscmks['Rsun']
    else:
        mass = float(mass) * sscmks['Mjup']
        radius = float(radius) * sscmks['Rjup']

    g = sscmks['G'] * mass / radius**2
    logg = numpy.log10(g)

    return logg
