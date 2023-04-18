'''system autofill ds'''
# -- IMPORTS -- ------------------------------------------------------
import numpy
import excalibur.system.core as syscore

# ----------------- --------------------------------------------------
# -- FILL BLANK UNCERTAINTY FIELD ------------------------------------
def fillUncertainty(param,param_value,param_uncertainty,error_type):
    '''
    Put in some nominal value for the parameter uncertainty
    '''

    # print()
    # print(param,error_type)
    # print(param_value)
    # print(param_uncertainty)

    autofilled = False
    try:
        fillvalue = float(param_uncertainty)
    except ValueError:
        autofilled = True
        dawgieStupidity = False
        if param=='spTyp':
            # (spectral type isn't mandatory, so probably never comes here)
            fillvalue = ''
        elif param=='T*':
            # for stellar temperature, 100 K uncertainty as default
            fillvalue = 100
        elif param=='inc':
            # inclination should be pretty close to 90degrees
            #  let's try 5deg uncertainty
            fillvalue = 5
            # make sure inclination range stays within 0-90 degrees
            # actually no, allow inclination to go 0-180, as in the Archive
            # if param_value + fillvalue > 90: fillvalue = 90 - param_value
        elif param=='ecc':
            # for eccentricity, set uncertainty to 20%, with a minimum of 0.1
            fillvalue = numpy.min([float(param_value) * 2.e-1, 0.1])
        elif param=='omega':
            # for argument of periastron, set a large uncertainty (120 degrees)
            fillvalue = 120
        elif param=='FEH*':
            # for metallicity, fallback uncertainty is 0.1 dex
            fillvalue = 0.1
        else:
            dawgieStupidity = True
        if dawgieStupidity:
            if param in ['LOGG*','logg']:
                # for planet or stellar log-g, fallback uncertainty is 0.1 dex
                fillvalue = 0.1
            elif param=='Hmag':
                # 0.1 dex for H band magnitude (for Kepler-1651)
                fillvalue = 0.1
            elif param=='period':
                # orbital period should be known pretty well. how about 1%?
                fillvalue = float(param_value) * 1.e-2
            elif param=='t0':
                # T_0 is known to much much better than 10%
                #   set it to 10% of the period?
                #   problem though - period is not passed in here
                # for now, set it to 1 hour uncertainty
                fillvalue = 1./24.
            elif param in ['rp','sma','mass','R*','M*','RHO*','L*']:
                # set uncertainty to 10% for planet radius, semi-major axis, mass
                #   same for stellar radius, mass, density, luminosity
                fillvalue = float(param_value) * 1.e-1
            else:
                # fallback option is to set uncertainty to 10%
                fillvalue = float(param_value) * 1.e-1
                print('another PARAM:',param)

        # make sure that the upper error is positive and the lower is negative
        if error_type=='uperr':
            fillvalue = abs(fillvalue)
        elif error_type=='lowerr':
            fillvalue = -abs(fillvalue)
        else:
            pass
            # exit('ERROR: unknown error_type')

    return fillvalue,autofilled

# ----------------- --------------------------------------------------
# -- SELECT THE BEST PARAMETER VALUE FROM VARIOUS ARCHIVE VALUES -----
def bestValue(values,uperrs,lowerrs,refs):
    '''
    From a list of parameter values, determine the most trustworthy value
    '''
    if values[0] != '':
        # step 1: if there is a default value at the start of the list, use that
        bestvalue = values[0]
        bestuperr = uperrs[0]
        bestlowerr = lowerrs[0]
        bestref = refs[0]
    else:
        # step 2: iterate from the end of the list inward, until getting a non-blank value
        #   (this assumes that the non-default values have been ordered by publish date,
        #    such that the end of the list is the most recently published value)
        bestvalue = ''
        bestref = ''
        bestyear = 0
        bestuperr = ''
        bestlowerr = ''
        for value,uperr,lowerr,ref in zip(values,uperrs,lowerrs,refs):
            try:
                year = int(ref[-4:])
            except ValueError:
                # there are some refs without a year; make them lower priority
                year = 1
            # select the most recently published non-blank value
            if value != '' and year >= bestyear:
                # use author name as tie-breaker for 2 with same year
                if year==bestyear and bestref < ref:
                    # print('skipping this same-year ref',year,ref,bestref)
                    pass
                else:
                    bestyear = year
                    bestvalue = value
                    bestuperr = uperr
                    bestlowerr = lowerr
                    bestref = ref

    return bestvalue,bestuperr,bestlowerr,bestref

# -------------------------------------------------------------------
def estimate_mass_from_radius(radius_Jup):
    '''
    Use an assumed mass-radius relationship to guess the planet mass
    '''

    sscmks = syscore.ssconstants(cgs=True)

    radius_Earth = radius_Jup * sscmks['Rjup'] / sscmks['Rearth']

    ChenKipping_is_acceptable = False

    if ChenKipping_is_acceptable:
        rocky_slope = 0.279
        gas_slope = 0.589
        R_0 = 1.008  # radius for 1 Earth mass
        transition_point_mass = 2.04  # Earth masses
        transition_point_radius = R_0 * transition_point_mass**rocky_slope
        # transition point is 1.23 Earth radii
        # print('power law transition point (R_Earth)',transition_point_radius)

        if radius_Earth < transition_point_radius:
            mass_Earth = (radius_Earth/R_0)**(1./rocky_slope)
            # at trans: mass_Earth = (transition_point_radius/R_0)**(1./rocky_slope)
            # which simplifies to: mass_Earth = transition_point_mass  check
        else:
            mass_Earth = transition_point_mass * \
                (radius_Earth/transition_point_radius)**(1./gas_slope)
            # at trans: mass_Earth = transition_point_mass

    else:
        # Traub2011 formula, modified to enforce Earth/Jupiter fit
        r0 = 0.375
        r1 = 11.
        m1 = 1000.
        w = 1.5
        if radius_Earth > r1:
            # for radius larger than Jupiter, assume Jupiter mass
            mass_Earth = sscmks['Mjup'] / sscmks['Mearth']

        elif radius_Earth < 1:
            # for planets smaller than Earth, assume Earth density
            mass_Earth = radius_Earth**3

        else:
            # regular formula
            mass_Earth = m1*10.**(-w*((r1-radius_Earth)/(radius_Earth-r0))**0.25)

    mass_Jup = mass_Earth * sscmks['Mearth'] / sscmks['Mjup']

    return mass_Jup

# -------------------------------------------------------------------
def derive_RHOstar_from_M_and_R(starInfo):
    '''
    If stellar density is blank, fill it in based on R* and M*
    '''

    # get Msun and Rsun definitions, for calculating stellar density from M*,R*
    sscmks = syscore.ssconstants(cgs=True)

    RHO_derived = []
    RHO_lowerr_derived = []
    RHO_uperr_derived = []
    RHO_ref_derived = []

    for R,Rerr1,Rerr2, M,Merr1,Merr2, RHO,RHOerr1,RHOerr2,RHOref in zip(
            starInfo['R*'],  starInfo['R*_lowerr'],  starInfo['R*_uperr'],
            starInfo['M*'],  starInfo['M*_lowerr'],  starInfo['M*_uperr'],
            starInfo['RHO*'],starInfo['RHO*_lowerr'],starInfo['RHO*_uperr'],
            starInfo['RHO*_ref']):

        # check for blank stellar density
        #  (but only update it if M* and R* are both defined)
        if RHO=='' and R!='' and M!='':
            newRHO = float(M)*sscmks['Msun'] / \
                (4.*numpy.pi/3. * (float(R)*sscmks['Rsun'])**3)
            # RHO_derived.append(str('%6.4f' %newRHO))
            RHO_derived.append(f'{newRHO:6.4f}')
            RHO_ref_derived.append('derived from M*,R*')

            # also fill in the uncertainty on RHO, based on R,M uncertainties
            if Rerr1=='' or Merr1=='':
                RHO_lowerr_derived.append('')
            else:
                newRHOfractionalError1 = -numpy.sqrt((3.*float(Rerr1)/float(R))**2 +
                                                     (float(Merr1)/float(M))**2)
                # RHO_lowerr_derived.append(str('%6.4f' %(newRHO * newRHOfractionalError1)))
                RHO_lowerr_derived.append(f'{(newRHO * newRHOfractionalError1):6.4f}')
            if Rerr2=='' or Merr2=='':
                RHO_uperr_derived.append('')
            else:
                newRHOfractionalError2 = numpy.sqrt((3.*float(Rerr2)/float(R))**2 +
                                                    (float(Merr2)/float(M))**2)
                # RHO_uperr_derived.append(str('%6.4f' %(newRHO * newRHOfractionalError2)))
                RHO_uperr_derived.append(f'{(newRHO * newRHOfractionalError2):6.4f}')
        else:
            RHO_derived.append(RHO)
            RHO_lowerr_derived.append(RHOerr1)
            RHO_uperr_derived.append(RHOerr2)
            RHO_ref_derived.append(RHOref)

    return RHO_derived, RHO_lowerr_derived, RHO_uperr_derived, RHO_ref_derived

# -------------------------------------------------------------------
def derive_SMA_from_P_and_Mstar(starInfo, planetLetter):
    '''
    If semi-major axis is blank, fill it in based on the orbital period and star mass
    '''

    # get G, Msun, AU definitions
    sscmks = syscore.ssconstants(cgs=True)

    sma_derived = []
    sma_lowerr_derived = []
    sma_uperr_derived = []
    sma_ref_derived = []

    for M,Merr1,Merr2, P,Perr1,Perr2, sma,sma_err1,sma_err2,sma_ref in zip(
            starInfo['M*'],starInfo['M*_lowerr'],starInfo['M*_uperr'],
            starInfo[planetLetter]['period'],
            starInfo[planetLetter]['period_lowerr'],
            starInfo[planetLetter]['period_uperr'],
            starInfo[planetLetter]['sma'],
            starInfo[planetLetter]['sma_lowerr'],
            starInfo[planetLetter]['sma_uperr'],
            starInfo[planetLetter]['sma_ref']):

        # check for blank stellar density
        #  (but only update it if M* and P are both defined)
        if sma=='' and M!='' and P!='':
            # M = 1
            # P = 365.25
            # giving these Earth parameters results in 0.9999874 AU
            GM = sscmks['G'] * float(M)*sscmks['Msun']
            newsma = (GM * (float(P)*sscmks['day'] /2./numpy.pi)**2)**(1./3.)
            newsma /= sscmks['AU']
            # sma_derived.append(str('%6.4f' %newsma))
            sma_derived.append(f'{newsma:6.4f}')
            # print('P M',P,M)
            # print('derived sma',newsma)
            # exit('test')
            sma_ref_derived.append('derived from P,M*')

            # also fill in the uncertainty on sma, based on R,M uncertainties
            if Perr1=='' or Merr1=='':
                sma_lowerr_derived.append('')
            else:
                newsma_fractionalError1 = -numpy.sqrt((2./3.*float(Perr1)/float(P))**2 +
                                                      (1./3.*float(Merr1)/float(M))**2)
                # sma_lowerr_derived.append(str('%6.4f' %(newsma * newsma_fractionalError1)))
                sma_lowerr_derived.append(f'{(newsma * newsma_fractionalError1):6.4f}')
            if Perr2=='' or Merr2=='':
                sma_uperr_derived.append('')
            else:
                newsma_fractionalError2 = numpy.sqrt((2./3.*float(Perr2)/float(P))**2 +
                                                     (1./3.*float(Merr2)/float(M))**2)
                # sma_uperr_derived.append(str('%6.4f' %(newsma * newsma_fractionalError2)))
                sma_uperr_derived.append(f'{(newsma * newsma_fractionalError2):6.4f}')
        else:
            sma_derived.append(sma)
            sma_lowerr_derived.append(sma_err1)
            sma_uperr_derived.append(sma_err2)
            sma_ref_derived.append(sma_ref)

    return sma_derived, sma_lowerr_derived, sma_uperr_derived, sma_ref_derived

# -------------------------------------------------------------------
def derive_LOGGstar_from_R_and_M(starInfo):
    '''
    If stellar log-g is blank, fill it in based on the star's radius and mass
    '''

    # get Msun and Rsun definitions
    sscmks = syscore.ssconstants(cgs=True)

    LOGG_derived = []
    LOGG_lowerr_derived = []
    LOGG_uperr_derived = []
    LOGG_ref_derived = []

    for R,Rerr1,Rerr2, M,Merr1,Merr2, LOGG,LOGGerr1,LOGGerr2,LOGGref in zip(
            starInfo['R*'],  starInfo['R*_lowerr'],  starInfo['R*_uperr'],
            starInfo['M*'],  starInfo['M*_lowerr'],  starInfo['M*_uperr'],
            starInfo['LOGG*'],starInfo['LOGG*_lowerr'],starInfo['LOGG*_uperr'],
            starInfo['LOGG*_ref']):

        # check for blank stellar log-g
        #  (but only update it if M* and R* are both defined)
        if LOGG=='' and R!='' and M!='':
            g = sscmks['G'] * float(M)*sscmks['Msun'] / (float(R)*sscmks['Rsun'])**2
            newLOGG = numpy.log10(g)
            # LOGG_derived.append(str('%6.4f' %newLOGG))
            LOGG_derived.append(f'{newLOGG:6.4f}')

            LOGG_ref_derived.append('derived from M*,R*')

            # also fill in the uncertainty on LOGG, based on R,M uncertainties
            if Rerr1=='' or Merr1=='':
                LOGG_lowerr_derived.append('')
            else:
                newLOGGfractionalError1 = -numpy.sqrt((2.*float(Rerr1)/float(R))**2 +
                                                      (float(Merr1)/float(M))**2)
                # LOGG_lowerr_derived.append(str('%6.4f' %numpy.log10
                LOGG_lowerr_derived.append(f'{numpy.log10(1 + newLOGGfractionalError1):6.4f}')
            if Rerr2=='' or Merr2=='':
                LOGG_uperr_derived.append('')
            else:
                newLOGGfractionalError2 = numpy.sqrt((2.*float(Rerr2)/float(R))**2 +
                                                     (float(Merr2)/float(M))**2)
                # LOGG_uperr_derived.append(str('%6.4f' %numpy.log10
                LOGG_uperr_derived.append(f'{numpy.log10(1 + newLOGGfractionalError2):6.4f}')
        else:
            LOGG_derived.append(LOGG)
            LOGG_lowerr_derived.append(LOGGerr1)
            LOGG_uperr_derived.append(LOGGerr2)
            LOGG_ref_derived.append(LOGGref)

    return LOGG_derived, LOGG_lowerr_derived, LOGG_uperr_derived, LOGG_ref_derived

# -------------------------------------------------------------------
def derive_LOGGplanet_from_R_and_M(starInfo, planetLetter, verbose=False):
    '''
    If planetary log-g is blank, fill it in based on the planet's radius and mass
    '''

    # get MJup and RJup definitions
    sscmks = syscore.ssconstants(cgs=True)

    logg_derived = []
    logg_lowerr_derived = []
    logg_uperr_derived = []
    logg_ref_derived = []

    # print()
    # print(starInfo[planetLetter])
    # print(starInfo[planetLetter].keys())
    # print()

    # this one is different from above - the 'logg' field doesn't exist yet
    if 'logg' in starInfo[planetLetter].keys() and verbose:
        print('ERROR: logg field shouldnt exist yet')
    # for R,Rerr1,Rerr2, M,Merr1,Merr2, logg,loggerr1,loggerr2,loggref in zip(
    for R,Rerr1,Rerr2, M,Merr1,Merr2 in zip(
            starInfo[planetLetter]['rp'],
            starInfo[planetLetter]['rp_lowerr'],
            starInfo[planetLetter]['rp_uperr'],
            starInfo[planetLetter]['mass'],
            starInfo[planetLetter]['mass_lowerr'],
            starInfo[planetLetter]['mass_uperr']):
        # starInfo[planetLetter]['logg'],
        # starInfo[planetLetter]['logg_lowerr'],
        # starInfo[planetLetter]['logg_uperr'],
        # starInfo[planetLetter][planetLetter]['logg_ref']):

        # if logg=='' and R!='' and M!='':
        # no need to check for blank planetary log-g; it doesn't exist in Archive table
        if R!='' and M!='':
            g = sscmks['G'] * float(M)*sscmks['Mjup'] / (float(R)*sscmks['Rjup'])**2
            logg = numpy.log10(g)
            # print('M,R,logg',M,R,logg)
            # logg_derived.append(str('%6.4f' %logg))
            logg_derived.append(f'{logg:6.4f}')

            logg_ref_derived.append('derived from Mp,Rp')

            # also fill in the uncertainty on logg, based on R,M uncertainties
            if Rerr1=='' or Merr1=='':
                logg_lowerr_derived.append('')
            else:
                loggfractionalError1 = -numpy.sqrt((2.*float(Rerr1)/float(R))**2 +
                                                      (float(Merr1)/float(M))**2)
                # print()
                # print('Rp fractional error',float(Rerr1),float(R))
                # print('Rp fractional error',float(Rerr1)/float(R))
                # print('Mp fractional error',float(Merr1),float(M))
                # print('Mp fractional error',float(Merr1)/float(M))
                # print('logg fractional error',-loggfractionalError1)
                # if loggfractionalError1
                # logg_lowerr_derived.append(str('%6.4f' %numpy.log10

                # this conditional avoids log(negative) error
                if loggfractionalError1 > -1:
                    logg_lowerr_derived.append(f'{numpy.log10(1 + loggfractionalError1):6.4f}')
                else:
                    # HD 23472 Trifonov2019 has large R,M error bars, probably a typo
                    # doesn't matter, since it's not the default publication
                    logg_lowerr_derived.append('-1')
            if Rerr2=='' or Merr2=='':
                logg_uperr_derived.append('')
            else:
                loggfractionalError2 = numpy.sqrt((2.*float(Rerr2)/float(R))**2 +
                                                  (float(Merr2)/float(M))**2)
                # logg_uperr_derived.append(str('%6.4f' %numpy.log10
                #                              (1 + loggfractionalError2)))
                logg_uperr_derived.append(f'{numpy.log10(1 + loggfractionalError2):6.4f}')
        else:
            logg_derived.append('')
            logg_lowerr_derived.append('')
            logg_uperr_derived.append('')
            logg_ref_derived.append('')

    return logg_derived, logg_lowerr_derived, logg_uperr_derived, logg_ref_derived
# -------------------------------------------------------------------
def derive_Lstar_from_R_and_T(starInfo):
    '''
    If stellar luminosity is blank, calculate it from the star's radius and temperature
    '''

    # get Tsun definition
    sscmks = syscore.ssconstants(cgs=True)

    Lstar_derived = []
    Lstar_lowerr_derived = []
    Lstar_uperr_derived = []
    Lstar_ref_derived = []

    # print(starInfo['L*'])
    # print(starInfo['L*_lowerr'])
    # print(starInfo['L*_uperr'])

    for R,Rerr1,Rerr2, T,Terr1,Terr2, Lstar,Lstarerr1,Lstarerr2,Lstarref in zip(
            starInfo['R*'],starInfo['R*_lowerr'],starInfo['R*_uperr'],
            starInfo['T*'],starInfo['T*_lowerr'],starInfo['T*_uperr'],
            starInfo['L*'],starInfo['L*_lowerr'],starInfo['L*_uperr'],
            starInfo['L*_ref']):

        # check for blank stellar luminosity
        #  (but only update it if R* and T* are both defined)
        if Lstar=='' and R!='' and T!='':
            newLstar = float(R)**2 * (float(T)/sscmks['Tsun'])**4
            # print('Lstar derived',newLstar)

            Lstar_derived.append(f'{newLstar:6.4f}')
            Lstar_ref_derived.append('derived from R*,T*')

            # also fill in the uncertainty on Lstar, based on R,M uncertainties
            if Rerr1=='' or Terr1=='':
                Lstar_lowerr_derived.append('')
            else:
                newLstarfractionalError1 = -numpy.sqrt((2.*float(Rerr1)/float(R))**2 +
                                                       (4.*float(Terr1)/float(T))**2)
                Lstar_lowerr_derived.append(f'{(newLstar * newLstarfractionalError1):6.4f}')
            if Rerr2=='' or Terr2=='':
                Lstar_uperr_derived.append('')
            else:
                newLstarfractionalError2 = numpy.sqrt((2.*float(Rerr2)/float(R))**2 +
                                                      (4.*float(Terr2)/float(T))**2)
                Lstar_uperr_derived.append(f'{(newLstar * newLstarfractionalError2):6.4f}')
        else:
            # SPECIAL ATTENTION FOR L*: the Archive gives it logged; correct here
            if Lstar!='':
                Lstar = 10.**float(Lstar)
                # print('Lstar fixed',Lstar)
                # print('Lstar errors before',Lstarerr1,Lstarerr2)
                # there's an Archive bug where some upper error bars are missing, eg HAT-P-40
                if Lstarerr1=='' and Lstarerr2!='':
                    Lstarerr1 = -float(Lstarerr2)
                if Lstarerr1!='':
                    Lstarerr1 = Lstar * (10**float(Lstarerr1) - 1)
                    Lstarerr1 = f'{Lstarerr1:6.4f}'
                if Lstarerr2!='':
                    Lstarerr2 = Lstar * (10**float(Lstarerr2) - 1)
                    Lstarerr2 = f'{Lstarerr2:6.4f}'
                Lstar = f'{Lstar:6.4f}'
                # print('Lstar errors after ',Lstarerr1,Lstarerr2)
            Lstar_derived.append(Lstar)
            Lstar_lowerr_derived.append(Lstarerr1)
            Lstar_uperr_derived.append(Lstarerr2)
            Lstar_ref_derived.append(Lstarref)

    return Lstar_derived, Lstar_lowerr_derived, Lstar_uperr_derived, Lstar_ref_derived
