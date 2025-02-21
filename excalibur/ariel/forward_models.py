'''ariel forward_models ds'''

# Heritage code shame:
# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments

# import os
# import excalibur
import excalibur.system.core as syscore
from excalibur.cerberus.core import hazelib
from excalibur.cerberus.forward_model import crbmodel


# ----------------------------------------------------------------------------------------------
def make_cerberus_atmos(
    wavelength_um,
    model_params,
    xslib,
    planet_letter,
    mixratios=None,
    Hsmax=15.0,
    solrad=10.0,
):
    '''
    Create a simulated spectrum using the code that's better than the other ones
    '''

    # print('modelparams',model_params)

    # EQUILIBRIUM TEMPERATURE
    Teq = model_params['Teq']

    # CLOUD/HAZE PARAMETERS
    ctp = model_params['CTP']
    hza = model_params['HScale']
    hzloc = model_params['HLoc']
    hzthick = model_params['HThick']

    # ABUNDANCES
    if mixratios:
        # use the given mixing ratios
        tceqdict = None
    else:
        # equilibrium chemistry
        tceqdict = {}
        tceqdict['XtoH'] = model_params['metallicity']
        tceqdict['CtoO'] = model_params['C/O']
        tceqdict['NtoO'] = 0
        # print('cloudfree forward model input chem =',tceqdict)

    ssc = syscore.ssconstants(mks=True)
    solidr = model_params['Rp'] * ssc['Rjup']  # MK
    # orbp = fin['priors'].copy()
    # orbp = model_params   # just hope this covers it ig
    # orbp = {'sma':model_params['sma']}
    # planet_letter = 'placeholder'
    orbp = {
        'R*': model_params['R*'],
        planet_letter: {'logg': model_params['logg']},
    }

    crbhzlib = {'PROFILE': []}
    # hazedir = os.path.join(excalibur.context['data_dir'], 'CERBERUS/HAZE')
    # print('hazedir: ',hazedir)
    hazelib(crbhzlib)
    # print('haze lib',crbhzlib)

    # print('wavelength range',wavelength_um[0],wavelength_um[-1])

    # CERBERUS FORWARD MODEL
    # fmc, fmc_by_molecule = crbmodel(None, float(hza), float(ctp), solidr, orbp,
    fmc, fmc_by_molecule = crbmodel(
        mixratios,
        float(hza),
        float(ctp),
        solidr,
        orbp,
        xslib['data'][planet_letter]['XSECS'],
        xslib['data'][planet_letter]['QTGRID'],
        float(Teq),
        wavelength_um,
        Hsmax=Hsmax,
        solrad=solrad,
        # np.array(ctxt.spc['data'][ctxt.p]['WB']),
        hzlib=crbhzlib,
        hzp='AVERAGE',
        hztop=float(hzloc),
        hzwscale=float(hzthick),
        cheq=tceqdict,
        pnet=planet_letter,
        verbose=False,
        debug=False,
        break_down_by_molecule=True,
    )

    return fmc, fmc_by_molecule
