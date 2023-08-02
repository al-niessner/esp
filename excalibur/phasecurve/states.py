'''Phasecurve Database Products View'''
# -- IMPORTS -- ------------------------------------------------------
import io

import dawgie

import excalibur

import numpy as np
import matplotlib.pyplot as plt
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class NormSV(dawgie.StateVector):
    '''phasecurve.normalization view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            for p in self['data'].keys():
                for v, m in zip(self['data'][p]['vignore'], self['data'][p]['trial']):
                    strignore = str(int(v)) + ' ' + m
                    visitor.add_declaration('VISIT IGNORED: ' + strignore)
                    pass
                vrange = self['data'][p]['vrange']
                for index, v in enumerate(self['data'][p]['visits']):
                    wave = self['data'][p]['wave'][index]
                    nspec = self['data'][p]['nspec'][index]
                    myfig = plt.figure()
                    plt.title('Visit: '+str(v))
                    for w,s in zip(wave, nspec):
                        select = (w > np.min(vrange)) & (w < np.max(vrange))
                        plt.plot(w[select], s[select], 'o')
                        pass
                    plt.ylabel('Normalized Spectra')
                    plt.xlabel('Wavelength [$\\mu$m]')
                    plt.xlim(np.min(vrange), np.max(vrange))
                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    visitor.add_image('...', ' ', buf.getvalue())
                    plt.close(myfig)
                    pass
                pass
            pass
        pass
    pass

class WhiteLightSV(dawgie.StateVector):
    '''phasecurve.whitelight view'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            for p in self['data'].keys():

                if 'HST' in self.__name:

                    visits = self['data'][p]['visits']
                    phase = self['data'][p]['phase']
                    allwhite = self['data'][p]['allwhite']
                    postim = self['data'][p]['postim']
                    postphase = self['data'][p]['postphase']
                    postlc = self['data'][p]['postlc']
                    postflatphase = self['data'][p]['postflatphase']
                    myfig = plt.figure(figsize=(10, 6))
                    plt.title(p)
                    for index, v in enumerate(visits):
                        plt.plot(phase[index], allwhite[index], 'k+')
                        plt.plot(postphase[index], allwhite[index]/postim[index],
                                 'o', label=str(v))
                        pass
                    if len(visits) > 14: ncol = 2
                    else: ncol = 1
                    plt.plot(postflatphase, postlc, '^', label='M')
                    plt.xlabel('Orbital Phase')
                    plt.ylabel('Normalized Post White Light Curve')
                    plt.legend(bbox_to_anchor=(1 + 0.1*(ncol - 0.5), 0.5), loc=5, ncol=ncol,
                               mode='expand', numpoints=1, borderaxespad=0., frameon=False)
                    plt.tight_layout(rect=[0,0,(1 - 0.1*ncol),1])
                    buf = io.BytesIO()
                    myfig.savefig(buf, format='png')
                    visitor.add_image('...', ' ', buf.getvalue())
                    plt.close(myfig)
                elif 'Spitzer' in self.__name:
                    # for each event
                    for i in range(len(self['data'][p])):
                        # plots are saved into sv
                        visitor.add_image('...', ' ', self['data'][p][i]['plot_bestfit'])
                        visitor.add_image('...', ' ', self['data'][p][i]['plot_residual_fft'])
                        visitor.add_image('...', ' ', self['data'][p][i]['plot_posterior'])
                        visitor.add_image('...', ' ', self['data'][p][i]['plot_pixelmap'])
                        # another centroid timeseries plot?
                    pass
                pass
            pass
        return
    pass
# -------- -----------------------------------------------------------
