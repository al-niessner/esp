# -- IMPORTS -- ------------------------------------------------------
import io

import dawgie

import excalibur

import numpy as np
import matplotlib.pyplot as plt
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class NormSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return
    
    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            for p in self['data'].keys():
                for v, m in zip(self['data'][p]['vignore'],
                                self['data'][p]['trial']):
                    strignore = str(int(v)) + ' ' + m
                    visitor.add_declaration('VISIT IGNORED: ' + strignore)
                    pass
                vrange = self['data'][p]['vrange']
                for v in self['data'][p]['visits']:
                    index = self['data'][p]['visits'].index(v)
                    wave = self['data'][p]['wave'][index]
                    nspec = self['data'][p]['nspec'][index]
                    myfig = plt.figure()
                    for w,s in zip(wave, nspec):
                        select = ((w > np.min(vrange)) &
                                  (w < np.max(vrange)))
                        plt.plot(w[select], s[select], 'o')
                        pass
                    plt.ylabel('Normalized Spectra')
                    plt.xlabel('Wavelength [$\mu$m]')
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
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return
    
    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            for p in self['data'].keys():
                visits = self['data'][p]['visits']
                phase = self['data'][p]['phase']
                allwhite = self['data'][p]['allwhite']
                postim = self['data'][p]['postim']
                postphase = self['data'][p]['postphase']
                postlc = self['data'][p]['postlc']
                postflatphase = self['data'][p]['postflatphase']
                mcpost = self['data'][p]['mcpost']
                for kn in ['inc', 'rprs', 'dtk',
                           'vslope', 'vitcp',
                           'oslope', 'ologtau', 'ologdelay']:
                    showme = ' '.join(sorted([k for k in mcpost.keys()
                                            if kn in k]))
                    visitor.add_declaration(showme)
                    pass        
                myfig = plt.figure(figsize=(10, 6))
                for v in visits:
                    index = visits.index(v)
                    plt.plot(phase[index], allwhite[index], 'k+')
                    plt.plot(postphase[index],
                             allwhite[index]/postim[index],
                             'o', label=str(v))
                    pass
                if len(visits) > 14: ncol = 2
                else: ncol = 1
                plt.plot(postflatphase, postlc, '^', label='M')
                plt.xlabel('Orbital Phase')
                plt.ylabel('Normalized Post White Light Curve')
                plt.legend(bbox_to_anchor=(1 + 0.1*(ncol - 0.5),
                                           0.5), loc=5, ncol=ncol,
                           mode='expand', numpoints=1,
                           borderaxespad=0., frameon=False)
                plt.tight_layout(rect=[0,0,(1 - 0.1*ncol),1])
                buf = io.BytesIO()
                myfig.savefig(buf, format='png')
                visitor.add_image('...', ' ', buf.getvalue())
                plt.close(myfig)
                pass
            pass
        pass
    pass
# -------- -----------------------------------------------------------
