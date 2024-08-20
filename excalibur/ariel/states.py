'''ariel Database Products View'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur

import io
import os

import matplotlib.image as img
import matplotlib.pyplot as plt

# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class PriorsSV(dawgie.StateVector):
    '''General format for ariel State Vector view'''
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,4)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        # self['target'] = excalibur.ValuesList()
        # self['planets'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        # self['spectrum'] = excalibur.ValuesDict()
        # self['spectrum_params'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller:excalibur.identity, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:

            # display plot saved as a state vector; easy peasy
            # (alternative below: load a saved .png file)
            plotStateVector = True

            target = self['data']['target']

            if plotStateVector:
                for planetLetter in self['data']['planets']:
                    for model in self['data']['models']:
                        # problem: for multiplanet systems, individual models may fail
                        # so there might be some missing planets here
                        # but there should be all models present, if any models present
                        if planetLetter in self['data'].keys():
                            if model in self['data'][planetLetter].keys():
                                visitor.add_image('...',
                                                  '------ simulated Ariel spectrum for '+target+' '+planetLetter+'  MODEL:'+model+' ------',
                                                  self['data'][planetLetter][model]['plot_simspectrum'])

            else:
                # determine the most recent RID from subdir filenames
                arielPlotDir = excalibur.context['data_dir'] + '/ariel/'
                subdirs = os.listdir(arielPlotDir)
                bestsubdir = 'RID000'
                for subdir in subdirs:
                    if subdir.startswith('RID') and len(subdir)==6 and \
                       subdir > bestsubdir: bestsubdir = subdir

                for planetLetter in self['data']['planets']:
                    for model in self['data']['models']:
                        myfig = plt.figure()
                        plotDir = arielPlotDir + bestsubdir
                        plot2show = img.imread(os.path.join(
                            plotDir,
                            'ariel_'+model+'Atmos_'+target+'_'+planetLetter+'.png'))
                        plt.imshow(plot2show)
                        plt.axis('off')
                        buf = io.BytesIO()
                        myfig.savefig(buf, format='png')
                        visitor.add_image('...',
                                          '------ simulated Ariel spectrum for '+target+' '+planetLetter+'  MODEL:'+model+' ------',
                                          buf.getvalue())
                        plt.close(myfig)

        return

# -------- -----------------------------------------------------------
