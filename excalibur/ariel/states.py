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
        self['target'] = excalibur.ValuesList()
        self['planets'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['spectrum'] = excalibur.ValuesDict()
        self['spectrum_params'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            # rid = int(os.environ.get('RUNID', None))   # nope doesn't work
            # rid = 33
            # rid = 999  # nope doesn't work (to get the latest one)
            # rid = 666

            arielPlotDir = excalibur.context['data_dir'] + '/ariel/'
            subdirs = os.listdir(arielPlotDir)
            bestsubdir = 'RID027'
            for subdir in subdirs:
                if subdir.startswith('RID') and subdir > bestsubdir:
                    bestsubdir = subdir

            for target,planetLetter in zip(self['target'],self['planets']):
                myfig = plt.figure()
                plotDir = arielPlotDir + bestsubdir
                # plotDir = arielPlotDir + 'RID' + f'{rid:03}'
                # '/ariel/RID' + str('%03i' %rid)
                plot2show = img.imread(os.path.join(
                    plotDir,
                    'ariel_taurexAtmos_'+target+'_'+planetLetter+'.png'))
                plt.imshow(plot2show)
                plt.axis('off')
                buf = io.BytesIO()
                myfig.savefig(buf, format='png')
                visitor.add_image('...',
                                  '------ simulated Ariel spectrum for '+target+' '+planetLetter+' ------',
                                  buf.getvalue())
                plt.close(myfig)

                # PLOT THE SAVED STATE VECTOR, RATHER THAN THE SAVED FILE

                # myfig = plt.figure()
                # plt.imshow(self['data'][planetLetter]['plot_simspectrum'])
                # plt.axis('off')
                # buf = io.BytesIO()
                # myfig.savefig(buf, format='png')
                visitor.add_image('...',
                                  '------ same plot from state vector ------',
                                  self['data'][planetLetter]['plot_simspectrum'])
                #                   buf.getvalue())
                # plt.close(myfig)

        return

# -------- -----------------------------------------------------------
