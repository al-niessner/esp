# -- IMPORTS -- ------------------------------------------------------
import io
import dawgie
import excalibur

# import numpy as np GMR: UNUSED FOR NOW
import matplotlib.pyplot as plt
# ------------- ------------------------------------------------------

# -- SV -- -----------------------------------------------------------
class PredictSV(dawgie.StateVector):
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
                visitor.add_declaration('PLANET: ' + p)
                visitor.add_declaration('PREDICTION: ' + self['data'][p]['prediction'])
                allwhite = self['data'][p]['allwhite']
                postlc = self['data'][p]['postlc']
                postsep = self['data'][p]['postsep']
                myfig = plt.figure(figsize=(10, 6))
                plt.title(p)
                plt.plot(postsep, allwhite, 'o')
                plt.plot(postsep, postlc, '^', label='M')
                plt.xlabel('Star / Planet separation [R$^*$]')
                plt.ylabel('Normalized Post White Light Curve')
                plt.legend(bbox_to_anchor=(1 + 0.1*(0.5), 0.5),
                           loc=5, ncol=1, mode='expand', numpoints=1,
                           borderaxespad=0., frameon=False)
                plt.tight_layout(rect=[0,0,(1 - 0.1),1])
                buf = io.BytesIO()
                myfig.savefig(buf, format='png')
                visitor.add_image('...', ' ', buf.getvalue())
                # plt.show()
                plt.close(myfig)
                pass
            pass
        pass
    pass
