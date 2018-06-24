# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur
# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class PriorsSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['PP'] = excalibur.ValuesList()
        self['priors'] = excalibur.ValuesDict()
        self['pignore'] = excalibur.ValuesDict()
        self['ignore'] = excalibur.ValuesList()
        self['autofill'] = excalibur.ValuesList()
        self['needed'] = excalibur.ValuesList()
        self['pneeded'] = excalibur.ValuesList()
        self['starmdt'] = excalibur.ValuesList()
        self['planetmdt'] = excalibur.ValuesList()
        self['extsmdt'] = excalibur.ValuesList()
        self['starkeys'] = excalibur.ValuesList()
        self['planetkeys'] = excalibur.ValuesList()
        self['exts'] = excalibur.ValuesList()
        self['starmdt'].extend(['R*', 'T*', 'FEH*', 'LOGG*'])
        self['planetmdt'].extend(['inc', 'period', 'ecc', 'rp',
                                  't0', 'sma', 'mass'])
        self['extsmdt'].extend(['_lowerr', '_uperr'])
        self['STATUS'].append(False)
        self['PP'].append(False)
        return
    
    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            vlabels = ['FORCE PARAMETER',
                       'MISSING MANDATORY PARAMETERS',
                       'MISSING PLANET PARAMETERS',
                       'PLANETS IGNORED',
                       'AUTOFILL']
            hlabels = ['/', 'VALUE']
            table = visitor.add_table(clabels=hlabels,
                                      rows=len(vlabels))
            table.get_cell(0, 0).add_primitive(vlabels[0])
            table.get_cell(0, 1).add_primitive(self['PP'][-1])
            table.get_cell(1, 0).add_primitive(vlabels[1])
            table.get_cell(1, 1).add_primitive(self['needed'])
            table.get_cell(2, 0).add_primitive(vlabels[2])
            table.get_cell(2, 1).add_primitive(self['pneeded'])
            table.get_cell(3, 0).add_primitive(vlabels[3])
            table.get_cell(3, 1).add_primitive(self['ignore'])
            table.get_cell(4, 0).add_primitive(vlabels[4])
            table.get_cell(4, 1).add_primitive(self['autofill'])
            
            starinfo = self['priors']
            skeys = self['starmdt']
            exts = self['exts']
            allstar = []
            for key in skeys:
                listkeys = [key]
                listkeys.extend([key+x for x in exts])
                allstar.append(listkeys)
                pass
            pkeys = self['planetmdt']
            pkeys.pop(pkeys.index('mass'))
            pkeys.append('logg')
            allplanet = []
            for key in pkeys:
                listkeys = [key]
                listkeys.extend([key+x for x in exts])
                allplanet.append(listkeys)
                pass
            labels = ['STAR', 'UPPER ERR', 'LOWER ERR', 'UNITS', 'REF']
            table = visitor.add_table(clabels=labels,
                                      rows=len(allstar))
            for starlabels in allstar:
                i = allstar.index(starlabels)
                for l in starlabels:
                    j = starlabels.index(l)
                    table.get_cell(i, j).add_primitive(l)
                    elem = starinfo[l]
                    table.get_cell(i, j).add_primitive(elem)
                    pass
                pass
            for c in starinfo['planets']:
                labels = ['PLANET '+c, 'UPPER ERR', 'LOWER ERR', 'UNITS', 'REF']
                table = visitor.add_table(clabels=labels,
                                          rows=len(allplanet))
                for starlabels in allplanet:
                    i = allplanet.index(starlabels)
                    for l in starlabels:
                        j = starlabels.index(l)
                        table.get_cell(i, j).add_primitive(l)
                        elem = starinfo[c][l]
                        table.get_cell(i, j).add_primitive(elem)
                        pass
                    pass
                pass
            pass
        return
    pass
# -------- -----------------------------------------------------------
