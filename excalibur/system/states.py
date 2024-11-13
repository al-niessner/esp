'''System Database Products View'''
# -- IMPORTS -- ------------------------------------------------------
import dawgie

import excalibur
import excalibur.system.plot as sysplot

# ------------- ------------------------------------------------------
# -- SV -- -----------------------------------------------------------
class PriorsSV(dawgie.StateVector):
    '''General format for system State Vector view'''
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,4)
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
        self['starnonmdt'] = excalibur.ValuesList()
        self['planetmdt'] = excalibur.ValuesList()
        self['planetnonmdt'] = excalibur.ValuesList()
        self['starkeys'] = excalibur.ValuesList()
        self['planetkeys'] = excalibur.ValuesList()
        self['exts'] = excalibur.ValuesList()
        self['STATUS'].append(False)
        self['PP'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller:excalibur.identity, visitor:dawgie.Visitor)->None:
        '''view ds'''
        if self['STATUS'][-1]:
            vlabels = ['FORCE PARAMETER',
                       'MISSING MANDATORY PARAMETERS',
                       'MISSING PLANET PARAMETERS',
                       'PLANETS IGNORED', 'AUTOFILL']
            hlabels = ['/', 'VALUE']
            table = visitor.add_table(clabels=hlabels, rows=len(vlabels))
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

            allstar = []
            for key in self['starmdt']:
                listkeys = [key]
                listkeys.extend([key+x for x in self['exts']])
                allstar.append(listkeys)
                pass
            pkeys = self['planetmdt']
            # why remove mass?  It's a nice parameter to see in the final table, no?
            # pkeys.pop(pkeys.index('mass'))
            # logg is now in the mandatory param list in core.py; don't add again here
            # pkeys.append('logg')
            allplanet = []
            for key in pkeys:
                listkeys = [key]
                listkeys.extend([key+x for x in self['exts']])
                allplanet.append(listkeys)
                pass
            labels = ['STAR', 'UPPER ERR', 'LOWER ERR',
                      'UNITS', 'REF']
            table = visitor.add_table(clabels=labels, rows=len(allstar))
            for starlabels in allstar:
                i = allstar.index(starlabels)
                for l in starlabels:
                    table.get_cell(i, starlabels.index(l)).add_primitive(l)
                    table.get_cell(i, starlabels.index(l)).add_primitive(self['priors'][l])
                    pass
                pass
            for c in self['priors']['planets']:
                labels = ['PLANET '+c, 'UPPER ERR', 'LOWER ERR',
                          'UNITS', 'REF']
                table = visitor.add_table(clabels=labels,
                                          rows=len(allplanet))
                for starlabels in allplanet:
                    i = allplanet.index(starlabels)
                    for l in starlabels:
                        table.get_cell(i, starlabels.index(l)).add_primitive(l)
                        table.get_cell(i, starlabels.index(l)).add_primitive(self['priors'][c][l])
                        pass
                    pass
                pass
            pass
        return
    pass
# -------- -----------------------------------------------------------
class PopulationSV(dawgie.StateVector):
    '''PopulationSV ds'''
    def __init__(self, name):
        self._version_ = dawgie.VERSION(2,0,0)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['data'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        '''name ds'''
        return self.__name

    def view(self, caller:excalibur.identity, visitor:dawgie.Visitor)->None:
        '''view ds'''
        to_process = [('----------------------Stellar Population Distributions----------------------',
                       self['data']['st_attrs'], self['data']['st_attrs_roudier62'], False),
                      ('---------------------Planetary Population Distributions---------------------',
                       self['data']['pl_attrs'], self['data']['pl_attrs_roudier62'], True)]
        # for title, attrs, attrs_roudier62,is_planet in to_process:
        for title, attrs, attrs_roudier62,_ in to_process:
            visitor.add_primitive(title)
            for key in attrs:
                sysplot.distrplot(key, attrs[key], attrs_roudier62[key], visitor, 'no units?')

        return

# -------------------------------------------------------------------
