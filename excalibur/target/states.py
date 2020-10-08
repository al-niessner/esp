# -- IMPORTS -- ------------------------------------------------------
import bokeh.embed
import bokeh.plotting  # the awesome plotting engine
import dawgie
import numpy

import excalibur
# ------------- ------------------------------------------------------
# -- TARGET -- -------------------------------------------------------
class TargetSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['starID'] = excalibur.ValuesDict()
        self['nexscie'] = excalibur.ValuesList()
        self['nexscic'] = excalibur.ValuesList()
        self['nexscix'] = excalibur.ValuesList()
        self['candidates'] = excalibur.ValuesList()
        self['starkeys'] = excalibur.ValuesList()
        self['planetkeys'] = excalibur.ValuesList()
        self['exts'] = excalibur.ValuesList()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            targetlist = [target for target in self['starID'].keys()]
            targetlist = sorted(targetlist)
            ntarget = len(targetlist)
            labels = ['Star', 'Aliases', 'Planet', 'Proposal']
            table = visitor.add_table(clabels=labels, rows=ntarget)
            for target in targetlist:
                starinfo = self['starID'][target]
                i = targetlist.index(target)
                table.get_cell(i, 0).add_primitive(target)
                for alias in starinfo['aliases']:
                    table.get_cell(i, 1).add_primitive(alias)
                    pass
                for planet in starinfo['planets']:
                    table.get_cell(i, 2).add_primitive(planet)
                    pass
                for pid in starinfo['PID']:
                    table.get_cell(i, 3).add_primitive(pid)
                    pass
                pass
            starinfo = self['starID'][targetlist[0]]
            if len(self['STATUS']) > 2:
                allstar = []
                skeys = self['starkeys']
                exts = self['exts']
                for key in skeys:
                    listkeys = [key]
                    listkeys.extend([key+x for x in exts])
                    allstar.append(listkeys)
                    pass
                allplanet = []
                pkeys = self['planetkeys']
                for key in pkeys:
                    listkeys = [key]
                    listkeys.extend([key+x for x in exts])
                    allplanet.append(listkeys)
                    pass
                labels = [targetlist[0],
                          'UPPER ERR', 'LOWER ERR', 'UNITS', 'REF']
                table = visitor.add_table(clabels=labels,
                                          rows=len(allstar))
                for starlabels in allstar:
                    i = allstar.index(starlabels)
                    for l in starlabels:
                        j = starlabels.index(l)
                        table.get_cell(i, j).add_primitive(l)
                        elem = starinfo[l][-1]
                        if elem:
                            table.get_cell(i, j).add_primitive(elem)
                            pass
                        else:
                            table.get_cell(i, j).add_primitive('NA')
                            pass
                        pass
                    pass
                for c in self['starID'][targetlist[0]]['planets']:
                    labels = ['PLANET '+c, 'UPPER ERR',
                              'LOWER ERR', 'UNITS', 'REF']
                    table = visitor.add_table(clabels=labels,
                                              rows=len(allplanet))
                    for starlabels in allplanet:
                        i = allplanet.index(starlabels)
                        for l in starlabels:
                            j = starlabels.index(l)
                            table.get_cell(i, j).add_primitive(l)
                            elem = starinfo[c][l][-1]
                            if elem:
                                table.get_cell(i, j).add_primitive(elem)
                                pass
                            else:
                                table.get_cell(i, j).add_primitive('NA')
                                pass
                            pass
                        pass
                    pass
                pass
            pass
        return
    pass
# ------------ -------------------------------------------------------
# -- FILTER -- -------------------------------------------------------
class FilterSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['activefilters'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            if len(self['STATUS']) < 3:
                labels = ['Active Filters']
                nf = self['activefilters']['TOTAL']
                table = visitor.add_table(clabels=labels, rows=nf)
                for name in self['activefilters']['NAMES']:
                    i = self['activefilters']['NAMES'].index(name)
                    table.get_cell(i, 0).add_primitive(name)
                    pass
                pass
            if len(self['STATUS']) > 2:
                ignorekeys = ['NAMES', 'TOTAL']
                actflts = [f for f in self['activefilters'].keys() if
                           f not in ignorekeys]
                labels = ['Filter', 'Frames collected']
                table = visitor.add_table(clabels=labels,
                                          rows=len(actflts))
                for flt in actflts:
                    i = actflts.index(flt)
                    number = len(self['activefilters'][flt]['TOTAL'])
                    table.get_cell(i, 0).add_primitive(flt)
                    table.get_cell(i, 1).add_primitive(number)
                    pass
                pass
            pass
        return
    pass
# ------------ -------------------------------------------------------
# -- DATABASE -- -----------------------------------------------------
class DatabaseSV(dawgie.StateVector):
    def __init__(self, name):
        self._version_ = dawgie.VERSION(1,1,1)
        self.__name = name
        self['STATUS'] = excalibur.ValuesList()
        self['name'] = excalibur.ValuesDict()
        self['STATUS'].append(False)
        return

    def name(self):
        return self.__name

    def view(self, visitor:dawgie.Visitor)->None:
        if self['STATUS'][-1]:
            ordlab = ['observatory', 'instrument', 'detector', 'filter', 'mode']
            table = visitor.add_table(clabels=ordlab, rows=1)
            for label in ordlab:
                vlist = [self['name'][n][label] for n in self['name'].keys()]
                i = ordlab.index(label)
                if vlist is not None:
                    for v in set(vlist):
                        total = len(vlist)
                        nb = vlist.count(v)
                        if v is not None:
                            percent = 1e2*vlist.count(v)/total
                            out = (str(v) +
                                   ': ' + str(int(nb)) +
                                   ' (' + str(round(percent)) + '%)')
                            table.get_cell(0, i).add_primitive(out)
                            pass
                        pass
                    pass
                pass
            pass
        return
    pass
# -------------- -----------------------------------------------------
# -- MONITOR -- -------------------------------------------------------
class MonitorSV(dawgie.StateVector):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,1)
        self['last'] = excalibur.ValuesDict()
        self['planet'] = excalibur.ValuesDict()
        self['runid'] = excalibur.ValuesList()
        return

    def name(self):
        return 'parameters'

    def view(self, visitor:dawgie.Visitor)->None:
        for k in sorted(self['last']):
            ks = k.split ('_')
            p = ks[0]
            value = self['last'][k]
            visitor.add_primitive ('Planet ' + p + ' parameter ' +
                                   '_'.join (ks[1:]) + ' last change: ' +
                                   str(value))

            if not numpy.isnan (value):
                values = []
                for v in self['planet'][k]:
                    try: values.append (float(v))
                    except ValueError: values.append (numpy.nan)
                    pass
                fig = bokeh.plotting.figure (title=('Change of ' +
                                                    '_'.join (ks[1:]) +
                                                    ' over RunIDs'),
                                             x_axis_label='Run ID',
                                             y_axis_label='Value')
                # GMR: Pass pylint, to be solved
                # fig.circle (self['runid'], values)
                js,div = bokeh.embed.components (fig)
                visitor.add_declaration (None, div=div, js=js)
                pass
            pass
        return
    pass
# -------------- -----------------------------------------------------
# -- ALERT --- -------------------------------------------------------
class AlertSV(dawgie.StateVector):
    def __init__(self):
        self._version_ = dawgie.VERSION(1,1,1)
        self['changes'] = excalibur.ValuesList()
        self['known'] = excalibur.ValuesList()
        self['table'] = excalibur.ValuesList()
        return

    def name(self):
        return 'parameters'

    def view(self, visitor:dawgie.Visitor)->None:
        visitor.add_declaration ('Last deltas', tag='h4')

        if self['changes']:
            visitor.add_declaration ('', list=True)
            for c in self['changes']: visitor.add_declaration (c, tag='li')
            visitor.add_declaration ('', list=False)
        else: visitor.add_primitive ('No change since last run')

        params = set()
        for te in self['table']: params.update (set (['_'.join(k.split ('_')[1:])
                                                      for k in te.keys()]))
        params = [p for p in sorted (params)]
        row = -1
        table = visitor.add_table(clabels=['target', 'planet'] + params, rows=1)
        for trg,pp in zip (self['known'], self['table']):
            planets = [p for p in sorted (set([k.split('_')[0] for k in pp.keys()]))]
            for planet in planets:
                row += 1
                table.get_cell (row, 0).add_primitive (trg)
                table.get_cell (row, 1).add_primitive (planet)
                for i,param in enumerate (params):
                    k = '_'.join ([planet, param])
                    table.get_cell (row, i+2).add_primitive (str(pp[k])
                                                             if k in pp else '-')
                    pass
                pass
            pass
        return
    pass
# ------------ -------------------------------------------------------
