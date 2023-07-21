'''Classifier Database Products View'''
# -- IMPORTS --------------------------------------------------------
import dawgie
import excalibur
import matplotlib.pyplot as plt
import io

# -- SV -------------------------------------------------------------
class PredictSV(dawgie.StateVector):
    '''PredictSV ds'''
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
                visitor.add_declaration('PLANET: ' + p)
                visitor.add_declaration('PREDICTION: ' + str(self['data'][p]['prediction']))
#                 allwhite = self['data'][p]['allwhite']
#                 postlc = self['data'][p]['postlc']
#                 postsep = self['data'][p]['postsep']
#                 myfig = plt.figure(figsize=(10, 6))
#                 plt.title(p)
#                 plt.plot(postsep, allwhite, 'o')
#                 plt.plot(postsep, postlc, '^', label='M')
#                 plt.xlabel('Star / Planet separation [R$^*$]')
#                 plt.ylabel('Normalized Post White Light Curve')
#                 plt.legend(bbox_to_anchor=(1 + 0.1*(0.5), 0.5),
#                            loc=5, ncol=1, mode='expand', numpoints=1,
#                            borderaxespad=0., frameon=False)
#                 plt.tight_layout(rect=[0,0,(1 - 0.1),1])
#                 buf = io.BytesIO()
#                 myfig.savefig(buf, format='png')
#                 visitor.add_image('...', ' ', buf.getvalue())
#                 # plt.show()
#                 plt.close(myfig)
                pass
            pass
        pass
    pass

# Summarize flags for each target
class Flags_SV(dawgie.StateVector):
    '''Flags_SV ds'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,0,0)
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

        flag_algs_info = {
            'count_points_wl': {
                'title': 'Points in Full and Total Transit',
                'field_descriptions': {
                    'full': 'Number of points in full transit',
                    'total': 'Number of points in total transit'
                }
            },
            'symmetry_wl': {
                'title': 'Light Curve Symmetry',
                'field_descriptions': {
                    'left': 'Number of points before transit',
                    'right': 'Number of points after transit'
                }
            },
            'rsdm': {
                'title': 'Residual Standard Deviation Mean (RSDM)',
                'field_descriptions': {
                    'mean_rsdm': 'Average RSDM'
                }
            },
            'perc_rejected': {
                'title': 'Percent Rejected',
                'field_descriptions': {
                    'percent_rejected_value': 'Percentage of spectral channels rejected in cumulative spectrum distribution'
                }
            },
            'median_error': {
                'title': 'Median Error',
                'field_descriptions': {
                    'median_error_value': 'Median error'
                }
            }
        }

        if self['STATUS'][-1]:

            for k in self['data'].keys():

                if k not in flag_algs_info:  # if it's a planet

                    visitor.add_declaration("_____")

                    visitor.add_declaration("PLANET: " + str(k))

                    visitor.add_declaration("Overall Flag: " + str(self['data'][k]['overall_flag']))

                    for alg in self['data'][k].keys():

                        if alg != 'overall_flag':
                            visitor.add_declaration(str(flag_algs_info[alg]['title']).upper())

                            visitor.add_declaration("Flag: " + str(self['data'][k][alg]['flag_color']))
                            visitor.add_declaration("Flag Description: " + str(self['data'][k][alg]['flag_descrip']))

                            for field in self['data'][k][alg].keys():
                                if field not in ("flag_color", "flag_descrip"):
                                    visitor.add_declaration(str(flag_algs_info[alg]['field_descriptions'][field]) + ": " + str(self['data'][k][alg][field]))

                else:  # if it's a metric that's not planet-specific

                    visitor.add_declaration("_____")

                    visitor.add_declaration(str(flag_algs_info[k]['title']).upper())

                    visitor.add_declaration("Flag: " + str(self['data'][k]['flag_color']))
                    visitor.add_declaration("Flag Description: " + str(self['data'][k]['flag_descrip']))

                    for field in self['data'][k].keys():
                        if field not in ("flag_color", "flag_descrip"):
                            visitor.add_declaration(str(flag_algs_info[k]['field_descriptions'][field]) + ": " + str(self['data'][k][field]))

        pass
    pass

# Summarize flags across ALL targets
class Flag_Summary_SV(dawgie.StateVector):
    '''Flag_Summary_SV ds'''
    def __init__(self, name):
        '''__init__ ds'''
        self._version_ = dawgie.VERSION(1,0,0)
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

        if 'classifier_flags' in self['data']:

            flag_colors = ['green', 'yellow', 'red']
            vlabels = list(self['data']['classifier_flags'].keys())
            hlabels = ['Data Quality Metric', 'Green', 'Yellow', 'Red']

            table = visitor.add_table(clabels=hlabels, rows=len(vlabels))

            # label rows with algorithm names
            for row, ele in enumerate(vlabels):
                table.get_cell(row, 0).add_primitive(vlabels[row])

                # to work around pylint "unused-variable" warning and "enumerate" requirement
                if ele:
                    pass

            # go through all the classifier algorithms
            for a, vlabel in enumerate(vlabels):

                alg_data = self['data']['classifier_flags'][vlabels[a]]

                # to work around pylint "unused-variable" warning and "enumerate" requirement
                if vlabel:
                    pass

                # go through all potential flag colors
                for i, c in enumerate(flag_colors):

                    # to work around pylint "unused-variable" warning and "enumerate" requirement
                    if c:
                        pass

                    color = flag_colors[i]
                    table_column = i + 1  # add 1 to account for 'Data Quality Metric' header.

                    if color in alg_data:
                        flag_count = alg_data[color]
                        table.get_cell(a, table_column).add_primitive(flag_count)
                    else:
                        table.get_cell(a, table_column).add_primitive(0)

            pass

        # note that 'gold' below refers to a yellow flag. the 'gold' matplotlib color to represent yellow.
        flag_algs_info = {
            'count_points_wl': {
                'suptitle': 'Points in Transit',
                'subplot_titles': {
                    'full': 'Number of points in full transit',
                    'total': 'Number of points in total transit',
                    'x_axis_label': 'Number of points'
                }
            },
            'symmetry_wl': {
                'suptitle': 'Light Curve Symmetry',
                'subplot_titles': {
                    'left': 'Number of points before transit',
                    'right': 'Number of points after transit',
                    'x_axis_label': 'Number of points'
                },
                'thresh_vals':{
                    'red': 0
                }
            },
            'rsdm': {
                'suptitle': 'Residual Standard Deviation Metric (RSDM)',
                'subplot_titles': {
                    'mean_rsdm': 'Average RSDM'
                },
                'thresh_vals':{
                    'gold': 8.804,
                    'red': 13.818
                }
            },
            'perc_rejected': {
                'suptitle': 'Spectral Channels Rejected',
                'subplot_titles': {
                    'percent_rejected_value': 'Percent Rejected'
                },
                'thresh_vals':{
                    'green': 0,
                    'gold': 31.844,
                    'red': 52.166
                }
            },
            'median_error': {
                'suptitle': 'Median Error',
                'subplot_titles': {
                    'median_error_value': 'Median error'
                },
                'thresh_vals':{
                    'gold': 2.5
                }
            }
        }

        if 'classifier_vals' in self['data']:

            for metric in self['data']['classifier_vals']:

                metric_info = self['data']['classifier_vals'][metric]

                suptitle = flag_algs_info[metric]['suptitle']

                for count, subplot in enumerate(metric_info):

                    subplot_title = flag_algs_info[metric]['subplot_titles'][subplot]
                    points_to_plot = metric_info[subplot]

                    plt.subplot(1, len(metric_info), (count + 1))
                    plt.hist(points_to_plot, edgecolor='navy', color='cornflowerblue')

                    if 'x_axis_label' in flag_algs_info[metric]['subplot_titles']:
                        plt.xlabel(flag_algs_info[metric]['subplot_titles']['x_axis_label'])
                        plt.title(subplot_title)
                    else:
                        plt.xlabel(subplot_title)

                    plt.suptitle(str(suptitle) + ' Across Targets')

                    # plot threshold lines
                    if 'thresh_vals' in flag_algs_info[metric]:
                        for color in flag_algs_info[metric]['thresh_vals']:
                            plt.axvline(flag_algs_info[metric]['thresh_vals'][color], color=color, linestyle='dashed', linewidth=1)

                    if metric == 'median_error':
                        plt.ylabel('Number of targets')
                    else:
                        plt.ylabel('Number of planets')

                # display the figure
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                visitor.add_image('...', ' ', buf.getvalue())
                plt.close()

        pass
