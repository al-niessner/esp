'''science functionality separated from dawgie'''

import logging; log = logging.getLogger(__name__)
import os
import re

from . import binding

ENV_NAME = 'EXCALIBUR_LEVER_AND_KNOB_SETTINGS'

def isolate(sv:{}, table:{str:{}}, tn:str)->None:
    return

def load(sv_dict:{str:{}}, targets)->None:
    '''load the configuation file into state vectors

    1. read the filename from environment variable
    2. load the file and treat it as XML
    3. have pybgen binding module parse the XML
    4. move data into state vectors
    '''
    fn = os.environ[ENV_NAME]
    with open (fn, 'rt', encoding='utf-8') as file: xml = file.read()
    settings = binding.CreateFromDocument(xml)
    controls = sv_dict['controls']
    for knob in controls:
        controls[knob] = controls[knob].new(getattr(settings.controls, knob))
    sv_dict['filters']['excludes'].extend (settings.filters.exclude)
    sv_dict['filters']['includes'].extend (settings.filters.include)
    for pymc in ['cerberus', 'spectrum']:
        cf = getattr(settings.pymc, pymc)
        sv = sv_dict[f'pymc-{pymc}']
        sv['default'] = sv['default'].new (cf.default)
        for override in cf.target:
            sv['overrides'][override.name] = override.steps
    for tn in settings.sequester.target:
        if tn.isRegex:
            regex = re.compile(tn.value())
            matching_targets = filter (lambda t,rex=regex:rex.match(t), targets)
            if matching_targets:
                for mt in matching_targets:
                    sv_dict['sequester']['targets'].append (mt, tn.because)
            else:
                log.warning ('Sequester regex %s produced no matches',
                             tn.value())
        elif tn.value() in targets:
            sv_dict['sequester']['targets'].append ((tn.value(), tn.because))
        else: log.warning ('Sequester target %s is not known', tn.value())
