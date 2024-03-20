'''science functionality separated from dawgie'''

import os

from . import binding

ENV_NAME = 'EXCALIBUR_LEVER_AND_KNOB_SETTINGS'

def load(sv_dict:{str:{}})->binding.lever_type:
    fn = os.environ[ENV_NAME]
    with open (fn, 'rt') as file: xml = file.read()
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
