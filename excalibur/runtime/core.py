'''science functionality separated from dawgie'''

import os

from . import binding

ENV_NAME = 'EXCALIBUR_LEVER_AND_KNOB_SETTINGS'

def load()->binding.lever_type:
    fn = os.environ[ENV_NAME]
    with open (fn, 'rt') as file: xml = file.read()
    return binding.CreateFromDocument(xml)
