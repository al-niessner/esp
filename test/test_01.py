'''unit tests for keeping pyxb and xsd up to date'''

import os
import unittest

class Pyxb(unittest.TestCase):
    def test_01(self):
        '''check that binding can be imported

        This will check that a recent enough version of pyxb is installed
        and that it matches the version that generated excalibur.runtime.binding
        '''
        try:
            import excalibur.runtime.binding
            self.assertTrue(True)
        except: self.assertTrue(False)

    def test_02(self):
        '''check that the binding is newer than the schema'''
        basedir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..', 'excalibur','runtime'))
        bin = os.path.getmtime(os.path.join (basedir, 'binding.py'))
        xsd = os.path.getmtime(os.path.join (basedir, 'levers.xsd'))
        self.assertLessEqual (xsd, bin, 'pyxbgen --schema-location=excalibur/runtime/levers.xsd --module=binding --module-prefix=excalibur.runtime')
