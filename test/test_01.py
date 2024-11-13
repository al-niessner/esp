'''unit tests for keeping pyxb and xsd up to date'''

import hashlib
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
        except: self.assertTrue(False,'problems with excalibur.runtime.binding')

    def test_02(self):
        '''check that the binding is newer than the schema'''
        basedir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..', 'excalibur','runtime'))
        with open(os.path.join(basedir,'autogen.md5'), 'rt') as file:
            for line in file.readlines():
                cksum,fn = line.split()
                with open (os.path.join (basedir,fn), 'br') as f:
                    content = f.read()
                self.assertEqual (cksum, hashlib.md5(content).hexdigest(),
                                  'please run .ci/autogen.sh because checksums '
                                  f'do not match for {fn}')
