#! /usr/bin/env python3

import os
import setuptools

excalibur = os.path.join ('excalibur', '__init__.py')
deps =['arviz==0.12.0',
       'astropy==5.0.4',
       'joblib==1.1.0',
       'ldtk==1.7.0',
       'lmfit==1.0.3',
       'pymc3==3.11.5',
       'scipy==1.7.3',
       'scikit-learn==1.1.0',
       ]
version = 'esp-git-rev'
with open (os.path.join (os.path.dirname (__file__), excalibur)) as f: t = f.read()
t = t.replace ('${UNDEFINED}', version)
with open (os.path.join (os.path.dirname (__file__), excalibur), 'tw') as f:\
     f.write (t)
setuptools.setup (name='excalibur',
                  version='0.0.0',
                  packages=setuptools.find_packages(),
                  setup_requires=deps,
                  src_root=os.path.abspath (os.path.dirname (__file__)),
                  install_requires=deps,
                  package_data={},
                  author='Gael Roudier',
                  author_email='Gael.Roudier@jpl.caltech.edu',
                  description='',
                  license='''*******************************************************************************
 **
 **           Copyright 2018, by the California Institute of Technology
 **    ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged.
 ** Any commercial use must be negotiated with the Office of Technology Transfer
 **                  at the California Institute of Technology.
 **
 **  This software may be subject to U.S. export control laws and regulations.
 **  By accepting this document, the user agrees to comply with all applicable
 **                        U.S. export laws and regulations.
 **   User has the responsibility to obtain export licenses, or other export
 **  authority as may be required before exporting such information to foreign
 **                countries or providing access to foreign persons.
 ******************************************************************************
 ** NTR: 50482''',
                  keywords='baysian mcmc',
                  url='https://github-fn.jpl.nasa.gov/EXCALIBUR/esp',
                  zip_safe=False)
