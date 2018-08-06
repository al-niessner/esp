## EXCALIBUR

###The Exoplanet Calibration Bayesian Unified Retrieval Pipeline.

EXCALIBUR is event driven where the events are defined as changes in data or algorithms; when events are detected, dependencies affected by the changes are re-processed. Calibration steps are transparent and quantified using a combination of accessible intermediate state vectors, auto-generation of calibration step documentation, and statistical metrics. EXCALIBUR implements a unified marginalization over both science and instrument model parameters coupled with a Bayesian, evidence-based, model selection capability.

### Organization

#### Data

Some of the items we do not maintain or are simply static. Executables and libraries that we do not maintain are kept out of the code management (here). Instead, the data and binary tool dependencies have been delivered in the following directories:

- /proj/sdp/bin --> tool executables
- /proj/sdp/lib --> tool libraries
- /proj/sdp/pkg --> tool source and build area

- /proj/sdp/data/cal --> calibration data for processing data sets
- /proj/sdp/data/res --> resource files
- /proj/sdp/data/sci --> actual science data to be processed

Calibration data are calibration files from STSCI

Resource data is that which the program needs every time it runs. An example would be the icon for a button in a GUI interface.

Science data is the primary information from a instrument.

#### Souce Code

The source code is then organized by language:
- Bash : used to make Python elements more accessible to the command line
- C : used for github hooks
- Python : bulk of the code

The Python has view key packages:
- exo.spec.ae : the algorithm engine developed by the scientists

### Documentation

[Fundamental Magic](https://github.jpl.nasa.gov/pages/niessner/sdp/Notebook/Fundamentals-Magic.slides.html) is a manager level explanation of what the pipeline does and how it can help development.

### [FAQ](https://github.jpl.nasa.gov/niessner/sdp/wiki/FAQs)
### [How To](https://github.jpl.nasa.gov/niessner/sdp/wiki/HOWTOs)
