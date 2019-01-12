## --< EXCALIBUR >--

Exoplanet Calibration Bayesian Unified Retrieval Pipeline.

EXCALIBUR reduces extrasolar system data into an exoplanet spectrum.
It includes CERBERUS, a line by line, plane parallel radiative transfer code modeling exoplanet atmospheres and a Bayesian parameter retrievial / model selection package. 

EXCALIBUR is an event driven pipeline where the events are defined as changes in data or algorithms; when events are detected, dependencies affected by the changes are re-processed. Calibration steps are transparent and quantified using a combination of accessible intermediate products and auto-generation of visual diagnostics.

### --< Organization >--

#### --< Data >--

Some of the items we do not maintain (external) are static. Static executables and libraries are kept out of the code management.

- /proj/sdp/bin (external executables such as pymc, ...)
- /proj/sdp/lib (external libraries)
- /proj/sdp/pkg (external source code and build area)
- /proj/sdp/data/cal (instrument reference calibration files such as STSCI files, ...)
- /proj/sdp/data/res (resource files such as web interface tools, ...)
- /proj/sdp/data/sci (on-disk data to be processed such as private datasets, ...)

#### --< TASKS / algorithms >--

TARGET
- create (IDs and filters)
- autofill (prior system information from NEXSCI)
- scrape (download and save available data)

SYSTEM
- validate (checks for system parameters completeness)
- finalize (delivers a comprehensive set of system parameters uniformly formatted, allows parameters over-ride)

DATA
- collect (sort data according to filters)
- calibration (extraction, wavelength solution, noise assessment)
- timing (transit, eclipse, full phase curve detection)

TRANSIT
- normalization (scaling of stellar spectrum to Out Of Transit relative quantities)
- whitelight (orbital solution and instrumental behavior recovery)
- spectrum (exoplanet spectrum recovery)

ECLIPSE

PHASECURVE

CERBERUS

### --< Source Code >--

[Github](https://github-fn.jpl.nasa.gov/EXCALIBUR/esp)
...........

