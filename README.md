# Spike-doctor
Basic Shiny app written in Python for the analysis of whole-cell current clamp recordings focusing mostly on spike detection and supthreshold/suprathreshold feature extraction.

- currently only works with Axon Instruments ABF format and current clamp recordings
- uses the awesome [pyABF](https://github.com/swharden/pyABF) library for reading ABF files and [eFEL](https://github.com/openbraininstitute/eFEL) for feature extraction

# How to use
Clone the repo. 
- requirements.txt file is provided for any python dependencies.
- repo also contains a working [devenv](https://devenv.sh/) setup, which will take care of setting everything up for you including a local virtual environment with a working python version.
- once you set up your Python environment, simply run `shiny run spike-doctor.py` and go to http://127.0.0.1:8000 in your web browser.
