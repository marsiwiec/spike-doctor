# Spike-doctor
Basic Shiny app written in Python for the analysis of whole-cell current clamp recordings focusing mostly on spike detection and supthreshold/suprathreshold feature extraction.

Currently only works with Axon Instruments ABF format and current clamp recordings.

Uses the awesome [pyABF](https://github.com/swharden/pyABF) library for reading ABF files and [eFEL](https://github.com/openbraininstitute/eFEL) for feature extraction.

<img src="https://github.com/marsiwiec/spike-doctor/blob/main/assets/spike-doctor.png?raw=true" width="80%">

# How to use
Clone the repo. 

A requirements.txt file is provided for any python dependencies.

The repo also contains a working [devenv](https://devenv.sh/) setup, which will take care of setting everything up for you including a local virtual environment with a working python version.

Once you set up your Python environment, simply run `shiny run spike-doctor.py` and go to http://127.0.0.1:8000 in your web browser.

# eFEL features

Scroll down the sidebar to choose which exact features you want included in the analysis results table. A sensible default has been preselected. The complete list of eFEL features along with their explanations is available as part of the [eFEL documentation](https://efel.readthedocs.io/en/latest/eFeatures.html).
