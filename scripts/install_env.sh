#!/bin/bash

conda env create -f ../envs/environment.yaml

conda activate selection_env

# Install pylandau for electron lifetime
pip install --no-build-isolation pylandau

# Install h5flow
pip install ../lib/h5flow

# change directory back to project-CMSE-602
cd ../