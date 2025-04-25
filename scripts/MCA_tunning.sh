#!/bin/bash

# ./dvc.sh -fs train_mca
./dvc.sh -fs train_mca_classifier
python3 ./no_obfuscation/mca.py
python3 ./mca_equivalence_experiment.py