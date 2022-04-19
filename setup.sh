#!/bin/bash
# Get the pybullet drones repo
git submodule update --init --recursive

# create the virtual python env
python3 -m venv ./python_virtual_env
source python_virtual_env/bin/activate

# install packages
cd gym-pybullet-drones
pip3 install -e .

cd ..
