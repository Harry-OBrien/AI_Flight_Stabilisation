# Vision Based Stabalisation

setup:
```
# Get the pybullet drones repo
git submodule update --init --recursive

# create the virtual python env
# python3 -m venv ./py_venv  # Uncomment this line for the first time setting up the repo
source py_venv/bin/activate

# install packages
cd gym-pybullet-drones
pip3 install -e .

cd ..

```