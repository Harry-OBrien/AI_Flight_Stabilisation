# Vision Based Stabalisation
### The Report
There is a detailed report in this repo on exactly how I implemented and optimised this system. Check it out [here](https://github.com/Harry-OBrien/AI_Flight_Stabilisation/blob/master/AI%20Flight%20Report.pdf).

### Run it yourself! 🥳
##### Setup
```
# Get the pybullet drones repo
git submodule update --init --recursive

# create the virtual python env
python3 -m venv ./py_venv
source py_venv/bin/activate

# install packages
cd gym-pybullet-drones
pip3 install -e .

cd ..

```
##### Run
```
python3 train_main.py
python3 run_main.py
```
