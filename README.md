# Vision Based Stabalisation
### About
In this repo, I have created an AI system capable of basic quadcopter flight by implementing a reinforcement learning (RL) algorithm (Twin Delayed Deep Deterministic Policy Gradient (TD3)), the state of the art in RL. The model takes in sensor data from the simulation and outputs rotor speeds between 0-100 in order to balance the quadcopter.

Something that I found when building this is that it makes a lot more sense to use traditional control algorithms for the stability of the quadcopter, and to use the RL algorithm to plan routes and manoeuvres of varying levels of aggression.

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
