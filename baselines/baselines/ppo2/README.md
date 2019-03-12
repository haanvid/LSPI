# PPO2

- Original paper: https://arxiv.org/abs/1707.06347
- Baselines blog post: https://blog.openai.com/openai-baselines-ppo/
- `mpirun -np 8 python -m baselines.ppo1.run_atari` runs the algorithm for 40M frames = 10M timesteps on an Atari game. See help (`-h`) for more options.
- `python -m baselines.ppo1.run_mujoco` runs the algorithm for 1M frames on a Mujoco environment.

-----------------------------------------------------------
Above is written by the person who modified the baselines code to make lstm policy work on ppo2.

 
# Running codes
```bash
ppo2-lstm/baselines/baselines/ppo2$ python run_roboschool.py 
```
This code only runs for the mujoco env.
ex) Hopper-v2, Ant-v2
