#!/usr/bin/env python
import argparse
import os, sys
import numpy as np
from baselines import bench, logger

def train(env_id, num_timesteps, seed, d_targ, load, point):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import LstmMlpPolicy, MlpPolicy
    import gym
    # import roboschool
    import multiprocessing
    import tensorflow as tf
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk

    set_global_seeds(seed)

    ncpu = multiprocessing.cpu_count()
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    
    nenvs = 32
    env = SubprocVecEnv([make_env(i) for i in range(nenvs)])
    env = VecNormalize(env)

    policy = MlpPolicy

    def adaptive_lr(lr, kl, d_targ):
        if kl < (d_targ / 1.5):
            lr *= 2.
        elif kl > (d_targ * 1.5):
            lr *= .5
        return lr


    ppo2.learn(policy=policy, env=env, nsteps=512, nminibatches=4,
        lam=0.95, gamma=0.99, noptepochs=15, log_interval=1,
        ent_coef=0.00,
        lr=adaptive_lr,
        cliprange=0.2,
        total_timesteps=num_timesteps,
        load=load,
        point=point,
        init_targ=d_targ)

def test(env_id, num_timesteps, seed, curr_path, point):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalizeTest
    from baselines.ppo2 import ppo2
    from baselines.ppo2.policies import LstmMlpPolicy, MlpPolicy
    import gym
    # import roboschool
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecTestEnv

    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        return env
    env = DummyVecTestEnv([make_env])
    running_mean = np.load('{}/log/mean.npy'.format(curr_path))
    running_var = np.load('{}/log/var.npy'.format(curr_path))
    env = VecNormalizeTest(env, running_mean, running_var)

    set_global_seeds(seed)
    policy = MlpPolicy

    ppo2.test(policy=policy, env=env, nsteps=2048, nminibatches=32, 
        load_path='{}/log/checkpoints/{}'.format(curr_path, point))


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Acrobot-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=100)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--d_targ', type=float, default=0.012)
    parser.add_argument('--point', type=str, default='00050')
    args = parser.parse_args()
    curr_path = sys.path[0]
    if args.train:
        logger.configure(dir='{}/log'.format(curr_path))
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            d_targ=args.d_targ, load=args.load, point=args.point)
    else:
        test(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            curr_path=curr_path, point=args.point)


if __name__ == '__main__':
    main()

