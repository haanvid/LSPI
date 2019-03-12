#!/usr/bin/env python
import argparse
import os, sys
import numpy as np
from baselines import bench, logger
from act_delay_gym_wrapper_aug_mdp import make_delayed_env
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def train(env_id, num_timesteps, seed, d_targ, load, point, delayed_steps, no_op_act):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.ppo2 import ppo2_aug_mdp
    from baselines.ppo2.policies import LstmMlpPolicy, MlpPolicy
    import gym
    # import roboschool
    import multiprocessing
    import tensorflow as tf
    from baselines.common.vec_env.subproc_vec_env_aug_mdp import SubprocVecEnv
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    def make_env(rank, delayed_steps, no_op_act):
        def _thunk():
            # env = gym.make(env_id)
            env = make_delayed_env(env_id=env_id, delayed_steps=delayed_steps, no_op_act=no_op_act)
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
    env = SubprocVecEnv([make_env(i, delayed_steps, no_op_act) for i in range(nenvs)])
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
        init_targ=d_targ,
        delayed_steps= delayed_steps)

def test(env_id, num_timesteps, seed, curr_path, point, delayed_steps, np_op_act):
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
    running_mean = np.load('{}/mlp-delay{}-aug_mdp-log/mean.npy'.format(curr_path,delayed_steps))
    running_var = np.load('{}/mlp-delay{}-aug_mdp-log/var.npy'.format(curr_path, delayed_steps))
    env = VecNormalizeTest(env, running_mean, running_var)

    set_global_seeds(seed)
    policy = MlpPolicy

    ppo2.test(policy=policy, env=env, nsteps=2048, nminibatches=32, 
        load_path='{}/mlp-delay{}-aug_mdp-log/checkpoints/{}'.format(curr_path, delayed_steps, point), delayed_steps=delayed_steps)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--env', help='environment ID', default='Ant-v2')
    # hopper: 11 observations and 3 actions
    parser.add_argument('--env', help='environment ID', default='Hopper-v2')

    parser.add_argument('--delayed', help='delayed steps', type=int, default=20)
    parser.add_argument('--seed', help='RNG seed', type=int, default=100)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--load', type=bool, default=False)
    parser.add_argument('--d_targ', type=float, default=0.012)
    parser.add_argument('--point', type=str, default='00050')
    args = parser.parse_args()
    curr_path = sys.path[0]
    if args.train:
        logger.configure(dir='{}/mlp-delay{}-aug_mdp-log'.format(curr_path, args.delayed))
        train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            d_targ=args.d_targ, load=args.load, point=args.point, delayed_steps=args.delayed, no_op_act=np.array([0,0,0]))
    else:
        test(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
            curr_path=curr_path, point=args.point, delayed_steps=args.delayed, np_op_act=np.array([0,0,0]))


if __name__ == '__main__':
    main()

