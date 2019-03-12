"""
Environments and wrappers for Sonic training .
"""
import gym
import numpy as np

# from retro_contest.local import make
# from baselines.common.atari_wrappers import WarpFrame, FrameStack
# import gym_remote.client as grc
import queue

class DelayObsRew(gym.Wrapper):
    def __init__(self, env, queue_len, no_op_act):
        print(locals())
        gym.Wrapper.__init__(self, env)
        self.queue_len = queue_len
        self.no_op_act = no_op_act
        self.init_reward_list = []


    def _insert_queue(self, q, q_size, item):
        dequed_item = None
        if q.qsize() == q_size:
            dequed_item = q.get_nowait()
        q.put_nowait(item)
        return q, dequed_item

    def reset(self):
        self.act_q = queue.Queue(self.queue_len)
        # self.rew_q = queue.Queue(self.queue_len)
        init_ob = self.env.reset()

        # Since the agent does not get any obs during the first k timesteps (when obs are k steps delayed)
        # When called env.reset(), the env_wrapper internally runs for k timesteps
        # and fill the queues and return the very first ob.
        # The action will be a "no-op"
        self.init_reward_list = []
        for i in range(self.queue_len):
            # print("%r (%s)" % (self.no_op_act, type(self.no_op_act)))
            # ob, reward, done, info = self.env.step(self.no_op_act)
            # Obs are not used for initial n delayed timesteps (self.queue_len)
            # _ = self._insert_queue(self.obs_q, self.queue_len, ob)
            # Rewards for the initial n delayed timesteps should be "0"
            # since they are added by the agent.
            init_ob, reward, done, info = self.env.step(self.no_op_act)
            tmp1, tmp2 = self._insert_queue(self.act_q, self.queue_len, self.no_op_act)
            self.init_reward_list.append(reward)

        return init_ob, np.array(self.init_reward_list)

    def step(self, action):


        # print("Real ob & rew : ",ob,", ",reward)
        # input("Press Enter")
        # prev_q = self.act_q
        self._act_q, dq_a  = self._insert_queue(self.act_q, self.queue_len, action)
        ob, reward, done, info = self.env.step(dq_a)
        # make information state
        ob = np.hstack([ob, np.array(list(self.act_a.queue))])

        return ob, reward, done, info


def make_delayed_env(env_id='MountainCar-v0', delay_steps=0, no_op_act=0):
    """
    Create an environment with some standard wrappers.
    """
    # env = grc.RemoteEnv('tmp/sock')
    # env_id = env_id.split(',')
    env = gym.make(env_id)
    if delay_steps != 0:
        env = DelayObsRew(env, queue_len = delay_steps, no_op_act = no_op_act)

    # env = gym.wrappers.TimeLimit(env, max_episode_steps=(50+delay_steps))
    # env = EditReward(env)
    # if stack:
    #     env = FrameStack(env, 4)
    return env


# def make_env_grc(stack=True, scale_rew=True):
#     env = grc.RemoteEnv('tmp/sock')
#     env = SonicDiscretizer(env)
#     if scale_rew:
#         env = RewardScaler(env)
#     env = WarpFrame(env)
#     env = EditReward(env)
#     if stack:
#         env = FrameStack(env, 4)
#     return env

#
# def make_env_id(env_id, delay_steps, no_op_act):
#     def fun():
#         return make_env(env_id=env_id, delay_steps=delay_steps, no_op_act =no_op_act )
#
#     return fun

