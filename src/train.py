#! /usr/bin/env python3
"""
PPO: Proximal Policy Optimization

Written by Patrick Coady (pat-coady.github.io)

PPO uses a loss function and gradient descent to approximate
Trust Region Policy Optimization (TRPO). See these papers for
details:

TRPO / PPO:
https://arxiv.org/pdf/1502.05477.pdf (Schulman et al., 2016)

Distributed PPO:
https://arxiv.org/abs/1707.02286 (Heess et al., 2017)

Generalized Advantage Estimation:
https://arxiv.org/pdf/1506.02438.pdf

And, also, this GitHub repo which was helpful to me during
implementation:
https://github.com/joschu/modular_rl

This implementation learns policies for continuous environments
in the OpenAI Gym (https://gym.openai.com/). Testing was focused on
the MuJoCo control tasks.
"""
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import monicars as mc
import numpy as np
from policy import Policy
from value_function import NNValueFunction
import scipy.signal
from utils import Logger, Scaler
from datetime import datetime
import os
import argparse
import signal
import pickle
import numbers
import math
from pytz import timezone


class GracefulKiller:
    """ Gracefully exit program on CTRL-C """
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        self.kill_now = True


def init_env(env_name, **kwargs):
    """
    Initialize environment, return dimension of observation
    and action spaces.

    Args:
        env_name: str environment name (e.g. "two_lanes_two_way")

    Returns: 3-tuple
        gym environment (object)
    """
    env = mc.Environment(env_name, **kwargs)

    return env


def move(x, y, heading, speed, acc):
    '''moves an npc to a new position, right now only functions with
    acceleration actions

    Args:
        acc: acceleration to be applied
        x: previous x position to iterate on
        y: previous y position to iterate on
        heading: the heading of the npc
        speed: previous speed of the npc
    Return:
        List which represents new state of npc
    '''
    speed += acc
    delta_x = speed * math.sin(heading)
    delta_y = speed * math.cos(heading)
    x += delta_x
    y += delta_y
    return [x, y, heading, speed]




def tracks_cars_off_screen(prev_obs, actions):
    '''Tracks cars off screen so that the observations do not go to zero

    Args:
        obs: list of current observations of npcs and agent
        prev_obs: list of previous observations of just npcs
        actions: list of actions for npcs

    Returns:
        list of new observations
    '''
    npcs_info = prev_obs[4:]
    xs = npcs_info[0::4]
    ys = npcs_info[1::4]
    headings = npcs_info[2::4]
    speeds = npcs_info[3::4]
    new_npc_obs = []
    for x, y, h, s, a in zip(xs, ys, headings, speeds, actions):
        moved = move(x, y, h, s, a)
        new_npc_obs += moved
    return prev_obs[0:4]+new_npc_obs




def distances(obs):
    '''Calculates the agents distance to the obstacle

    Args:
        obs: list of current observations

    Returns:
        float distance between agent and npc
    '''
    a = np.array(obs[0:2])
    b = np.array(obs[4:6])
    dist = np.linalg.norm(a-b)
    return dist



def run_episode(env, policy, scaler):
    """ Run single episode with option to animate

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range

    Returns: 4-tuple of NumPy arrays
        observes: shape = (episode len, obs_dim)
        actions: shape = (episode len, act_dim)
        rewards: shape = (episode len,)
        unscaled_obs: useful for training scaler, shape = (episode len, obs_dim)
        unscaled_augie: useful for training scaler, shape = (episode len, obs_dim)
    """
    obs = env.reset()
    obs = np.array(obs) # need to add general track cars when they go off the screen
    observes, actions, rewards, unscaled_obs = [], [], [], []
    done = False
    step = 0.0
    scale, offset = scaler.get()
    scale[-1] = 1.0  # don't scale time step feature
    offset[-1] = 0.0  # don't offset time step feature
    while not done:
        obs = obs.astype(np.float32).reshape((1, -1))
        obs = np.append(obs, [[step]], axis=1)  # add time step feature
        unscaled_obs.append(obs)
        obs = (obs - offset) * scale  # center and scale observations
        observes.append(obs)
        action = policy.sample(obs).reshape((1, -1)).astype(np.float32)
        actions.append(action)
        accel_rand = np.random.normal(0,0.05)
        act = action.tolist()[0]
        if len(act) == 1:
            act += [0]
        obs, reward, done = env.step(act, npc_action = [[accel_rand,0]])
        prev_obs = obs
        dist = distances(obs)
        #obs += [dist]
        obs = np.array(obs)
        if not isinstance(reward, numbers.Real):
            reward = np.asscalar(reward)
        rewards.append(reward)
        step += 1e-3  # increment time step feature
        if step == 1.5: #bounding the number of steps that can be taken this should be environment specific but it is to prevent long episodes
    	    done = True
    return (np.concatenate(observes), np.concatenate(actions),
            np.array(rewards, dtype=np.float64), np.concatenate(unscaled_obs), env.npc_manager[0].crash*1)


def run_policy(env, policy, scaler, logger, episodes):
    """ Run policy and collect data for a minimum of min_steps and min_episodes

    Args:
        env: ai gym environment
        policy: policy object with sample() method
        scaler: scaler object, used to scale/offset each observation dimension
            to a similar range
        logger: logger object, used to save stats from episodes
        episodes: total episodes to run

    Returns: list of trajectory dictionaries, list length = number of episodes
        'observes' : NumPy array of states from episode
        'actions' : NumPy array of actions from episode
        'rewards' : NumPy array of (un-discounted) rewards from episode
        'unscaled_obs' : NumPy array of (un-discounted) rewards from episode
    """
    total_steps = 0
    tot_stuck = 0
    trajectories = []
    for e in range(episodes):
        observes, actions, rewards, unscaled_obs, stuck = run_episode(env, policy, scaler)
        total_steps += observes.shape[0]
        trajectory = {'observes': observes,
                      'actions': actions,
                      'rewards': rewards,
                      'unscaled_obs': unscaled_obs}
        trajectories.append(trajectory)
        tot_stuck += stuck 
    unscaled = np.concatenate([t['unscaled_obs'] for t in trajectories])
    rew = np.concatenate([t['rewards'] for t in trajectories])
    scaler.update(unscaled, rew)  # update running statistics for scaling observations
    logger.log({'_MeanReward': np.mean([t['rewards'].sum() for t in trajectories]),
                'Steps': total_steps, 'total leader stucks': tot_stuck})

    return trajectories, tot_stuck


def discount(x, gamma):
    """ Calculate discounted forward sum of a sequence at each point """
    return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]


def add_disc_sum_rew(trajectories, gamma, mu, sig):
    """ Adds discounted sum of rewards to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        gamma: discount

    Returns:
        None (mutates trajectories dictionary to add 'disc_sum_rew')
    """
    for trajectory in trajectories:
        rewards = normalize_rew(trajectory, mu, sig, gamma)
        disc_sum_rew = discount(rewards, gamma)
        trajectory['disc_sum_rew'] = disc_sum_rew

def normalize_rew(trajectory, mu, sig, gamma, original = False):
    if original:
        if gamma < 0.999:  # don't scale for gamma ~= 1
            rewards = trajectory['rewards'] * (1 - gamma)
        else:
            rewards = trajectory['rewards']
    else:
        if sig == 0:
            rewards = (trajectory['rewards'])
        else:
            rewards = (trajectory['rewards'])/np.sqrt(sig)
    return rewards

def add_value(trajectories, val_func):
    """ Adds estimated value to all time steps of all trajectories

    Args:
        trajectories: as returned by run_policy()
        val_func: object with predict() method, takes observations
            and returns predicted state value

    Returns:
        None (mutates trajectories dictionary to add 'values')
    """
    for trajectory in trajectories:
        observes = trajectory['observes']
        values = val_func.predict(observes)
        trajectory['values'] = values


def add_gae(trajectories, gamma, lam, mu, sig):
    """ Add generalized advantage estimator.
    https://arxiv.org/pdf/1506.02438.pdf

    Args:
        trajectories: as returned by run_policy(), must include 'values'
            key from add_value().
        gamma: reward discount
        lam: lambda (see paper).
            lam=0 : use TD residuals
            lam=1 : A =  Sum Discounted Rewards - V_hat(s)

    Returns:
        None (mutates trajectories dictionary to add 'advantages')
   for trajectory in trajectories:
        # Lucas' correction
    Returns:
        None (mutates trajectories dictionary to add 'advantages')
    """
    for trajectory in trajectories:
        rewards = normalize_rew(trajectory, mu, sig, gamma)
        values = trajectory['values']
        # temporal differences
        tds = rewards - values + np.append(values[1:] * gamma, 0)
        advantages = discount(tds, gamma * lam)
        trajectory['advantages'] = advantages


def build_train_set(trajectories):
    """

    Args:
        trajectories: trajectories after processing by add_disc_sum_rew(),
            add_value(), and add_gae()

    Returns: 4-tuple of NumPy arrays
        observes: shape = (N, obs_dim)
        actions: shape = (N, act_dim)
        advantages: shape = (N,)
        disc_sum_rew: shape = (N,)
    """
    observes = np.concatenate([t['observes'] for t in trajectories])
    unscaled_observes = np.concatenate([t['unscaled_obs'] for t in trajectories])
    actions = np.concatenate([t['actions'] for t in trajectories])
    disc_sum_rew = np.concatenate([t['disc_sum_rew'] for t in trajectories])
    advantages = np.concatenate([t['advantages'] for t in trajectories])
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    return observes, actions, advantages, disc_sum_rew, unscaled_observes


def log_batch_stats(observes, actions, advantages, disc_sum_rew, logger, episode):
    """ Log various batch statistics """
    logger.log({'_mean_obs': np.mean(observes),
                '_min_obs': np.min(observes),
                '_max_obs': np.max(observes),
                '_std_obs': np.mean(np.var(observes, axis=0)),
                '_mean_act': np.mean(actions),
                '_min_act': np.min(actions),
                '_max_act': np.max(actions),
                '_std_act': np.mean(np.var(actions, axis=0)),
                '_mean_adv': np.mean(advantages),
                '_min_adv': np.min(advantages),
                '_max_adv': np.max(advantages),
                '_std_adv': np.var(advantages),
                '_mean_discrew': np.mean(disc_sum_rew),
                '_min_discrew': np.min(disc_sum_rew),
                '_max_discrew': np.max(disc_sum_rew),
                '_std_discrew': np.var(disc_sum_rew),
                '_Episode': episode
                })


def main(env_name, num_episodes, gamma, lam, kl_targ, batch_size, hid1_mult,
         policy_logvar, print_results, act_dim, obs_dim, **kwargs):
    """ Main training loop

    Args:
        env_name: OpenAI Gym environment name, e.g. 'Hopper-v1'
        num_episodes: maximum number of episodes to run
        gamma: reward discount factor (float)
        lam: lambda from Generalized Advantage Estimate
        kl_targ: D_KL target for policy update [D_KL(pi_old || pi_new)
        batch_size: number of episodes per policy training batch
        hid1_mult: hid1 size for policy and value_f
        (mutliplier of obs dimension)
        policy_logvar: natural log of initial policy variance
    """
    killer = GracefulKiller()
    env = init_env(env_name, **kwargs)
    # add 1 to obs dimension for time step feature (see run_episode())
    obs_dim += 1
    tz = timezone('America/Montreal') # Montreal Timezone
    dt = datetime.now(tz) # Create unique directories
    now = dt.strftime('%Y-%m-%d %H_%M_%S')
    logger = Logger(logname=env_name, now = now)
    aigym_path = os.path.join('/tmp', env_name, now)
    scaler = Scaler(obs_dim)
    val_func = NNValueFunction(obs_dim, hid1_mult)
    policy = Policy(obs_dim, act_dim, kl_targ, hid1_mult, policy_logvar)
    # run a few episodes of untrained policy to initialize scaler:
    run_policy(env, policy, scaler, logger, episodes=5)
    episode = 0
    kl_terms = np.array([])
    beta_terms = np.array([])
    if print_results:
        rew_graph = np.array([])
        mean_rew_graph = np.array([])
        dir = './log-files/' + env_name + '/' + now + '/'
    while episode < num_episodes:
        trajectories, tot_stuck = run_policy(env, policy, scaler, logger,
                                  episodes=batch_size)
        episode += len(trajectories)
        # add estimated values to episodes
        add_value(trajectories, val_func)
        # calculated discounted sum of Rs
        add_disc_sum_rew(trajectories, gamma, scaler.mean_rew,
                         np.sqrt(scaler.var_rew))
        add_gae(trajectories, gamma, lam, scaler.mean_rew,
                np.sqrt(scaler.var_rew))  # calculate advantage
        disc0 = [t['disc_sum_rew'][0] for t in trajectories]
        # concatenate all episodes into single NumPy arrays
        observes, actions, advantages, disc_sum_rew, unscaled_observes = build_train_set(trajectories)
        # add various stats to training log:
        log_batch_stats(observes, actions, advantages, disc_sum_rew, logger,
                        episode)
        policy.update(observes, actions, advantages, logger)  # update policy
        val_func.fit(observes, disc_sum_rew, logger)  # update value function
        logger.write(display=True)  # write logger results to file and stdout
        if killer.kill_now:
            if raw_input('Terminate training (y/[n])? ') == 'y':
                break
            killer.kill_now = False
        if print_results:
            rew_graph = np.append(rew_graph,disc0)
            x1 = list(range(1,(len(rew_graph)+1)))
            rewards = plt.plot(x1,rew_graph)
            plt.title('Standard PPO')
            plt.xlabel("Episode")
            plt.ylabel("Discounted sum of rewards")
            plt.savefig( "log-learning_curve.png")
            plt.close()
            mean_rew_graph = np.append(mean_rew_graph,np.mean(disc0))
            x2 = list(range(1,(len(mean_rew_graph)+1)))
            mean_rewards = plt.plot(x2,mean_rew_graph)
            plt.title('Standard PPO')
            plt.xlabel("Batch")
            plt.ylabel("Mean of Last Batch")
            plt.savefig("learning_curve2.png")
            plt.close()
    if print_results:
        print('running simulations')
        tr, tot_stuck = run_policy(env, policy, scaler, logger, episodes=1000)
        print('done')
        sum_rewww = [t['rewards'].sum() for t in tr]
        sum_rewww += tot_stuck
        hist_dat = np.array(sum_rewww)
        fig = plt.hist(hist_dat,bins=2000, edgecolor='b', linewidth=1.2)
        plt.title('Standard PPO')
        plt.xlabel("Sum of Rewards")
        plt.ylabel("Frequency")
        plt.savefig("standard_ppo.png")
        plt.close()
        with open('sum_rew_final_policy.pkl', 'wb') as f:
            pickle.dump(sum_rewww, f)
        logger.final_log()
    logger.close()
    policy.close_sess()
    val_func.close_sess()
