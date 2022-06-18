# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from envs.sibrivalry.ant_maze.create_maze_env import create_maze_env

import gym
import numpy as np
# import torch
from gym.utils import seeding

class AntMazeEnv(gym.GoalEnv):
  """Wraps the HIRO/SR Ant Environments in a gym goal env."""
  def __init__(self, variant='AntMaze-SR', eval=False):

    self.done_env = False
    if eval:
      self.dist_threshold = 5.0
    else:
      self.dist_threshold = np.sqrt(2)
    state_dims = 30
    
    
    mazename = variant.split('-')
    if len(mazename) == 2:
      mazename, test_goals = mazename
      assert mazename == 'AntMaze'
      self.goal_dims = [0, 1]
      self.eval_dims = [0, 1]
      if test_goals == 'SR':
        self.sample_goal = lambda: self.np_random.uniform([-3.5, 12.5], [3.5, 19.5]).astype(np.float32)
        self.dist_threshold = 1.0
        if eval:
          self.done_env = True
      else: # HIRO VERSION
        self.sample_goal = lambda: np.array([0., 16.], dtype=np.float32)
    else:
      mazename = mazename[0]
      self.goal_dims = [0, 1, 3, 4]
      self.eval_dims = [0, 1, 2, 3]
      state_dims = 33
      if mazename == 'AntPush':
        self.sample_goal = lambda: np.array([0., 19., 2., 8.], dtype=np.float32)
        if eval:
          self.eval_dims = [0, 1]
      elif mazename == 'AntFall':
        self.sample_goal = lambda: np.array([0., 27., 8., 16.], dtype=np.float32)
        if eval:
          self.eval_dims = [0, 1]
      else:
        raise ValueError('Bad maze name!')


    self.maze = create_maze_env(mazename) # this returns a gym environment
    self.seed()
    self.max_steps = 500
    self.dist_threshold = 1.0


    self.action_space = self.maze.action_space
    observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dims,)) 
    goal_space        = gym.spaces.Box(-np.inf, np.inf, (len(self.goal_dims),)) # first few coords of state
    self.observation_space = gym.spaces.Dict({
        'observation': observation_space,
        'desired_goal': goal_space,
        'achieved_goal': goal_space
    })
    self.num_steps = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.maze.wrapped_env.seed(seed)
    return [seed]



  def step(self, action):
    next_state, _, _, _ = self.maze.step(action)
    next_state = next_state.astype(np.float32)

    s_xy = next_state[self.goal_dims]
    reward = self.compute_reward(s_xy, self.g_xy, None)
    info = {}
    self.num_steps += 1
    
    is_success = np.allclose(0., reward)
    done = is_success and self.done_env
    info['is_success'] = is_success
    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    obs = {
      'observation': next_state,
      'achieved_goal': s_xy,
      'desired_goal': self.g_xy,
    }
          
    return obs, reward, done, info

  def reset(self): 
    self.num_steps = 0

    ## Not exactly sure why we are reseeding here, but it's what the SR env does
    _ = self.maze.wrapped_env.seed(self.np_random.randint(np.iinfo(np.int32).max))
    s = self.maze.reset().astype(np.float32)
    _ = self.maze.wrapped_env.seed(self.np_random.randint(np.iinfo(np.int32).max))
    self.g_xy = self.sample_goal()

    return {
      'observation': s,
      'achieved_goal': s[self.goal_dims],
      'desired_goal': self.g_xy,
    }

  def render(self):
    self.maze.render()

  def compute_reward(self, achieved_goal, desired_goal, info):
    if len(achieved_goal.shape) == 2:
      ag = achieved_goal[:,self.eval_dims]
      dg = desired_goal[:,self.eval_dims]
    else:
      ag = achieved_goal[self.eval_dims]
      dg = desired_goal[self.eval_dims]
    d = np.linalg.norm(ag - dg, axis=-1)
    return -(d >= self.dist_threshold).astype(np.float32)

class AntMazeEnvFull(gym.GoalEnv):
  """Wraps the HIRO/SR Ant Environments in a gym goal env."""
  def __init__(self, variant='AntMaze-SR', eval=False):

    self.done_env = False
    if eval:
      self.dist_threshold = 5.0
    else:
      self.dist_threshold = np.sqrt(2)
    state_dims = 29
    
    
    mazename = variant.split('-')
    if len(mazename) == 2:
      mazename, test_goals = mazename
      assert mazename == 'AntMaze'
      self.goal_dims = list(range(29))
      self.eval_dims = [0, 1]
      if test_goals == 'SR':
        self.sample_goal = lambda: self.np_random.uniform([-3.5, 12.5], [3.5, 19.5]).astype(np.float32)
        self.dist_threshold = 1.0
        if eval:
          self.done_env = True
      else: # HIRO VERSION
        self.sample_goal = lambda: np.array([0., 16.], dtype=np.float32)
    else:
      mazename = mazename[0]
      self.goal_dims = [0, 1, 3, 4]
      self.eval_dims = [0, 1, 2, 3]
      state_dims = 33
      if mazename == 'AntPush':
        self.sample_goal = lambda: np.array([0., 19., 2., 8.], dtype=np.float32)
        if eval:
          self.eval_dims = [0, 1]
      elif mazename == 'AntFall':
        self.sample_goal = lambda: np.array([0., 27., 8., 16.], dtype=np.float32)
        if eval:
          self.eval_dims = [0, 1]
      else:
        raise ValueError('Bad maze name!')


    self.maze = create_maze_env(mazename) # this returns a gym environment
    self.seed()
    self.max_steps = 500
    self.dist_threshold = 1.0


    self.action_space = self.maze.action_space
    observation_space = gym.spaces.Box(-np.inf, np.inf, (state_dims,)) 
    goal_space        = gym.spaces.Box(-np.inf, np.inf, (len(self.goal_dims),)) # first few coords of state
    self.observation_space = gym.spaces.Dict({
        'observation': observation_space,
        'desired_goal': goal_space,
        'achieved_goal': goal_space
    })
    self.num_steps = 0

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    self.maze.wrapped_env.seed(seed)
    return [seed]



  def step(self, action):
    next_state, _, _, _ = self.maze.step(action)
    next_state = next_state.astype(np.float32)

    s_xy = next_state[self.goal_dims]
    reward = self.compute_reward(s_xy, self.g_xy, None)
    info = {}
    self.num_steps += 1
    
    is_success = np.allclose(0., reward)
    done = is_success and self.done_env
    info['is_success'] = is_success
    if self.num_steps >= self.max_steps and not done:
      done = True
      info['TimeLimit.truncated'] = True

    obs = {
      'observation': next_state,
      'achieved_goal': s_xy,
      'desired_goal': self.g_xy,
    }
          
    return obs, reward, done, info

  def reset(self): 
    self.num_steps = 0

    ## Not exactly sure why we are reseeding here, but it's what the SR env does
    _ = self.maze.wrapped_env.seed(self.np_random.randint(np.iinfo(np.int32).max))
    s = self.maze.reset().astype(np.float32)
    _ = self.maze.wrapped_env.seed(self.np_random.randint(np.iinfo(np.int32).max))
    other_dims = np.array([8.19890515e-01,  9.95145602e-01,
        3.48547286e-02, -6.19350100e-02,  6.80766655e-02, -4.42372144e-02,
       -4.81461428e-02,  1.45511675e-02, -7.75746132e-02,  5.00618279e-02,
        3.65949561e-02,  4.99939194e-02,  1.02664477e-02, -2.27597298e-01,
        8.01758031e-02, -8.81610163e-02,  7.77121806e-02,  3.36722131e-02,
        2.27648027e-02, -2.24103019e-02, -3.77942221e-02, -6.56355237e-02,
        1.35722257e-01,  6.93039877e-02, -1.71162114e-01, -1.12083335e-01,
        1.76819156e-02])
    self.g_xy = np.concatenate((self.sample_goal(), other_dims))

    return {
      'observation': s,
      'achieved_goal': s[self.goal_dims],
      'desired_goal': self.g_xy,
    }

  def render(self, mode):
    return self.maze.render(mode)

  def compute_reward(self, achieved_goal, desired_goal, info):
    if len(achieved_goal.shape) == 2:
      ag = achieved_goal[:,self.eval_dims]
      dg = desired_goal[:,self.eval_dims]
    else:
      ag = achieved_goal[self.eval_dims]
      dg = desired_goal[self.eval_dims]
    d = np.linalg.norm(ag - dg, axis=-1)
    return -(d >= self.dist_threshold).astype(np.float32)