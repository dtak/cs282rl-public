from __future__ import absolute_import, print_function, unicode_literals, division
import numpy as np
import scipy.stats.distributions

from nose.tools import *
from cs282rl import domains
from cs282rl.domains.gridworld import Maze

maze = [
    '###',
    '#o#',
    '#*#',
    '###']

def test_maze_wrapper():
    maze_wrapper = Maze(maze)
    eq_(maze_wrapper[0, 0], '#')
    eq_(maze_wrapper[2, 1], '*')
    assert_raises(IndexError, lambda: maze_wrapper[-1, 0])
    eq_(maze_wrapper.positions_containing('*'), [(2, 1)])
    open_positions = maze_wrapper.positions_not_containing('#')
    eq_(list(sorted(open_positions)), [(1, 1), (2, 1)])


def test_gridworld_basic():
    task = domains.GridWorld(maze)
    start_state = task.observe()
    assert np.isscalar(start_state)
    assert 0 <= start_state < task.num_states
    assert len(task.actions) > 1


def test_has_samples():
    assert 'trivial' in domains.GridWorld.samples


def test_actions():
    task = domains.GridWorld(maze)
    eq_(task.num_actions, len(task.actions))
    is_S_action = [action[0] == 1 and action[1] == 0 for action in task.actions]
    assert np.sum(is_S_action) == 1


def test_goals():
    # Since there is only one possible empty state, we can check the outcomes of all possible actions.
    for test_terminal in [False, True]:
        task = domains.GridWorld(maze, terminal_markers='*' if test_terminal else '')

        task.reset()
        start_state = task.observe()
        eq_(start_state, 1*3 + 1)

        resulting_states = []
        resulting_rewards = []
        for action_idx, action in enumerate(task.actions):
            task.reset()
            cur_state, reward = task.perform_action(action_idx)
            if action[0] == 1 and action[1] == 0:
                # Moving down got us the reward.
                eq_(reward, 10)
                if test_terminal:
                    # Episode ended.
                    assert cur_state is None
                else:
                    assert cur_state == 2*3 + 1
            else:
                eq_(cur_state, start_state)
                eq_(reward, 0)


def test_stochasticity():
    task = domains.GridWorld(maze, action_error_prob=.5)

    # Try to go South. Half the time we'll take a random action, and for 1/4 of
    # those we'll also go South, so we'll get a reward 1/2(1) + 1/2(1/4) = 5/8
    # of the time.
    action_idx = [action[0] == 1 and action[1] == 0 for action in task.actions].index(True)

    times_rewarded = 0
    N = 10000
    for i in range(N):
        task.reset()
        observation, reward = task.perform_action(action_idx)
        if reward:
            times_rewarded += 1

    assert .025 < scipy.stats.distributions.binom.cdf(times_rewarded, N, 5./8) < .975
