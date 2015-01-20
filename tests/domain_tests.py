from __future__ import absolute_import, print_function, unicode_literals, division
import numpy as np
import scipy.stats.distributions

from nose.tools import *
from cs282rl import domains

maze_structure = [
    '###',
    '#o#',
    '#*#',
    '###']


def setup():
    global maze
    maze = domains.Maze(maze_structure)

def test_maze():
    eq_(maze[0, 0], '#')
    eq_(maze[2, 1], '*')
    assert_raises(IndexError, lambda: maze[-1, 0])
    eq_(maze.positions_containing('*'), [(2, 1)])
    open_positions = maze.positions_not_containing('#')
    eq_(list(sorted(open_positions)), [(1, 1), (2, 1)])


def test_maze_basic():
    task = domains.FullyObservableSimpleMazeTask(maze)
    start_state = task.observe()
    assert 0 <= start_state < task.num_states
    assert len(task.actions) > 1
    cur_state, reward = task.perform_action(1)
    assert np.isscalar(cur_state)
    assert np.isscalar(reward)


def test_actions():
    task = domains.FullyObservableSimpleMazeTask(maze)
    eq_(task.num_actions, len(task.actions))
    is_S_action = [action[0] == 1 and action[1] == 0 for action in task.actions]
    assert np.sum(is_S_action) == 1


def test_goals():
    # Since there is only one possible empty state, we can check the outcomes of all possible actions.
    for absorbing_end_state in [False, True]:
        task = domains.FullyObservableSimpleMazeTask(maze, absorbing_end_state=absorbing_end_state)
        task.reset()
        start_state = task.observe()
        eq_(start_state, 1*3 + 1)

        resulting_states = []
        resulting_rewards = []
        for action_idx, action in enumerate(task.actions):
            task.reset()
            cur_state, reward = task.perform_action(action_idx)
            if action[0] == 1 and action[1] == 0:
                eq_(reward, 10)
                if absorbing_end_state:
                    eq_(cur_state, task.num_states - 1)
                    cur_state, reward = task.perform_action(np.random.choice(4))
                    eq_(cur_state, task.num_states - 1)
                    eq_(reward, 0)
                else:
                    # We got bumped back to start.
                    eq_(cur_state, start_state)
            else:
                eq_(cur_state, start_state)
                eq_(reward, 0)


def test_stochasticity():
    task = domains.FullyObservableSimpleMazeTask(maze, action_error_prob=.5)

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
