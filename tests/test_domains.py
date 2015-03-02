from __future__ import absolute_import, print_function, unicode_literals, division
import numpy as np

import pytest
from cs282rl import domains
from cs282rl.domains.gridworld import Maze

maze = [
    '###',
    '#o#',
    '#*#',
    '###']

def test_maze_wrapper():
    maze_wrapper = Maze(maze)
    assert maze_wrapper.flatten_index((2, 1)) == 2 * 3 + 1
    assert maze_wrapper.unflatten_index(2 * 3 + 1) == (2, 1)
    assert maze_wrapper.get_flat(0) == '#'
    assert maze_wrapper.get_unflat((0, 0)) == '#'
    assert maze_wrapper.get_flat(2*3 + 1) == '*'
    assert maze_wrapper.get_unflat((2, 1)) == '*'
    with pytest.raises(IndexError):
        maze_wrapper.get_flat(-1)
    with pytest.raises(IndexError):
        maze_wrapper.get_flat(3 * 4)
    with pytest.raises(IndexError):
        maze_wrapper.get_unflat([-1, 0])
    with pytest.raises(IndexError):
        maze_wrapper.get_unflat([3, 4])
    assert maze_wrapper.flat_positions_containing('*') == [2 * 3 + 1]
    open_positions = maze_wrapper.flat_positions_not_containing('#')
    assert list(sorted(open_positions)) == [1 * 3 + 1, 2 * 3 + 1]


def test_gridworld_basic():
    task = domains.GridWorld(maze)
    start_state = task.observe()
    assert np.isscalar(start_state)
    assert 0 <= start_state < task.num_states
    assert len(task.actions) == 4
    assert task.num_actions == 4


def test_has_samples():
    assert 'trivial' in domains.GridWorld.samples


def test_actions():
    task = domains.GridWorld(maze)
    assert task.num_actions == len(task.actions)
    is_S_action = [action[0] == 1 and action[1] == 0 for action in task.actions]
    assert np.sum(is_S_action) == 1


def test_goals():
    # Since there is only one possible empty state, we can check the outcomes of all possible actions.
    for test_terminal in [False, True]:
        task = domains.GridWorld(maze, terminal_markers='*' if test_terminal else '')

        task.reset()
        start_state = task.observe()
        assert start_state == 1*3 + 1
        assert not task.is_terminal(start_state)

        resulting_states = []
        resulting_rewards = []
        for action_idx, action in enumerate(task.actions):
            task.reset()
            cur_state, reward = task.perform_action(action_idx)
            if action[0] == 1 and action[1] == 0:
                # Moving down got us the reward.
                assert reward == 10
                assert cur_state == 2*3 + 1
                if test_terminal:
                    # This state is "terminal": nothing does anything here.
                    assert task.is_terminal(cur_state)
                    for action2 in task.actions:
                        cur_state, reward = task.perform_action(action_idx)
                        assert cur_state == 2*3 + 1
                        assert reward == 0
                else:
                    assert not task.is_terminal(cur_state)
            else:
                assert cur_state == start_state
                assert reward == 0


def test_old_API_goals():
    # This is exactly the same as the old test_goals, so we make sure not to break people using the old API.

    # Since there is only one possible empty state, we can check the outcomes of all possible actions.
    for test_terminal in [False, True]:
        task = domains.GridWorld(maze, terminal_markers='*' if test_terminal else '')

        task.reset()
        start_state = task.observe_old()
        assert start_state == 1*3 + 1

        resulting_states = []
        resulting_rewards = []
        for action_idx, action in enumerate(task.actions):
            task.reset()
            cur_state, reward = task.perform_action_old(action_idx)
            if action[0] == 1 and action[1] == 0:
                # Moving down got us the reward.
                assert reward == 10
                if test_terminal:
                    # Episode ended.
                    assert cur_state is None
                    assert task.observe_old() is None
                else:
                    assert cur_state == 2*3 + 1
            else:
                assert cur_state == start_state
                assert reward == 0


def trials_required_to_bound_error(epsilon, delta):
    # Hoeffding bound: P(|err| > epsilon) < 2exp(-2*n*epsilon**2)
    return int(-np.log(delta/2) / (2 * epsilon**2) + 10)


def test_stochasticity():
    task = domains.GridWorld(maze, action_error_prob=.5)

    # Try to go South. Half the time we'll take a random action, and for 1/4 of
    # those we'll also go South, so we'll get a reward 1/2(1) + 1/2(1/4) = 5/8
    # of the time.
    action_idx = [action[0] == 1 and action[1] == 0 for action in task.actions].index(True)

    times_rewarded = 0
    epsilon = .01
    N = trials_required_to_bound_error(epsilon=epsilon, delta=.0001)
    for i in range(N):
        task.reset()
        observation, reward = task.perform_action(action_idx)
        if reward:
            times_rewarded += 1

    correct_prob = 5./8
    assert np.abs(times_rewarded / N - correct_prob) < epsilon


def test_as_mdp():
    tiny_maze = ['o.*']
    task = domains.GridWorld(tiny_maze, action_error_prob=0., rewards={'*': 10, 'moved': -1, 'hit-wall': -1}, terminal_markers='*', directions="NSEW")
    transition_probabilities, rewards = task.as_mdp()
    def only(state):
        res = [0] * 3
        res[state] = 1.
        return res
    assert np.allclose(transition_probabilities, [
        [only(0), only(0), only(1), only(0)],
        [only(1), only(1), only(2), only(0)],
        [only(2), only(2), only(2), only(2)]])


def equal_ignoring_nan(a, b):
    return np.all(np.isclose(a, b) | np.isnan(a - b))

def test_as_mdp_stochastic():
    tiny_maze = ['o.*']
    task = domains.GridWorld(tiny_maze, action_error_prob=.5, rewards={'*': 10, 'moved': -1, 'hit-wall': -1}, terminal_markers='*', directions="NSEW")
    transition_probabilities, rewards = task.as_mdp()

    # Conservatively, use a union bound for the independent estimations for each state transition probability.
    epsilon = .1
    N = trials_required_to_bound_error(epsilon=epsilon, delta=.0001) * task.num_states
    transitions_observed = np.zeros(task.num_states)
    rewards_observed = np.zeros(task.num_states)
    for state in range(task.num_states):
        for action in range(task.num_actions):
            transitions_observed.fill(0)
            rewards_observed.fill(np.nan)
            for i in range(N):
                task.state = state
                new_state, reward = task.perform_action(action)
                transitions_observed[new_state] += 1
                rewards_observed[new_state] = reward
            print(state, action)
            assert np.all(np.abs(transitions_observed / N - transition_probabilities[state, action]) < epsilon)
            assert equal_ignoring_nan(rewards_observed, rewards[state, action])
