from __future__ import absolute_import, print_function, unicode_literals, division
import numpy as np
import scipy.stats.distributions

import pytest
from cs282rl import domains
from cs282rl.domains.gridworld import Maze


def test_chainworld_goals():
    task = domains.ChainWorld(left_length=1, left_reward=10, right_length=3, right_reward=500, on_chain_reward=-1, p_return_to_start=0)
    assert task.num_states == 5
    assert task.num_actions == 2
    assert task.observe() == 1
    assert not task.is_terminal(task.observe())

    # try moving left (action 0)
    state, reward = task.perform_action(0)
    assert state == 0
    assert reward == 10
    assert task.is_terminal(state)

    # try resetting.
    task.reset()
    assert task.observe() == 1

    # try moving right (action 1)
    state, reward = task.perform_action(1)
    assert state == 2
    assert reward == -1
    assert not task.is_terminal(state)

    state, reward = task.perform_action(1)
    assert state == 3
    assert reward == -1
    assert not task.is_terminal(state)

    state, reward = task.perform_action(1)
    assert state == 4
    assert reward == 500
    assert task.is_terminal(state)


def test_chainworld_return_to_start():
    task = domains.ChainWorld(left_length=1, left_reward=10, right_length=500, right_reward=500, on_chain_reward=-1, p_return_to_start=.5)

    times_at_start = 0
    task.reset()
    N = 10000
    for i in range(N):
        state, reward = task.perform_action(1)
        if state == 1:
            times_at_start += 1

    threshold = .01
    assert threshold < scipy.stats.distributions.binom.cdf(times_at_start, N, 1/2) < (1 - threshold)


def test_chainworld_max_reward():
    task = domains.ChainWorld(left_length=1, left_reward=10, right_length=500, right_reward=500, on_chain_reward=-1, p_return_to_start=.5)
    assert task.get_max_reward() == 500
