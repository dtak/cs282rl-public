from __future__ import absolute_import, print_function, unicode_literals, division
import numpy as np

import pytest
from cs282rl.domains import hiv_treatment

state_dimensions = 6


def test_basic():
    simulator = hiv_treatment.HIVTreatment()
    state = simulator.observe()
    assert state.shape == (state_dimensions,)
    reward, new_state = simulator.perform_action(3)
    assert np.isscalar(reward)
    assert new_state.shape == (state_dimensions,)


def test_batch_generation():
    num_patients = 3
    N = 201
    state_histories, action_histories, reward_histories = (
        hiv_treatment.HIVTreatment.generate_batch(num_patients=num_patients))
    assert state_histories.shape == (num_patients, N, state_dimensions)
    assert action_histories.shape == (num_patients, N)
    assert reward_histories.shape == (num_patients, N)

    state_histories, action_histories, reward_histories = (
        hiv_treatment.HIVTreatment.generate_batch(num_patients=num_patients, policy=hiv_treatment.always_do(2)))

    # Just assert this doesn't crash:
    handles = hiv_treatment.visualize_hiv_history(state_histories[0], action_histories[0])
    hiv_treatment.visualize_hiv_history(state_histories[1], action_histories[1], handles=handles)
