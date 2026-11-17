import numpy as np
from efprob import Space, Channel, SpaceAtom, State

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from BayesianSurgeryNet import BayesianSurgeryNet

# We use the data from the smoking example
omega1 = [
    0.5,   # S=0, T=0, C=0
    0.1,   # S=0, T=0, C=1
    0.01,  # S=0, T=1, C=0
    0.02,  # S=0, T=1, C=1
    0.1,   # S=1, T=0, C=0
    0.05,  # S=1, T=0, C=1
    0.02,  # S=1, T=1, C=0
    0.2    # S=1, T=1, C=1
]
vars1 = ['S', 'T', 'C']

omega2 = [
    0.5,   # S=0, T=0, C=0
    0.01,  # S=0, T=1, C=0
    0.1,   # S=0, T=0, C=1
    0.02,  # S=0, T=1, C=1
    0.1,   # S=1, T=0, C=0
    0.02,  # S=1, T=1, C=0
    0.05,  # S=1, T=0, C=1
    0.2    # S=1, T=1, C=1
]
vars2 = ['S', 'C', 'T']

space1 = Space(*[SpaceAtom(var, [0, 1]) for var in vars1])
bsn1 = BayesianSurgeryNet(omega1, vars1)
bsn2 = BayesianSurgeryNet(omega2, vars2)

def test_comb_disint():
    # Given
    expected_f = np.array([
        [[0.525, 0.21],
        [0.105, 0.42]],
        [[0.24666667, 0.03363636],
        [0.12333333, 0.33636364]]
    ])

    expected_g = np.array([
        [0.95238095, 0.40540541],
        [0.04761905, 0.59459459]
    ])

    state = State(omega1, space1)

    # When
    # Disintegrate on S and observe T
    f, g = bsn1._comb_disint(state, Space(bsn1.sp[0]), [1, 0, 0], Space(bsn1.sp[1]), [0, 0, 1])

    # Then
    assert(g.array.shape == expected_g.shape)
    assert(f.array.shape == expected_f.shape)
    
    assert np.allclose(f.array, expected_f)
    assert np.allclose(g.array, expected_g)

def test_comb_compose():
    # Given
    expected_f = np.array([
        [[0.525, 0.21],
        [0.105, 0.42]],
        [[0.24666667, 0.03363636],
        [0.12333333, 0.33636364]]
    ])

    expected_g = np.array([
        [0.95238095, 0.40540541],
        [0.04761905, 0.59459459]
    ])

    expected_omega_cut = np.array([
        [[0.36746032, 0.10873016],
        [0.00580087, 0.01800866]],
        [[0.15641892, 0.04628378],
        [0.07243243, 0.22486486]]
    ])

    # When
    composed_state = bsn1._comb_compose(
        Channel(expected_f, Space(bsn1.sp[1]), Space(bsn1.sp[0], bsn1.sp[1])),
        Channel(expected_g, Space(bsn1.sp[0]), Space(bsn1.sp[1])),
        Space(bsn1.sp[0]),
        Space(bsn1.sp[1]),
        Space(bsn1.sp[2]),
    )

    # Then
    assert composed_state.array.shape == expected_omega_cut.shape
    assert np.allclose(composed_state.array, expected_omega_cut)

def test_cut_and_compute():
    # Given
    expected_results = np.array([
        [[0.74652237, 0.4577027 ],
        [0.25347763, 0.5422973 ]]
    ])

    # When
    result = bsn1.cut_and_compute('S', 'C')

    # Then
    assert np.allclose(result.array, expected_results)

def test_reorder():
    # Given
    permutation = ['S', 'C', 'T']
    new_omega = [
        0.5,   # S=0, T=0, C=0
        0.01,  # S=0, T=1, C=0
        0.1,   # S=0, T=0, C=1
        0.02,  # S=0, T=1, C=1
        0.1,   # S=1, T=0, C=0
        0.02,  # S=1, T=1, C=0
        0.05,  # S=1, T=0, C=1
        0.2    # S=1, T=1, C=1
    ]

    expected_state = State(new_omega, Space(*[SpaceAtom(var, [0, 1]) for var in permutation]))

    # Then
    state = bsn1._reorder_states(permutation)

    # Expect
    assert state == expected_state

def test_reorder2():
    # Given
    permutation = ['T', 'S', 'C']
    new_omega = [
        0.5,   # S=0, T=0, C=0
        0.1,   # S=0, T=0, C=1
        0.1,   # S=1, T=0, C=0
        0.05,  # S=1, T=0, C=1
        0.01,  # S=0, T=1, C=0
        0.02,  # S=0, T=1, C=1
        0.02,  # S=1, T=1, C=0
        0.2    # S=1, T=1, C=1
    ]

    expected_state = State(new_omega, Space(*[SpaceAtom(var, [0, 1]) for var in permutation]))

    # Then
    state = bsn1._reorder_states(permutation)

    # Expect
    assert state == expected_state

def test_cut_and_compute2():
    # Given
    expected_results = np.array([
        [[0.74652237, 0.4577027 ],
        [0.25347763, 0.5422973 ]]
    ])

    # Then
    result = bsn2.cut_and_compute('S', 'C')

    # Expect
    assert np.allclose(result.array, expected_results)

