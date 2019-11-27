"""Tests for HMM.
"""

from efprob import *
from hmm import *

def test_wheather_hmm():
    W = Space("Weather", ['C', 'S', 'R'])
    A = Space("Activities", ['I', 'O'])
    t = chan_fromstates([State([0.5, 0.2, 0.3], W),
                         State([0.15, 0.8, 0.05], W),
                         State([0.2, 0.2, 0.6], W)], W)
    e = chan_fromstates([flip(0.5,A), flip(0.2,A), flip(0.9,A)], W)
    s = point_state('C', W)
    h = Hmm(s, t, e)
    #
    # Stationary state
    #
    ss = State([0.25, 0.5, 0.25],W)
    assert (t >> ss) == ss
    #
    # Validity of sequence of observations, in two ways
    #
    assert ((((idn(A @ A) @ tuple_chan(e,t)) \
              * (idn(A) @ tuple_chan(e,t)) \
              * tuple_chan(e,t)) >> s).MM(1,1,1,0) \
            >= (point_pred('O',A) @ point_pred('I',A) @ point_pred('I',A))) \
            == 0.1674
    assert (s >= ((e << point_pred('O',A)) & \
                  (t << ((e << point_pred('I',A)) & \
                         (t << (e << point_pred('I',A))))))) == 0.1674
    assert h.forward_of_points(['O', 'I', 'I']) == 0.1674
    assert h.backward_of_points(['O', 'I', 'I']) == 0.1674
    assert np.isclose(h.validity_of_points(['O', 'I', 'I']), 0.1674)
    #
    # Filtering with observations
    #
    s2 = t >> (s / (e << point_pred('O', A)))
    s3 = t >> (s2 / (e << point_pred('I', A)))
    s4 = t >> (s3 / (e << point_pred('I', A)))
    assert s4 == State([1867/6696, 347/1395, 15817/33480], W)
    assert h.filter_of_points(['O', 'I', 'I']) \
        == State([1867/6696, 347/1395, 15817/33480], W)


def test_hallway():
    X = Space("Cells", [1,2,3,4,5])
    Y = Space("Walls", [2,3])
    s = point_state(3,X)
    t = chan_fromstates([State([3/4,1/4,0,0,0],X),
                         State([1/4,1/2,1/4,0,0],X),
                         State([0,1/4,1/2,1/4,0],X),
                         State([0,0,1/4,1/2,1/4],X),
                         State([0,0,0,1/4,3/4],X)], X)
    e = chan_fromstates([State([0,1],Y),
                         State([1,0],Y),
                         State([1,0],Y),
                         State([1,0],Y),
                         State([0,1],Y)], X)
    h = Hmm(s, t, e)
    obs = [2,2,3,2,3,3]
    #
    # Validity computation
    #
    assert (h >= obs) == 3/512
    q5 = e << point_pred(3,Y)
    q4 = (e << point_pred(3,Y)) & (t << q5)
    q3 = (e << point_pred(2,Y)) & (t << q4)
    q2 = (e << point_pred(3,Y)) & (t << q3)
    q1 = (e << point_pred(2,Y)) & (t << q2)
    q0 = (e << point_pred(2,Y)) & (t << q1)
    assert (s >= q0) == 3/512
    #
    # Filtering computation
    #
    s2 = t >> (s / (e << point_pred(2,Y)))
    assert s2 == State([0, 1/4, 1/2, 1/4, 0], X)
    s3 = t >> (s2 / (e << point_pred(2,Y)))
    assert s3 == State([1/16, 1/4, 3/8, 1/4, 1/16], X)
    s4 = t >> (s3 / (e << point_pred(3,Y)))
    assert s4 == State([3/8, 1/8, 0, 1/8, 3/8], X)
    s5 = t >> (s4 / (e << point_pred(2,Y)))
    assert s5 == State([1/8, 1/4, 1/4, 1/4, 1/8], X)
    s6 = t >> (s5 / (e << point_pred(3,Y)))
    assert s6 == State([3/8, 1/8, 0, 1/8, 3/8], X)
    s7 = t >> (s6 / (e << point_pred(3,Y)))
    assert s7 == State([3/8, 1/8, 0, 1/8, 3/8], X)
    assert h.filter_of_points([2, 2, 3, 2, 3, 3]) \
        == State([3/8, 1/8, 0, 1/8, 3/8], X)
    #
    # Most likeli sequence
    #
    assert h.viterbi_of_points(obs) == [3, 2, 1, 2, 1, 1]
