import math
import numpy as np
import efprob as ep

def test_validity():
    s = ep.State([0.2, 0.8], ep.bool_sp)
    p = ep.Predicate([1, 0.5], ep.bool_sp)
    assert math.isclose(s >= p, 0.6)

def test_state_trans():
    s = ep.State([0.2, 0.8], ep.bool_sp)
    c = ep.Channel([0.1, 0.3,
                    0.9, 0.7], ep.bool_sp, ep.bool_sp)
    s2 = ep.State([0.26, 0.74], ep.bool_sp)
    assert c >> s == s2

def test_iter_all():
    sp = ep.bool_sp @ ep.range_sp(5)
    assert (list(sp.iter_all()) ==
            [(True, 0), (True, 1), (True, 2), (True, 3), (True, 4),
             (False, 0), (False, 1), (False, 2), (False, 3), (False, 4)])

def test_channel():
    p = ep.Predicate([1, 0.5], ep.bool_sp)
    c = ep.Channel([0.1, 0.3,
                    0.9, 0.7], ep.bool_sp, ep.bool_sp)
    s = ep.State([2/11, 9/11], ep.bool_sp)
    assert (c / p)(True) == s
