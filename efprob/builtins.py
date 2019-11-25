"""
Built in functions
"""

from .helpers import _prod
from .core import (Space, range_sp, Channel, State, Predicate,
                   uniform_state, idn, discard, proj, truth, point_pred)
import numpy as np
import functools
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


##############################################################
#
# Built in functions for sample spaces
#
##############################################################

#
# Two element sample space, named 2, with elements True, False
#
bool_sp = Space("2", [True, False])

#
# Empty sample space; it can also be obtained as range_sp(0)
#
init_sp = Space(None,[])

#
# Singleton sample space; it can also be obtained as range_sp(1).
# It works as unit for the cartesian product (written as @) of 
# sample spaces.
#
one_sp = Space()

#
# Sample space for a dice 
#
pips_sp = Space("pips", [1,2,3,4,5,6])

#
# Sample space for a coin
#
coin_sp = Space("coin", ['H', 'T'])



##############################################################
#
# Built in functions for channels
#
##############################################################

#
# Abbreviations
#
chan_fromfun = Channel.fromfun
chan_fromklmap = Channel.fromklmap
chan_fromstates = Channel.fromstates


def copy2(sp):
    """ binary copy channel on sample space sp """
    size = sp.size()
    array = np.zeros((size, size, size))
    np.einsum('iii->i', array)[:] = 1.0
    return Channel(array, sp, sp @ sp)


def copy(sp, n=2):
    """ n-ary copy channel on sample space sp, defined recursively """
    if n == 1:
        return idn(sp)
    if n % 2 == 0:
        if n==2:
            return copy2(sp)
        return (copy(sp, n/2) @ copy(sp, n/2)) * copy2(sp)
    return (idn(sp) @ copy(sp, n-1)) * copy2(sp)


def tuple_chan(*chans):
    """ tuple of a list/tuple of channels, of arbitrary length; the domains
    of these channels should all be the same """
    l = len(chans)
    if l < 2:
        raise ValueError('Tupling requires at least two channels')
    c = functools.reduce(lambda x,y: x @ y, chans)
    return c * copy(chans[0].dom, l)


def proj1(sp1, sp2):
    """ shorthand for first projection sp1 @ sp2 -> sp1,
    using the projection function proj defined in core 
    """
    return proj(sp1 @ sp2, 1)


def proj2(sp1, sp2):
    """ shorthand for second projection sp1 @ sp2 -> sp2"""
    return proj(sp1 @ sp2, 2)


def swap(sp1, sp2):
    """ Exchange/swap channel sp1 @ sp2 -> sp2 @ sp1 """
    size1 = sp1.size()
    size2 = sp2.size()
    array = np.zeros((size2, size1, size1, size2))
    np.einsum('ijji->ij', array)[:] = 1.0
    return Channel(array, sp1 @ sp2, sp2 @ sp1)


def uniform_chan(dom, cod):
    """ Channel dom -> cod which consists of uniform states """
    return chan_fromklmap(lambda x: uniform_state(cod), dom, cod)


def random_chan(dom, cod):
    """ Random channel via disintegration of random state on the product """
    return chan_fromklmap(lambda *x: random_state(cod), dom, cod)
#s = random_state(dom @ cod)
#    return s[ [0,1] : [1,0] ]


def convex_sum(prob_item_list):
    """ Yields convex sum

    r1 * a1 + ... + rn * an 

    for input list of the form 
    
    [ (r1, an), ..., (rn, an) ] 

    This function be used for states and channels
    """
    return functools.reduce(operator.add, (r * s for r, s in prob_item_list))


def cocopy2(sp):
    """ binary cocopy (multiset) channel sp @ sp -> sp """
    size = sp.size()
    array = np.zeros((size, size, size))
    np.einsum('iii->i', array)[:] = 1.0
    return Channel(array, sp @ sp, sp)

def cap(sp):
    """ Cap multiset """
    return discard(sp) * cocopy2(sp)



##############################################################
#
# Built in functions for states (distributions / multisets)
#
##############################################################

#
# Abbreviations
#
state_fromfun = State.fromfun

#
# Empty state
#
init_state = State([], init_sp)


def point_state(point, sp):
    """ singleton / Dirac state for the element called point the sample
    space sp 
    """
    if not isinstance(point, tuple):
        point = (point,)
    array = np.zeros(sp.shape)
    array[sp.get_index(*point)] = 1
    return State(array, sp)


def random_state(sp):
    """ Randomly generated state on sample space sp """
    array = np.random.random_sample(sp.shape)
    array = array / array.sum()
    return State(array, sp)


#
# Fair dice
#
dice = uniform_state(pips_sp)

def flip(r, sp=None):
    """ coin with bias r, from the unit interval [0,1], by default on the
    boolean space with elements True, False """
    if isinstance(sp, Space):
        return State([r, 1-r], sp)
    else:
        return State([r, 1-r], bool_sp)

def coin(r):
    return flip(r, coin_sp)


def cup(sp):
    """ cup as multiset """
    size = sp.size()
    array = np.zeros((size, size))
    np.einsum('ii->i', array)[:] = 1
    return State(array, sp @ sp)


def cup_state(sp):
    """ cup, but now normalized to a probability distribution (state) """
    return 1/sp.size() * cup(sp)


def tvdist(s, t):
    """ Total variation distance between discrete states """
    if s.dom != t.dom:
        raise Exception('Distance requires equal spaces')
    return 0.5 * sum(abs(np.ndarray.flatten(s.array) \
                         - np.ndarray.flatten(t.array)))


def Mval(self, data):
    """ Multiple-state (M) validity of data, where data is a multiset of
        predicates. The formula is:
    
        Product over p in data, of (self >= p)^(multiplicity of p)
        """
    vals = [ self.expectation(*a) ** data(*a) for a in data.sp.iter_all() ]
    val = _prod(vals)
    return val


def Mval_point(state, data):
    """ Multiple-state (M) validity, where data is a multiset of
        points, to be used as point predicates"""
    vals = [ state(*a) ** data(*a) for a in data.sp.iter_all() ]
    val = _prod(vals)
    return val


def log_Mval_point(state, data):
    """ multiple-state (M) validity, where data is a multiset of
        points, to be used as point predicates"""
    vals = [ math.log(state(*a) ** data(*a)) for a in data.sp.iter_all() ]
    val = functools.reduce(lambda p1, p2: p1 + p2, vals, 0)
    return val


def Cval(state, data):
    """ Copied-state (C) validity of data, where data is a multiset of
        predicates"""
    def expon(a, b) : return a ** b
    preds = [ expon(*a, data(*a)) for a in data.sp.iter_all() ]
    pred = functools.reduce(lambda p1, p2: p1 & p2, 
                            preds, 
                            truth(state.sp))
    return state >= pred


def Cval_point(state, data):
    """ copied-state (C) validity of data, where data is a multiset of
        points, to be used as point predicates"""
    preds = [ point_pred(*a, state.sp) ** data(*a) for a in data.sp.iter_all() ]
    pred = functools.reduce(lambda p1, p2: p1 & p2, 
                            preds, 
                            truth(state.sp))
    return state >= pred


def Mval_chan(state, chan, data):
    """multiple-state (M) validity, along a channel, where data is a
        pulled back along the channel """
    # for each element, state.sp.get(*a), of the codomain
    vals = [ (chan >> state).expectation(*a) ** data(*a)
             for a in data.sp.iter_all() ]
    val = functools.reduce(lambda p1, p2: p1 * p2, vals, 1)
    return val

def Mval_point_chan(state, chan, data):
    """multiple-state (M) validity, along a channel, where point-data is a
        pulled back along the channel """
    # for each element, state.sp.get(*a), of the codomain
    vals = [ (chan >> state)(*a) ** data(*a) for a in data.sp.iter_all() ]
    val = functools.reduce(lambda p1, p2: p1 * p2, vals, 1)
    return val


def Cval_chan(state, chan, data):
    """ copied-state (C) validity of data, where data is a multiset of
        predicates"""
    def lshift(a, b) : return a << b
    preds = [ lshift(chan, *a) ** data(*a) for a in data.sp.iter_all() ]
    pred = functools.reduce(lambda p1, p2: p1 & p2, 
                            preds, 
                            truth(state.sp))
    return state >= pred


def Cval_point_chan(state, chan, data):
    """ copied-state (C) validity of data, where data is a multiset of
        predicates"""
    preds = [ (chan << point_pred(*a, chan.cod)) ** data(*a)
              for a in data.sp.iter_all() ]
    pred = functools.reduce(lambda p1, p2: p1 & p2, 
                            preds, 
                            truth(state.sp))
    return state >= pred


def Mlrn(state, data):
    """ Multiple-state learning with data, as mulitset of predicates """
    def cond(a,b) : return a / b
    freqs = [ data(*data.sp.get(*a))
              for a in np.ndindex(*data.sp.shape) ]
    freq = sum(freqs)
    stats = [ (data(*a) / freq) * cond(state, *a)
              for a in data.sp.iter_all() ]
    stat = functools.reduce(lambda s1, s2: s1 + s2, stats)
    return stat


def Mlrn_chan(state, chan, data):
    """ Multiple-state learning along a channel with data, 
        as mulitset of predicates """
    def lshift(a,b) : return a << b
    freqs = [ data(*data.sp.get(*a))
              for a in np.ndindex(*data.sp.shape) ]
    freq = sum(freqs)
    stats = [ (data(*a) / freq) * (state / lshift(chan, *a))
              for a in data.sp.iter_all() ]
    stat = functools.reduce(lambda s1, s2: s1 + s2, stats)
    return stat


def Mlrn_point_chan(state, chan, data):
    """ Multiple-state conditioning with data, as mulitset of predicates """
    freqs = [ data(*a) for a in data.sp.iter_all() ]
    freq = sum(freqs)
    stats = [ (data(*a) / freq) * (state / (chan << point_pred(a, chan.cod)))
              for a in data.sp.iter_all() ]
    stat = functools.reduce(lambda s1, s2: s1 + s2, stats)
    return stat


def Clrn(state, data):
    """ Copied-state learning with data, as mulitset of predicates """
    def expon(a, b) : return a ** b
    preds = [ expon(*a, data(*a)) for a in data.sp.iter_all() ]
    pred = functools.reduce(lambda p1, p2: p1 & p2, 
                            preds, 
                            truth(state.sp))
    return state / pred


def Clrn_chan(state, chan, data):
    """ Copied-state learning along a channel with data, 
        as mulitset of predicates """
    def lshift(a,b) : return a << b
    preds = [ lshift(chan, *a) ** data(*a) for a in data.sp.iter_all() ]
    pred = functools.reduce(lambda p1, p2: p1 & p2, 
                            preds, 
                            truth(state.sp))
    return state / pred


def Clrn_point_chan(state, chan, data):
    """ Copied-state conditioning along a channel with data, 
        as mulitset of points predicates """
    def lshift(a,b) : return a << b
    preds = [ (chan << point_pred(a, chan.cod)) ** data(*a)
              for a in data.sp.iter_all() ]
    pred = functools.reduce(lambda p1, p2: p1 & p2, 
                            preds, 
                            truth(state.sp))
    return state / pred


def binomial(N, p):
    """ Binomial distribution on {0,1,2,...,N} with probability p in [0,1] """
    Nfac = math.factorial(N)
    def binom_coeff(k):
        return Nfac / (math.factorial(k) * math.factorial(N-k))
    return State([binom_coeff(k) * (p ** k) * ((1-p) ** (N-k)) 
                  for k in range(N+1)],
                 range_sp(N+1))


def poisson(lam, ub):
    """Poisson distribution with rate parameter `lam' and upperbound ub.
    The distribution is restricted to the finite interval [0, ub-1];
    hence the values have to adjusted so that they sum up to 1 on this
    interval.
    """
    probabilities = [(lam ** k) * (math.e ** -lam) / math.factorial(k) 
                     for k in range(ub)]
    s = sum(probabilities)
    return State([p/s for p in probabilities], range_sp(ub))


def discretized_space(low_bound, up_bound, steps):
    """ Discretization of the real interval [low_bound, up_bound] into
    a sample space with steps many points """ 
    step_size = (up_bound - low_bound) / steps
    points = []
    vals = []
    for i in range(steps):
        p = low_bound + i * step_size
        points = points + [ p ]
    return Space(None, points)

def discretized_state(fun, low_bound, up_bound, steps):
    """ A state with steps many points on the real line as sample space,
    between a lower and an upper bound. On these points the normalized
    values of a (probability density) function fun are the multiplicities.
    """
    dsp = discretized_space(low_bound, up_bound, steps)
    vals = [fun(p) for p in dsp[0].list]
    tot = sum(vals)
    return State([v/tot for v in vals], dsp)


def discretized_uniform(low_bound, up_bound, steps):
    """ Uniform distribution on discretized interval [low_bound, up_bound] """
    return discretized_state(lambda x: 1, low_bound, up_bound, steps)


def discretized_beta(alpha, beta, steps):
    """ Beta distribution on discretized interval [0,1] """
    return discretized_state(lambda x: stats.beta.pdf(x, a = alpha, b = beta),
                             0, 1, steps)

def discretized_exponential(lamb, up_bound, steps):
    """ Exponential distribution on discretized interval [0, up_bound] """
    return discretized_state(lambda x: stats.expon.pdf(x, scale = 1/lamb),
                             0, up_bound, steps)


##############################################################
#
# Built in functions for Predicates (or observables)
#
##############################################################

#
# Abbreviations
#
pred_fromfun = Predicate.fromfun

#
# Point predicates for the two-element Boolean sample space,
# using point_pred function defined in core.
#
yes_pred = point_pred(True, bool_sp)
no_pred = point_pred(False, bool_sp)

# truth predicate is already defined in core

def falsity(sp):
    """ Predicate that is always 0 on sample space sp """
    array = np.zeros(sp.shape)
    return Predicate(array, sp)

def random_pred(sp):
    """ Randomly generated predicate on sample space sp """
    array = np.random.random_sample(sp.shape)
    return Predicate(array, sp)

def eq_pred(sp):
    """ Equality relation on sp @ sp, returning 1 on pairs of equal
    elements and 0 everywhere else
    """
    size = sp.size()
    array = np.zeros((size, size))
    np.einsum('ii->i', array)[:] = 1.0
    return Predicate(array, sp @ sp)


##############################################################
#
# Bayesian network auxiliaries
#
##############################################################

#
# The domain that is standardly used: bnd = Bayesian Network Domain
#
bnd = Space(None, ['t', 'f'])

#
# Basic (sharp) predicates on this domain
#
tt = Predicate([1,0], bnd)
ff = ~tt

def bn_prior(r): 
    """ Function for modelling an initial node, as prior state """
    return State([r, 1-r], bnd)


def bn_pred(r,s):
    """ Function for a predicate on a state, in a Bayesian network """
    return Predicate([r,s], bnd)

bn_pos_pred = bn_pred(1,0)
bn_neg_pred = bn_pred(0,1)


def cpt(*ls):
    """ Conditional probability table converted into a channel. The input is
    a list of probabilities, of length 2^n, where n is the number of
    predecessor nodes.
    """
    n = len(ls)
    if n == 0:
        raise Exception('Conditional probability table must have non-empty list of probabilities')
    log = math.log(n, 2)
    if log != math.floor(log):
        raise Exception('Conditional probability table must have 2^n elements')
    log = int(log)
    sp = functools.reduce(lambda s1, s2: s1 @ s2, [bnd] * log)
    states = [flip(r, bnd) for r in ls]
    return chan_fromstates(states, sp)
