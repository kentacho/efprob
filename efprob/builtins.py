"""
Built in functions
"""

import functools
import math
import random
import operator
import numpy as np
import scipy.stats as stats
import scipy.special
from .helpers import _prod
from .core import (Space, range_sp, Channel, State, Predicate, convex_sum,
                   uniform_state, idn, discard, proj, truth, falsum, point_pred)

binom = scipy.special.binom

def identity(x): return x

##############################################################
#
# Basic functions for sample spaces
#
##############################################################

#
# Two element sample space, named 2, with elements True, False
#
bool_sp = Space("2", [True, False])

#
# Empty sample space; it can also be obtained as range_sp(0)
#
init_sp = Space(None, [])

#
# Singleton sample space; it can also be obtained as range_sp(1).
# It works as unit for the cartesian product (written as @) of
# sample spaces.
#
one_sp = Space()

#
# Sample space for a dice
#
pips_sp = Space("pips", [1, 2, 3, 4, 5, 6])

#
# Sample space for a coin
#
coin_sp = Space("coin", ['H', 'T'])


##############################################################
#
# Basic channels
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
        if n == 2:
            return copy2(sp)
        return (copy(sp, n / 2) @ copy(sp, n / 2)) * copy2(sp)
    return (idn(sp) @ copy(sp, n - 1)) * copy2(sp)


def tuple_chan(*chans):
    """ tuple of a list/tuple of channels, of arbitrary length; the domains
    of these channels should all be the same """
    l = len(chans)
    if l < 2:
        raise ValueError('Tupling requires at least two channels')
    c = functools.reduce(lambda x, y: x @ y, chans)
    return c * copy(chans[0].dom, l)


def proj1(sp1, sp2):
    """ shorthand for first projection sp1 @ sp2 -> sp1,
    using the projection function proj defined in core
    """
    return proj(sp1 @ sp2, 1)


def proj2(sp1, sp2):
    """ shorthand for second projection sp1 @ sp2 -> sp2"""
    return proj(sp1 @ sp2, 2)


def unif_proj(K):
    """ Uniform projection channel sp ** (K+1) -> sp** K """
    def out_fun(sp):
        dom_sp = sp ** (K+1)
        cod_sp = sp ** K
        def chan_fun(*xs):
            pairs = []
            for i in range(K+1):
                ys = list(xs).copy()
                del ys[i]
                pairs += [ (1/(K+1), point_state(tuple(ys), cod_sp)) ]
            return convex_sum(pairs)
        return chan_fromklmap(chan_fun, dom_sp, cod_sp)
    return out_fun


# def probproj(K):
#     """ channel sp**(K+1) -> sp**K for removing each element """
#     def out_fun(sp):
#         dom_sp = sp ** (K+1)
#         cod_sp = sp ** K
#         tuple_len = len(sp)
#         def chan_fun(*t):
#             weighted_states = []
#             for i in range(K+1):
#                 weighted_states += [ (1/(K+1), 
#                                       point_state(t[:i*tuple_len] + \
#                                                   t[(i+1)*tuple_len:], 
#                                                   cod_sp)) ]
#             return convex_sum(weighted_states)
#         return chan_fromklmap(chan_fun, dom_sp, cod_sp)
#     return out_fun


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
    """ Random channel via pointwise random states """
    return chan_fromklmap(lambda *x: random_state(cod), dom, cod)


def cocopy2(sp):
    """ binary cocopy (multiset, non-probabilistic) channel sp @ sp -> sp """
    size = sp.size()
    array = np.zeros((size, size, size))
    np.einsum('iii->i', array)[:] = 1.0
    return Channel(array, sp @ sp, sp)


def cap(sp):
    """ Cap multiset sp @ sp -> 1 """
    return discard(sp) * cocopy2(sp)



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


def masks(N):
    """ all masks, i.e. lists of zeros and ones only, of length N """
    seqs = [ [] ]
    for i in range(N):
        new_seqs = [ ] 
        for s in seqs:
            new_seqs += [ [0] + s, [1] + s ]
        seqs = new_seqs
    return seqs


def all_sharp_predicates(sp):
    N = sp.size()
    seqs = masks(N)
    return [ Predicate(s, sp) for s in seqs ]


def free_pred_ext(K):
    """ free extension of predicate to multisets, of size K """
    def out_fun(pred):
        sp = pred.sp
        dom = multiset_space(K)(sp)
        def out_fun(mult): 
            r = 1
            for indices in np.ndindex(sp.shape):
                r *= pred(*sp.get(*indices)) ** mult.array[indices]
            return r
        return pred_fromfun(out_fun, dom)
    return out_fun


def random_test(dom, N):
    """ random N-test of predicates on space dom, obtained from 
    channel dom -> N """
    c = random_chan(dom, range_sp(N))
    return [ pred_fromfun(lambda x: c(x)(i), dom) for i in range(N) ]



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
# Empty multiset, on empty space
#
init_state = State([], init_sp)


def empty_state(sp):
    """ empty multiset on sample space sp """
    return state_fromfun(lambda *x: 0, sp)


def unit_state(sp):
    """ constant one multiset on sample space sp """
    return state_fromfun(lambda *x: 1, sp)


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


def random_multiset(K):
    """ Randomly generated multiset of size K on sample space sp """
    def out_fun(sp):
        N = sp.size()
        ls = []
        for i in range(N-1):
            r = random.randint(0, K - sum(ls) - 1)
            ls += [r]
        ls += [ K - sum(ls) ]
        ar = np.array(ls).reshape(sp.shape)
        return State(ar, sp)
    return out_fun


#
# Fair dice
#
dice = uniform_state(pips_sp)


def flip(r, sp=None):
    """ coin with bias r, from the unit interval [0,1], by default on the
    boolean space with elements True, False """
    if isinstance(sp, Space):
        return State([r, 1 - r], sp)
    else:
        return State([r, 1 - r], bool_sp)


def coin(r):
    """ coin with bias r """
    return flip(r, coin_sp)


def cup(sp):
    """ cup as multiset on sp @ sp, which is 1 on diagonal """
    size = sp.size()
    array = np.zeros((size, size))
    np.einsum('ii->i', array)[:] = 1
    return State(array, sp @ sp)


def cup_state(sp):
    """ cup, but now normalized to a probability distribution (state) """
    return 1 / sp.size() * cup(sp)


def tvdist(s, t):
    """ Total variation distance between discrete states """
    if s.dom != t.dom:
        raise Exception('Distance requires equal spaces')
    return 0.5 * sum(abs(np.ndarray.flatten(s.array)
                         - np.ndarray.flatten(t.array)))


def kldiv(s, t, log="2"):
    """ Kullback-Leibler divergence between discrete states s,t, with
    assumption that supp(s) is a subset of supp(t). It not a runtime
    error results, via division by zero """
    def logar(x):
        if x == 0:
            return 0
        if log == "2":
            return math.log2(x)  
        return math.log(x)
    if s.dom != t.dom:
        raise Exception('Divergens requires equal spaces')
    return s >= Predicate.fromfun(lambda x: 0 if t(x) == 0 else logar(s(x)/t(x)), s.sp)


def kldiv2(s, t):
    """ Kullback-Leibler divergence with base 2 """
    return kldiv(s, t, log="2")


def kldive(s, t):
    """ Kullback-Leibler divergence with base e """
    return kldiv(s, t, log="e")


#########################################
#
# Numbers defined for multiset
#
#########################################


def is_natural_multiset(mult):
    """ returns True if mult is a multiset with natural numbers only
    as multiplicities """
    ar = mult.array
    floor_ar = np.vectorize(math.floor)(ar)
    return np.array_equal(ar, floor_ar)


def facto(mult):
    """ Product of factorials of multiplicities in the multiset """
    if not is_natural_multiset(mult):
        raise Exception("Coefficient only works for natural multisets")
    result = 1
    for a in mult.sp.iter_all():
        result *= math.factorial(mult(*a))
    return result


def coefficient(multiset):
    """ multinomial coefficient of the multiset """
    S = multiset.size_as_nat()
    return math.factorial(S) / facto(multiset)


def multiset_binom(m1, m2):
    """ binomial for multisets, which is defined if m2 <= m1 """
    if not is_natural_multiset(m1) or not is_natural_multiset(m2):
        raise Exception("Multiset binomial only works for natural multisets")
    if m1.sp != m2.sp:
        raise Exception("Multiset binomial only works for multisets on the same space")
    result = 1
    for a in m1.sp.iter_all():
        result *= binom(m1(*a), m2(*a))
    return result


def bibinom(n, K):
    """ Double bracket binomial (( - )), aka. multichoose """
    return binom(n + K - 1, K)


def multiset_bibinom(m1, m2):
    """ bibinomial for multisets, which is defined if 
    supp(m1) subset of supp(m2) """
    if not is_natural_multiset(m1) or not is_natural_multiset(m2):
        raise Exception("Multiset bibinomial only works for natural multisets")
    s1 = m1.size_as_nat()
    s2 = m2.size_as_nat()
    nominator = bibinom(s1, s2)
    prod = 1
    for indices in np.ndindex(*m1.sp.shape):
        a1 = m1.array[indices]
        if a1 > 0:
            a2 = m2.array[indices]
            prod *= bibinom(a1, a2)
    return prod / nominator 


def belows_num(mult):
    """ number of multisets strictly below mult """
    result = 0
    for a in mult.sp.iter_all():
        m = mult(*a)
        result = (m + 1) * result + m
    return math.floor(result)




# def multiset_mulnom(ms):
#     """ multinomial for multisets, which is defined if m2 <= m1 """
#     if len(ms) == 0:
#         raise Exception("Multiset multinomial requires at least one input")
#     m = ms[0]
#     c = coefficient(m)
#     for x in ms[1:]:
#         m += x
#         c *= coefficient(x)
#     return coefficient(m) / c


#########################################
#
# Spaces of multisets
#
#########################################


def sumsequences(N,K):
    """ 
    All sequences of length N with numbers adding up to K;
    The number of such is sequences is given by the multiset number:
    N+K-1 over K.
    """
    seqs = [ [] ]
    for i in range(N-1):
        new_seqs = []
        for s in seqs:
            for j in range(K - sum(s) + 1):
                new_seqs += [ [j] + s ]
        seqs = new_seqs
    return [ [ K - sum(s) ] + s for s in seqs ]


def multiset_space(K):
    """ Space of (natural) multisets of size K """
    def out_fun(sp):
        sumseqs = sumsequences(sp.size(), K)
        list_of_multisets = [ State(s,sp) for s in sumseqs ]
        return Space("Multisets of size " + str(K), list_of_multisets)
    return out_fun


def multiset_list_belows(K):
    """ List of all multisets of size K below given multiset mult """
    def out_fun(mult):
        if not is_natural_multiset(mult):
            raise Exception("Belows only works for natural multisets")
        candidates = multiset_space(K)(mult.sp)
        belows = []
        for m in candidates.iter_all():
            if mult.__ge__(*m):
                belows.append(*m)
        return belows
    return out_fun


def multiset_space_size_below(K):
    """ Space of (natural) multisets of size <= K """
    def out_fun(sp):
        sumseqs = []
        for i in range(K+1):
            sumseqs += sumsequences(sp.size(), i)
        list_of_multisets = [ State(s,sp) for s in sumseqs ]
        return Space("Multisets of size <= " + str(K), list_of_multisets)
    return out_fun


def multiset_space_below(mult):
    """ Space of (natural) multisets <= mult """
    if not is_natural_multiset(mult):
        raise Exception("Belows only works for natural multisets")
    S = mult.size_as_nat()
    mult_list = []
    for i in range(S):
        mult_list += multiset_list_belows(i)(mult)
    return Space("multisets <= " + str(mult), mult_list + [ mult ])



#########################################
#
# Accumulation and arrangement for multisets
#
#########################################


def accumulate_fun(K):
    """ function K:nat -> Space -> Space**K -> Mlt 
    that accumulates list ls over sp into multiset over sp """
    def sp_fun(sp):
        tuple_len = len(sp)
        def tuple_fun(t):
            state = empty_state(sp)
            for i in range(K):
                """ t is assumped to be a flat tuple, without inner brackets """
                item = t[ i*tuple_len : (i+1)*tuple_len ]
                state += point_state(item, sp)
            return state
        return tuple_fun
    return sp_fun


# def accumulate_fun(K, sp, t):
#     """ function nat -> Space -> Mlt 
#     that accumulates list ls over sp into multiset over sp """
#     tuple_len = len(sp)
#     state = empty_state(sp)
#     for i in range(K):
#         """ t is assumped to be a flat tuple, without inner brackets """
#         item = t[ i*tuple_len : (i+1)*tuple_len ]
#         state += point_state(item, sp)
#     return state


def accumulate(K):
    """ deterministic channel that accumulates list ls over 
    sp into multiset over sp """
    def out_fun(sp):
        dom_sp = sp ** K
        cod_sp = multiset_space(K)(sp)
        def chan_fun(*t):
            return point_state(accumulate_fun(K)(sp)(t), cod_sp)
        return chan_fromklmap(chan_fun, dom_sp, cod_sp)
    return out_fun


# def arrange(K):
#     """ channel that returns all tuples that accumulate to a multiset """
#     def out_fun(sp):
#         dom_sp = multiset_space(K)(sp)
#         cod_sp = sp ** K
#         def chan_fun(mult):
#             tuples = [t for t in cod_sp.iter_all() 
#                       if accumulate_fun(K)(sp)(t) == mult ]
#                       #if accumulate_fun(K, sp, t) == mult ]
#             L = len(tuples)
#             pairs = [ (1/L, point_state(t, cod_sp)) for t in tuples ]
#             return convex_sum(pairs)
#         return chan_fromklmap(chan_fun, dom_sp, cod_sp)
#     return out_fun


def arrange(K):
    """ channel that returns all tuples that accumulate to a multiset """
    def out_fun(sp):
        dom_sp = multiset_space(K)(sp)
        cod_sp = sp ** K
        def arrange_fun(mult):
            tuples = [t for t in cod_sp.iter_all() 
                      if accumulate_fun(K)(sp)(t) == mult ]
            L = len(tuples)
            pairs = [ (1/L, point_state(t, cod_sp)) for t in tuples ]
            return convex_sum(pairs)
        return chan_fromklmap(arrange_fun, dom_sp, cod_sp)
    return out_fun



#########################################
#
# Draw channels for multisets
#
#########################################


def draw(K):
    """ channel Mlt[K+1](sp) --> sp @ Mlt[K](sp) for drawing one element """
    def out_fun(sp):
        dom_sp = multiset_space(K+1)(sp)
        cod_sp = multiset_space(K)(sp)
        def chan_fun(m):
            ar = m.array
            """ pairs of the probability of the draw with the point state of 
            the drawn element and the remaining multiset """
            pairs = []
            for indices in np.ndindex(*sp.shape):
                if ar[indices] > 0:
                    a = np.copy(ar)
                    a[indices] -= 1
                    e = sp.get(*indices)
                    pairs += [ (ar[indices]/(K+1), 
                                point_state((*e, State(a, sp)), sp @ cod_sp)) ]
            return convex_sum(pairs)
        return chan_fromklmap(chan_fun, dom_sp, sp @ cod_sp)
    return out_fun


def drawdelete(K):
    """ channel Mlt[K+1](sp) --> Mlt[K](sp) for drawing one element """
    def out_fun(sp):
        dom_sp = multiset_space(K+1)(sp)
        cod_sp = multiset_space(K)(sp)
        def chan_fun(m):
            ar = m.array
            """ pairs of the probability of the draw with the point state of 
            the remaining multiset """
            pairs = []
            for indices in np.ndindex(*sp.shape):
                if ar[indices] > 0:
                    a = np.copy(ar)
                    a[indices] -= 1
                    pairs += [ (ar[indices]/(K+1), 
                                point_state(State(a, sp), cod_sp)) ]
            return convex_sum(pairs)
        return chan_fromklmap(chan_fun, dom_sp, cod_sp)
    return out_fun

def drawadd(K):
    """ channel Mlt[K](sp) --> Mlt[K+1](sp) for adding one element """
    def out_fun(sp):
        dom_sp = multiset_space(K)(sp)
        cod_sp = multiset_space(K+1)(sp)
        def chan_fun(m):
            ar = m.array
            """ pairs of the probability of the draw with the point state of 
            the remaining multiset """
            pairs = []
            for indices in np.ndindex(*sp.shape):
                if ar[indices] > 0:
                    a = np.copy(ar)
                    a[indices] += 1
                    pairs += [ (ar[indices]/K, 
                                point_state(State(a, sp), cod_sp)) ]
            return convex_sum(pairs)
        return chan_fromklmap(chan_fun, dom_sp, cod_sp)
    return out_fun


#########################################
#
# Special channels for multisets
#
#########################################


def flrn_chan(K):
    """ frequentis learning, as channel Mlt[K](sp) -> Dst(sp) """
    def out_fun(sp):
        dom_sp = multiset_space(K)(sp)
        return chan_fromklmap(lambda m: m.flrn(), dom_sp, sp)
    return out_fun


def zip_chan(K):
    """ zip two lists to a combined list, as channel """
    def out_fun(sp1,sp2):
        dom_sp = (sp1 ** K) @ (sp2 ** K)
        cod_sp = (sp1 @ sp2) ** K
        def chan_fun(*t):
            states = [ point_state(pair, sp1 @ sp2) 
                       for pair in list(zip(t[0:K], t[K:2*K])) ]
            tensor = functools.reduce(lambda s1, s2: s1 @ s2, states)
            return tensor
        return chan_fromklmap(chan_fun, dom_sp, cod_sp)
    return out_fun


def mzip(K):
    """ multiset zip channel Mlt[K](sp1) @ Mlt[K](sp2) -> Mlt[K](sp1 @ sp2) """
    def outfun(sp1, sp2):
        return accumulate(K)(sp1 @ sp2) * zip_chan(K)(sp1, sp2) * \
            (arrange(K)(sp1) @ arrange(K)(sp2))
    return outfun


def multiset_sum_chan(K1, K2):
    """ deterministic channel Mlt[K1](sp) x Mlt[K2](sp) --> Mlt[K1+K2](sp) """
    def aux_fun(sp):
        dom = multiset_space(K1)(sp) @ multiset_space(K2)(sp)
        cod = multiset_space(K1 + K2)(sp)
        return chan_fromklmap(lambda m1,m2: point_state(m1 + m2, cod), dom, cod)
    return aux_fun


def multiset_tensor_chan(K1, K2):
    """ deterministic channel Mlt[K1](sp) x Mlt[K2](sp) --> Mlt[K1xK2](sp) """
    def aux_fun(sp1, sp2):
        dom = multiset_space(K1)(sp1) @ multiset_space(K2)(sp2)
        cod = multiset_space(K1 * K2)(sp1 @ sp2)
        return chan_fromklmap(lambda m1,m2: point_state(m1 @ m2, cod), dom, cod)
    return aux_fun



#########################################
#
# Urn distributions: binomial, multinomial,
# hypergeometric, polya
#
#########################################


def binomial(K):
    """ Binomial distribution on {0,1,2,...,N} with probability r in [0,1] """
    return lambda r: state_fromfun(lambda k: stats.binom.pmf(k, K, r), 
                                   range_sp(K + 1))


def multinomial(K):
    """ Multinomial distribution on a state, as function sending a number K
    to the distribution of multisets of size K
    """
    def outfun(state):
        cod_sp = multiset_space(K)(state.sp)
        probs = [ coefficient(*m) * Eval_point(state, *m) 
                  for m in cod_sp.iter_all() ]
        return State(probs, cod_sp)
    return outfun


def hypgeom(L,K):
    """Hyper geometric distribution on multisets of size K """
    if L < K:
        raise Exception("Hypergeometric requires bigger urn")
    denominator = binom(L,K)
    def out_fun(space): 
        dom = multiset_space(L)(space)
        cod = multiset_space(K)(space)
        def chan_fun(urn):
            if not is_natural_multiset(urn):
                raise Exception("Hypergeometric only works for natural multisets")
            pairs = [ (multiset_binom(urn, *m) / denominator,
                       point_state(m, cod)) 
                      for m in cod.iter_all() if urn.__ge__(*m) ]
            return convex_sum(pairs)
        return chan_fromklmap(chan_fun, dom, cod)
    return out_fun


def polya(L,K):
    """ polya channel Mlt[L](space) --> Mlt[L+K](space) """
    def out_fun(space):
        dom = multiset_space(L)(space)
        cod = multiset_space(K)(space)
        def chan_fun(urn):
            if not is_natural_multiset(urn):
                raise Exception("Hypergeometric only works for natural multisets")
            pairs = [ (multiset_bibinom(urn, *draw),
                       point_state(draw, cod))         
                      for draw in cod.iter_all() if urn.supports(*draw) ]
            return convex_sum(pairs)
        return chan_fromklmap(chan_fun, dom, cod)
    return out_fun



def multiset_distribution(K):
    """ Distribution over multisets of size K, same as
        multinomial(K)(uniform_state(sp)) """
    def out_fun(sp):
        N = sp.size() ** K
        dom = multiset_space(K)(sp)
        return state_fromfun(lambda m: coefficient(m) / N, dom)
    return out_fun



#########################################
#
# Negative urn distributions
#
#########################################


def negmultinomial(state, upper=20):
    sp = state.sp
    def out_fun(draw):
        if not is_natural_multiset(draw):
            raise Exception("Negative multinomial only works for natural multisets")
        K = draw.size_as_nat()
        if K == 0:
            raise Exception("Draw has wrong size in negative multinomial")
        ran = range(K, upper+1)
        ar = np.zeros(upper + 1 - K)
        cod_sp = Space(None, list(ran))
        for i in ran:
            wi = multinomial(i-1)(state)
            for d in wi.sp.iter_all():
                if wi(*d) > 0:
                    for a in sp.iter_all():
                        pa = point_state(a, draw.sp)
                        draw_a = draw - pa
                        if draw_a.__le__(*d) and not draw.__le__(*d):
                            ar[i-K] += multinomial(i-1)(state)(*d) * state(*a)
        return State(ar, cod_sp)
    return out_fun


def neghypgeom(urn, draw):
    if not is_natural_multiset(urn) or not is_natural_multiset(draw):
        raise Exception("Negative hypergeometric only works for natural multisets")
    L = urn.size_as_nat()
    K = draw.size_as_nat()
    if (not draw <= urn) or K == 0:
        raise Exception("Draw has wrong size in negative hypergeometric")
    cod = Space(None, list(range(K,L+1)))
    ar = np.zeros(L+1-K)
    for i in range(K,L+1):
        wi = hypgeom(L,i-1)(draw.sp)(urn)
        for d in wi.sp.iter_all():
            if wi(*d) > 0:
                for a in draw.sp.iter_all():
                    if draw(*a) > 0:
                        draw_a = draw - point_state(a, draw.sp)
                        if draw_a.__le__(*d) and not draw.__le__(*d):
                            ar[i-K] += wi(*d) * urn.__sub__(*d).flrn()(*a)
                            #print(i, draw_a, d, wi(*d), urn.__sub__(*d).flrn()(*a) )
    return State(ar, cod)


def negpolya(urn, draw, upper=20):
    if not is_natural_multiset(urn) or not is_natural_multiset(draw):
        raise Exception("Negative Polya only works for natural multisets")
    L = urn.size_as_nat()
    K = draw.size_as_nat()
    ran = range(K, upper+1)
    ar = np.zeros(upper + 1 - K)
    cod_sp = Space(None, list(ran))
    for i in range(K,upper+1):
        wi = polya(L,i-1)(draw.sp)(urn)
        for d in wi.sp.iter_all():
            if wi(*d) > 0:
                for a in draw.sp.iter_all():
                    if draw(*a) > 0:
                        #print("draw:", d, a )
                        draw_a = draw - point_state(a, draw.sp)
                        if draw_a.__le__(*d) and not draw.__le__(*d):
                            ar[i-K] += wi(*d) * urn.__add__(*d).flrn()(*a)
                            #print(i, draw_a, d, wi(*d), urn.__sub__(*d).flrn()(*a) )
    return State(ar, cod_sp)





def pml(mult_of_dist):
    if not is_natural_multiset(mult_of_dist):
        raise Exception("pml only works for natural multisets")
    K = mult_of_dist.size_as_nat()
    mult_sp = mult_of_dist.sp
    sp = mult_sp[0][0].sp
    def power(x, n): return x ** math.floor(n)
    tensor_states = [ power(*mult_sp.get(*indices), 
                            math.floor(mult_of_dist.array[indices]))
                      for indices in np.ndindex(*mult_sp.shape)
                      if mult_of_dist.array[indices] != 0 ]
    tensor = functools.reduce(lambda x, y: x @ y, tensor_states)
    cod_sp = multiset_space(K)(sp)
    pairs = [ (tensor(*t), point_state(accumulate_fun(K)(sp)(t), cod_sp)) 
              for t in tensor.sp.iter_all() ]
    return convex_sum(pairs)


def pml_chan(K):
    def out_fun(dst_sp, mlt_sp):
        dom_sp = multiset_space(K)(dst_sp)
        cod_sp = multiset_space(K)(mlt_sp)
        return chan_fromklmap(pml, dom_sp, cod_sp)
    return out_fun




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
tt = Predicate([1, 0], bnd)
ff = ~tt


def bn_prior(r):
    """ Function for modelling an initial node, as prior state """
    return State([r, 1 - r], bnd)


def bn_pred(r, s):
    """ Function for a predicate on a state, in a Bayesian network """
    return Predicate([r, s], bnd)


bn_pos_pred = bn_pred(1, 0)
bn_neg_pred = bn_pred(0, 1)


def cpt(*ls):
    """ Conditional probability table converted into a channel. The input is
    a list of probabilities, of length 2^n, where n is the number of
    predecessor nodes.
    """
    n = len(ls)
    if n == 0:
        raise Exception(
            'Conditional probability table must have non-empty list of probabilities')
    log = math.log(n, 2)
    if log != math.floor(log):
        raise Exception('Conditional probability table must have 2^n elements')
    log = int(log)
    sp = functools.reduce(lambda s1, s2: s1 @ s2, [bnd] * log)
    states = [flip(r, bnd) for r in ls]
    return chan_fromstates(states, sp)



##############################################################
#
# Discretised spaces and distributions
#
##############################################################


def discretized_space(low_bound, up_bound, steps):
    """ Discretization of the real interval [low_bound, up_bound] into
    a sample space with steps many points """
    step_size = (up_bound - low_bound) / steps
    points = []
    for i in range(steps):
        #p = low_bound + i * step_size
        p = low_bound + (2 * i + 1) / 2 * step_size
        points = points + [p]
    return Space(None, points)


def discretized_state(fun, low_bound, up_bound, steps):
    """ A state with steps many points on the real line as sample space,
    between a lower and an upper bound. On these points the normalized
    values of a (probability density) function fun are the multiplicities.
    """
    dsp = discretized_space(low_bound, up_bound, steps)
    vals = [fun(p) for p in dsp[0].list]
    tot = sum(vals)
    return State([v / tot for v in vals], dsp)


def discretized_uniform(low_bound, up_bound, steps):
    """ Uniform distribution on discretized interval [low_bound, up_bound] """
    return discretized_state(lambda x: 1, low_bound, up_bound, steps)


def discretized_beta(alpha, beta, steps):
    """ Beta distribution on discretized interval [0,1] """
    return discretized_state(lambda x: stats.beta.pdf(x, a=alpha, b=beta),
                             0, 1, steps)


def discretized_exponential(lamb, up_bound, steps):
    """ Exponential distribution on discretized interval [0, up_bound] """
    return discretized_state(lambda x: stats.expon.pdf(x, scale=1 / lamb),
                             0, up_bound, steps)



##############################################################
#
# External and intern valdity and learning, in various forms, 
# for predicates and points
#
##############################################################


def Eval(state, data):
    """ External validity of data, where data is a multiset of
        predicates. The formula is:

        Product over p in data, of (self >= p)^(multiplicity of p)
        """
    vals = [state.expectation(*a) ** data(*a) for a in data.sp.iter_all()]
    val = _prod(vals)
    return val


def log_Eval(self, data):
    """ External validity of data, where data is a multiset of
        predicates. The formula is:

        Product over p in data, of (self >= p)^(multiplicity of p)
        """
    vals = [data(*a) * math.log(self.expectation(*a)) 
            for a in data.sp.iter_all()]
    return sum(vals) 


def Eval_point(state, data):
    """ External validity, where data is a multiset of
        points, to be used as point predicates"""
    vals = [state(*a) ** data(*a) for a in data.sp.iter_all()]
    val = _prod(vals)
    return val


def log_Eval_point(state, data):
    """ natural logarithm of external validity, where data is a 
        multiset of points, to be used as point predicates"""
    vals = [data(*a) * math.log(state(*a)) for a in data.sp.iter_all()]
    return sum(vals) 


def Eval_chan(state, chan, data):
    """External validity, along a channel, where data is a
        pulled back along the channel """
    # for each element, state.sp.get(*a), of the codomain
    vals = [(chan >> state).expectation(*a) ** data(*a)
            for a in data.sp.iter_all()]
    val = functools.reduce(lambda p1, p2: p1 * p2, vals, 1)
    return val


def Eval_point_chan(state, chan, data):
    """External validity, along a channel, where point-data is a
        pulled back along the channel """
    # for each element, state.sp.get(*a), of the codomain
    vals = [(chan >> state)(*a) ** data(*a) for a in data.sp.iter_all()]
    val = functools.reduce(lambda p1, p2: p1 * p2, vals, 1)
    return val


def Elrn(state, data):
    """ External learning with data, as mulitset of predicates """
    def cond(a, b): return a / b
    freqs = [data(*data.sp.get(*a))
             for a in np.ndindex(*data.sp.shape)]
    freq = sum(freqs)
    stats = [(data(*a) / freq) * cond(state, *a)
             for a in data.sp.iter_all()]
    stat = functools.reduce(lambda s1, s2: s1 + s2, stats)
    return stat


def Elrn_chan(state, chan, data):
    """ External learning along a channel with data,
        as mulitset of predicates """
    def lshift(a, b): return a << b
    freqs = [data(*data.sp.get(*a))
             for a in np.ndindex(*data.sp.shape)]
    freq = sum(freqs)
    stats = [(data(*a) / freq) * (state / lshift(chan, *a))
             for a in data.sp.iter_all()]
    stat = functools.reduce(lambda s1, s2: s1 + s2, stats)
    return stat


def Elrn_point_chan(state, chan, data):
    """ External conditioning with data, as mulitset of predicates """
    freqs = [data(*a) for a in data.sp.iter_all()]
    freq = sum(freqs)
    stats = [(data(*a) / freq) * (state / (chan << point_pred(a, chan.cod)))
             for a in data.sp.iter_all()]
    stat = functools.reduce(lambda s1, s2: s1 + s2, stats)
    return stat


def Ival(state, data):
    """ Internal validity of data, where data is a multiset of predicates"""
    def expon(a, b): return a ** b
    preds = [expon(*a, data(*a)) for a in data.sp.iter_all()]
    pred = functools.reduce(lambda p1, p2: p1 & p2,
                            preds,
                            truth(state.sp))
    return state >= pred


def log_Ival(state, data):
    """ natural logarithm of internal validity of data, where data 
        is a (natural) multiset of predicates"""
    def cond(a, b): return a / b
    val = 0
    for a in data.sp.iter_all():
        for i in range(math.floor(data(*a))):
            val += math.log(state.expectation(*a))
            state = cond(state, *a)
    return val


def Ival_point(state, data):
    """ internal validity of data, where data is a multiset of
        points, to be used as point predicates"""
    preds = [point_pred(a, state.sp) ** data(*a) for a in data.sp.iter_all()]
    pred = functools.reduce(lambda p1, p2: p1 & p2,
                            preds,
                            truth(state.sp))
    return state >= pred


def log_Ival_point(state, data):
    """ natural logarithm of internal validity of data, where data 
        is a (natural) multiset of points, to be used as point predicates"""
    def cond(a, b): return a / b
    val = 0
    for a in data.sp.iter_all():
        pred = point_pred(a, state.sp)
        for i in range(math.floor(data(*a))):
            val += math.log(state >= pred)
            state = state / pred
    return val


def Ival_chan(state, chan, data):
    """ Internal validity of data, where data is a multiset of
        predicates"""
    def lshift(a, b): return a << b
    preds = [lshift(chan, *a) ** data(*a) for a in data.sp.iter_all()]
    pred = functools.reduce(lambda p1, p2: p1 & p2,
                            preds,
                            truth(state.sp))
    return state >= pred


def log_Ival_chan(state, chan, data):
    """ natural logarithm of internal validity of data, where 
        data is a multiset ofpredicates"""
    def lshift(a, b): return a << b
    val = 0
    for a in data.sp.iter_all():
        pred = lshift(chan, *a)
        for i in range(math.floor(data(*a))):
            val += math.log(state >= pred)
            state = state / pred
    return val


def Ival_point_chan(state, chan, data):
    """ Internal validity of data, where data is a multiset of
        predicates"""
    preds = [(chan << point_pred(a, chan.cod)) ** data(*a)
             for a in data.sp.iter_all()]
    pred = functools.reduce(lambda p1, p2: p1 & p2,
                            preds,
                            truth(state.sp))
    return state >= pred


def log_Ival_point_chan(state, chan, data):
    """ natural logarithm of internal validity of data, where data
        is a (natural) multiset of points, to be used as point predicates"""
    val = 0
    for a in data.sp.iter_all():
        pred = chan << point_pred(a, chan.cod)
        for i in range(math.floor(data(*a))):
            val += math.log(state >= pred)
            state = state / pred
    return val


def Ilrn(state, data):
    """ Internal learning with data, as mulitset of predicates.
    The implementation is highly inefficient but avoids too small numbers """
    def cond(s, p): return s / p
    for a in data.sp.iter_all():
        #print( state )
        for i in range(math.floor(data(*a))):
            state = cond(state, *a)
    return state 


def Ilrn_chan(state, chan, data):
    """ Internal learning along a channel with data,
        as mulitset of predicates """
    def lshift(a, b): return a << b
    preds = [lshift(chan, *a) ** data(*a) for a in data.sp.iter_all()]
    pred = functools.reduce(lambda p1, p2: p1 & p2,
                            preds,
                            truth(state.sp))
    return state / pred


def Ilrn_point_chan(state, chan, data):
    """ Internal conditioning along a channel with data,
        as mulitset of points predicates """
    def lshift(a, b): return a << b
    # iterated version
    for a in data.sp.iter_all():
        pred = chan << point_pred(a, chan.cod)
        for i in range(math.floor(data(*a))):
            state = state / pred
    return state


##############################################################
#
# Varia
#
##############################################################



def poisson(lam, ub):
    """Poisson distribution with rate parameter `lam' and upperbound ub.
    The distribution is restricted to the finite interval [0, ub-1];
    hence the values have to adjusted so that they sum up to 1 on this
    interval.
    """
    probabilities = [(lam ** k) * (math.e ** -lam) / math.factorial(k)
                     for k in range(ub)]
    s = sum(probabilities)
    return State([p / s for p in probabilities], range_sp(ub))


def sigmoid(x):
    """ sigmoid function R -> [0,1]; works also for arrays """
    return 1 / (1 + np.exp(-x))


def softmax(vec):
    """ softmax function R^n -> D(n), protected against overflow """
    s = vec - np.max(vec)
    exps = np.exp(s)
    t = np.sum(exps)
    return State(exps / t, range_sp(len(vec)))


def sip(vec):
    """ softmax inner product state on 2^N, for vector of length N """
    N = len(vec)
    sp = range_sp(2) ** N
    Z = sum([ math.exp(np.inner(vec, s)) for s in sp.iter_all() ])
    probs = [ math.exp(np.inner(vec, s)) / Z for s in sp.iter_all() ]
    return State(probs , sp)

