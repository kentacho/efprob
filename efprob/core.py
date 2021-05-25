"""
EfProb:
Channel-based library for discrete probability calculations
"""

import functools
import numpy as np
import warnings
import math
import matplotlib.pyplot as plt
from .helpers import (_prod, mask_sum, mask_restrict)

# Number of decimals in printing
float_format_spec = ".3g"

def convex_sum(prob_item_list):
    """ Yields convex sum

    r1 * a1 + ... + rn * an

    for input list of the form

    [ (r1, an), ..., (rn, an) ]

    This function be used both for states and channels
    """
    return functools.reduce(lambda x,y: x+y, (r * s for r, s in prob_item_list))


class NormalizationError(Exception):
    """Raised when normalization fails"""


class SpaceAtom:
    """The class Space (see below) represents multi-dimensional sample
    spaces, where each product component is given by a SpaceAtom, as
    defined here. Such an atom consist of a name (label) and a list of
    elements

    """

    def __init__(self, label, list_):
        self.label = label
        if not isinstance(list_, list):
            raise ValueError(str(list_) + " is not a list")
        self.list = list_

    def __str__(self):
        return "{}: {}".format(self.label, self.list)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.list == other.list

    def __getitem__(self, key):
        return self.list[key]

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)


class Space:
    """Class of sample spaces, given by a list of SpaceAtom's The shape of
    the space is the list of numbers of elements in each product
    component. The main operation on spaces is parallel composition,
    commonly written as @. It corresponds to a concatenation of
    shapes.

    """

    def __init__(self, *list_sa):
        if list_sa and not isinstance(list_sa[0], (tuple, SpaceAtom)):
            if len(list_sa) == 2:
                list_sa = [(list_sa[0], list_sa[1])]
        self._sp = [sa if isinstance(sa, SpaceAtom)
                    else SpaceAtom(sa[0], sa[1])
                    for sa in list_sa]
        self.shape = tuple(len(s.list) for s in self._sp)

    def __str__(self):
        if not self._sp:
            return "()"
        return " @ ".join("(" + str(atom) + ")" for atom in self._sp)

    def name(self):
        if not self._sp:
            return "()"
        return " @ ".join(" " + atom.label + " " for atom in self._sp)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self._sp == other._sp

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, key):
        return self._sp[key]

    def __iter__(self):
        """ Iterator for the list of atoms """
        return iter(self._sp)

    def iter_all(self):
        """ Iterator for all elements of the space """
        return (self.get(*a) for a in np.ndindex(*self.shape))

    def __len__(self):
        return len(self._sp)

    def size(self):
        """ Total number of elements """
        return _prod(self.shape)

    def __add__(self, other):
        """ Parallel product, same as matmul """
        return self @ other

    def __matmul__(self, other):
        """ Parallel product, same as add """
        return Space(*(self._sp + other._sp))

    def __pow__(self, n):
        """ iterated parallel product """
        if n == 0:
            raise ValueError("Power must be at least 1")
        return functools.reduce(lambda s1, s2: s1 @ s2, [self] * n)

    def copy(self):
        return Space(*self._sp)

    def append(self, elm):
        """ Extend the current space with an atom space at the end """
        if not isinstance(elm, SpaceAtom):
            raise ValueError("The appended element must be"
                             "an instance of SpaceAtom")
        self._sp.append(elm)
        self.shape += (len(elm.list),)

    def get(self, *index):
        """ index is a tuple (i_1, ... i_n) of indices for getting the
        corresponding elements from the space, when it is an n-ary
        product, i.e. when it has length n """
        return tuple(self._sp[n].list[i] for n, i in enumerate(index))

    def get_index(self, *args):
        # print( args )
        # for n, a in enumerate(args):
        #     print(n, a, self._sp[n].list.index(a) )
        return tuple(self._sp[n].list.index(a) for n, a
                     in enumerate(args))

    def marginal_mask(self, *mask):
        """Mask is a list of 0 and 1 whose length is the same as the dimension
        of the space. Marginal keeps those components of the sample
        space where there is 1 in the mask.

        """
        return Space(*[sa for (sa, m)
                       in zip(self._sp, mask) if m])

    MM = marginal_mask

    def marginal_label(self, *labels):
        """This method assumes that all atoms in the space have names. It
        keeps those components that have their name in the list.

        """
        sp = []
        for l in labels:
            temp = [sa for sa in self._sp if sa.label == l]
            if len(temp) != 1:
                raise ValueError(
                    "Label {} occurs multiple times or not at all".format(l))
            sp.append(temp[0])
        return Space(*sp)

    ML = marginal_label


def range_sp(n):
    """ Sample space of n elements {0,1,...,n-1} """
    return Space(str(n), list(range(n)))




class Channel:
    """Probabilistic function from sample space dom to sample space cod.
    For each element of the domain, it gives a state on the space
    cod. The channel involves an array, representing a (probabilistic)
    matrix, basically of dimension cod x dom. The shape of the channel
    is the shape of this array.

    """

    def __init__(self, array, dom, cod):
        if isinstance(dom, Space):
            self.dom = dom
        else:
            self.dom = Space(dom)
        if isinstance(cod, Space):
            self.cod = cod
        else:
            self.cod = Space(cod)

        shape = self.cod.shape + self.dom.shape
        #self.array = np.asarray(array, dtype=float).reshape(shape)
        self.array = np.asarray(array).reshape(shape)
        self.dom_size = self.dom.size()
        self.cod_size = self.cod.size()

    @classmethod
    def fromfun(cls, fun, dom, cod):
        """Creates a channel from a function dom x cod -> reals that
        represents the matrix / array.

        """
        array = np.empty(cod.shape + dom.shape)
        #array = np.empty(cod.shape + dom.shape, dtype=float)
        for domi in np.ndindex(*dom.shape):
            for codi in np.ndindex(*cod.shape):
                array[codi + domi] = fun(*dom.get(*domi))(*cod.get(*codi))
        return Channel(array, dom, cod)

    @classmethod
    def fromklmap(cls, fun, dom, cod):
        """Creates a channel from a function that maps each element of dom
        to a state.

        """
        array = np.empty(cod.shape + dom.shape)
        #array = np.empty(cod.shape + dom.shape, dtype=float)
        for domi in np.ndindex(*dom.shape):
            array[(...,) + domi] = fun(*dom.get(*domi)).array
        return Channel(array, dom, cod)

    @classmethod
    def fromstates(cls, states, dom=None):
        """Creates a channel from list of states, all on the same sample
        space, where domain of the channel corresponds either to the
        length of the list of states, or to the optional argument dom;
        in the latter case the length of the list of states must equal
        the size of dom.

        """

        if not states:
            raise ValueError("List of states is empty")
        if not dom:
            dom = range_sp(len(states))
        if len(states) != _prod(dom.shape):
            raise ValueError(
                "Number of states does not match the given domain")
        cod = states[0].sp
        shape = cod.shape + dom.shape
        array = np.empty(shape)
        #array = np.empty(shape, dtype=float)
        for index, s in zip(np.ndindex(*dom.shape), states):
            array[(...,) + index] = s.array
        return Channel(array, dom, cod)

    def __repr__(self):
        """ Prints a channel as mapping dom -> cod, without the array """
        return "Channel of type: {} --> {}".format(self.dom, self.cod)

    def __eq__(self, other):
        """ Equality checks that domains and codomains match and that
        array values are 'close' """
        return isinstance(self, type(other)) and \
            self.dom == other.dom and \
            self.cod == other.cod and \
            np.all(np.isclose(self.array, other.array))

    def parcomp(self, other):
        """Parallel composition / product of channels, corresponding to
        Kronecker product of the matrices. This product can be written
        as @.

        """
        array = np.kron(self.array.reshape(self.cod_size,
                                           self.dom_size),
                        other.array.reshape(other.cod_size,
                                            other.dom_size))
        return Channel(array,
                       self.dom @ other.dom,
                       self.cod @ other.cod)

    def __matmul__(self, other):
        """ parallel composition / product of channels, written as: @ """
        return self.parcomp(other)

    def __pow__(self, n):
        """ iterated parallel product """
        if n == 0:
            raise ValueError("Power must be at least 1")
        return functools.reduce(lambda s1, s2: s1 @ s2, [self] * n)

    def seqcomp(self, other):
        """ Sequential composition: self after other"""
        if self.dom != other.cod:
            raise ValueError("Channels are not composable: {} != {}".format(
                self.dom, other.cod))
        array = np.dot(self.array.reshape(self.cod_size,
                                          self.dom_size),
                       other.array.reshape(other.cod_size,
                                           other.dom_size))
        return Channel(array, other.dom, self.cod)

    def __mul__(self, other):
        """ sequential composition of channels, written as: * """
        return self.seqcomp(other)

    def smul(self, scalar):
        """ Scalar multiplication, applied entrywise to the array of the
        channel
        """
        ar = scalar * self.array
        if isinstance(self, State) or isinstance(self, Predicate):
            return type(self)(ar, self.sp)
        else:
            return Channel(ar, self.dom, self.cod)

    def __rmul__(self, other):
        """ Shorthand for scalar multiplication """
        return self.smul(other)

    def marginal_mask(self, *mask):
        """Mask is a list of 0 and 1 whose length is the same as the dimension
        of the space. Marginal keeps those components of the codomain
        of the channel where there is 1 in the mask.

        """
        axes = tuple(n for n, s in enumerate(mask) if not s)
        array = self.array.sum(axes)
        sp = self.cod.marginal_mask(*mask)
        return Channel(array, self.dom, sp)

    MM = marginal_mask

    def normalize(self):
        """Normalize the channel pointwise, so that it becomes probabilistic """
        total = self.array.sum(tuple(range(len(self.cod))))
        total_inv = 1.0 / total
        zeros = np.isinf(total_inv)
        if zeros.any():
            warnings.warn("Normalization failed because of zero probabilities: "
                          "Zero distributions will appear",
                          RuntimeWarning)
            total_inv[zeros] = 0.0
        ar = self.array * \
            total_inv.reshape((1,) * len(self.cod) + self.dom.shape)
        if isinstance(self, State) or isinstance(self, Predicate):
            return type(self)(ar, self.sp)
        else:
            return Channel(ar, self.dom, self.cod)

    def conditional(self, pred):
        """Return a conditioning with pred, applied pointwise to the states
        on the codomain.
        """
        if self.cod != pred.sp:
            raise ValueError('Space mismatch in conditioning')
        try:
            return Channel.fromklmap(
                lambda x: self(x) / pred, self.dom, self.cod)
            # Channel(pred.array * self.array,
            #         self.dom,
            #         self.cod).normalize()
        except NormalizationError as e:
            raise NormalizationError(
                "Conditioning failed: {}".format(e)) from None

    def __truediv__(self, pred):
        """ Conditioning, written as: / """
        return self.conditional(pred)

    def disintegration_mask(self, *mask):
        """Disintegration pointwise, according to mask, whose length should
        match the dimension of the codomain. Where there is a 0 in the
        mask, the corresponding product component is added to the
        domain of the channel. In general, disintegration turns a
        joint probability into a conditional one. This is used here
        pointwise for channels.

        """
        dom = self.dom.copy()
        cod = Space()
        stay_axes = []
        move_axes = []
        for n, (x, s) in enumerate(zip(self.cod, mask)):
            if s:
                cod.append(x)
                stay_axes.append(n)
            else:
                dom.append(x)
                move_axes.append(n)
        n = len(self.cod)
        axes = (tuple(stay_axes)
                + tuple(n + a for a in range(len(self.dom)))
                + tuple(move_axes))
        array = self.array.transpose(axes)
        return Channel(array, dom, cod).normalize()

    DM = disintegration_mask

    def dagger(self, state):
        """ Inverse of a channel, given a state on its domain """
        return Channel.fromklmap(lambda *a:
                                 state / (self << point_pred(a, self.cod)),
                                 self.cod,
                                 self.dom)

    def __getitem__(self, key):
        """Interpretes the key as a pair of masks, written as

        chan[ conclusion_mask : condition_mask ]

        and returns the pointwise conditional probability channel

        """
        conclusion_mask = key.stop
        condition_mask = key.start
        n = len(self.cod)
        if len(condition_mask) != n or len(conclusion_mask) != n:
            raise Exception('Mask mismatch in conditional probability')
        sum_mask = mask_sum(conclusion_mask, condition_mask)
        marginal = self.MM(*sum_mask)
        sub_cond_mask = mask_restrict(sum_mask, condition_mask)
        return marginal.disintegration_mask(*sub_cond_mask)

    def get_state(self, *args):
        """ Turns a channel into an actual function, so that it can be
        applied to elements of its sample space.
        """
        index = self.dom.get_index(*args)
        array = self.array[(...,) + index]
        return State(array, self.cod)

    def __call__(self, *args):
        """ Performs the actual function call for get_state """
        return self.get_state(*args)

    def as_state(self):
        """ Turns channels 1 -> cod into state on space cod """
        if not all(len(l) == 1 for l in self.dom):
            raise ValueError("Domain is not the unit")
        return State(self.array, self.cod)

    def as_pred(self):
        """ Turns channels dom -> 1 into predicate on space dom """
        if not all(len(l) == 1 for l in self.cod):
            raise ValueError("Codomain is not the unit")
        return Predicate(self.array, self.dom)

    def as_scalar(self):
        """ Turns channels 1 -> 1 into a number """
        if _prod(self.dom) != 1 or _prod(self.cod) != 1:
            raise ValueError("Not scalar type")
        return self.array.item()

    def __rshift__(self, stat):
        """ State transformation chan >> stat """
        return (self * stat).as_state()

    def __lshift__(self, pred):
        """ predicate transformation chan << pred """
        return (pred * self).as_pred()

    def __add__(self, other):
        """ Pointwise addition of channels, written as: self + other """
        if self.dom != other.dom or self.cod != other.cod:
            raise ValueError(
                'Equality of domains and of codomains is required for addition')
        ar = self.array + other.array
        if isinstance(self, State) or isinstance(self, Predicate):
            return type(self)(ar, self.sp)
        else:
            return Channel(ar, self.dom, self.cod)

    def __sub__(self, other):
        """ Pointwise subtraction, written as: self - other """
        if self.dom != other.dom or self.cod != other.cod:
            raise ValueError(
                'Equality of domains and of codomains is required for subtraction')
        ar = self.array - other.array
        if isinstance(self, State) or isinstance(self, Predicate):
            return type(self)(ar, self.sp)
        else:
            return Channel(ar, self.dom, self.cod)

    def __and__(self, other):
        """ Pointwise multiplication/conjunction, written as: self & other """
        if self.dom != other.dom or self.cod != other.cod:
            raise ValueError('Equality of domains and of '
                             'codomains is required for conjunction')
        ar = self.array * other.array
        if isinstance(self, State) or isinstance(self, Predicate):
            return type(self)(ar, self.sp)
        else:
            return Channel(ar, self.dom, self.cod)

    def __invert__(self):
        """ Pointwise orthosupplmenent, written as: ~self """
        ar = 1.0 - self.array
        if isinstance(self, State) or isinstance(self, Predicate):
            return type(self)(ar, self.sp)
        else:
            return Channel(ar, self.dom, self.cod)

    def __or__(self, other):
        """De Morgan dual of conjunction, written as: self | other """
        return ~(~self & ~other)

    def hadamard_inv(self):
        """ Hadamard inverse: elementwise multiplicative inverse """
        ar = np.vectorize(lambda x: 1 / x)(self.array)
        if isinstance(self, State):
            return State(ar, self.sp)
        if isinstance(self, Predicate):
            return Predicate(ar, self.sp)
        return Channel(ar, self.dom, self.cod)

    def sqrt(self):
        """ elementwise square root """
        ar = np.vectorize(lambda x: math.sqrt(x))(self.array)
        if isinstance(self, State):
            return State(ar, self.sp)
        if isinstance(self, Predicate):
            return Predicate(ar, self.sp)
        return Channel(ar, self.dom, self.cod)


def idn(sp):
    """ Identity channel sp -> sp on space sp; it does nothing """
    return Channel(np.eye(_prod(sp.shape)), sp, sp)


def discard(sp):
    """ Discard channel sp -> one; it deletes everything """
    array = np.ones(sp.shape)
    return Channel(array, sp, Space())


def proj(sp, mask):
    """Projection channel according to the mask; the mask can be either a
       proper mask (a list of 0 and 1) or a number i; the latter case
       is interpreted as the i-th projection mask with a 1 only at
       position i. This i must be in the range 1,...,n, where n is the
       length of the currrent space (unlike for list access, which
       starts at 0).

    """
    if len(sp) == 0:
        raise ValueError('Projection works on non-trivial space')
    if not isinstance(mask, list):
        i = mask
        mask = [0] * len(sp)
        mask[i - 1] = 1
    if len(sp) != len(mask):
        raise ValueError('Length mismatch in projection channel')
    chans = [idn(Space(sp[i])) if mask[i] else discard(Space(sp[i]))
             for i in range(len(mask))]
    return functools.reduce(lambda x, y: x @ y, chans)


class State(Channel):
    """Probabilistic distribution (or multiset) on sample space sp, where
    the frequencies are given by a (probabilistic) matrix array. It is
    not enforced that the entries add up to one, so State can also be
    used for multisets.

    """

    def __init__(self, array, sp):
        super().__init__(array, Space(), sp)
        self.sp = self.cod

    @classmethod
    def fromfun(cls, fun, dom):
        """Creates a state from a function dom -> reals that
        represents the matrix / array.

        """
        def f():
            return fun
        return super().fromfun(f, Space(), dom).as_state()

    def __repr__(self):
        """ Prints state in ket notation: r1|x1> + ... + rn|xn> """
        return " + ".join(
            "{:{spec}}|{}>".format(
                self.array[indices],
                ",".join([str(d) for d in self.sp.get(*indices)]),
                spec=float_format_spec)
            for indices in np.ndindex(*self.sp.shape) 
            # the next line ensures that only non-zero items are printed;
            # comment it out if zero items are relevant
            if self.array[indices] != 0
        ) 

    def as_pred(self):
        """ Turn a state into a predicate, without changing the matrix. """
        return Predicate(self.array, self.sp)

    def __matmul__(self, other):
        """ Parallel composition / product of states, written as: @ """
        return super().parcomp(other).as_state()

    def expectation(self, pred=None):
        """The expected value / expectation / validity of a predicate/random
        variable is computed; if no predicate argument is given, it is
        assumed that there is an inclusion mapping from the space of
        self to the real numbers.
        """
        if not pred:
            pred = Predicate.fromfun(lambda x: x, self.sp)
        return (pred >> self).as_scalar()

    def __ge__(self, pred):
        """ validity of pred in the current state, written as: self >= pred """
        if isinstance(pred, State):
            # comparison
            return pred <= self
        return self.expectation(pred)

    def __le__(self, mult):
        """ order relation, especially useful for multisets """
        return np.all(np.less_equal(self.array, mult.array))

    def supports(self, other):
        """ support of other is contained in support of self """
        out = True
        for indices in np.ndindex(self.sp.shape):
            out = out and (other.array[indices] == 0 or self.array[indices] > 0)
        return out

    def size(self):
        """ sum of multiplicities; is one for distributions """
        return np.sum(self.array)

    def size_as_nat(self):
        """ sum of multiplicities, as natural number, esp. for natural 
        multisets """
        return math.floor(self.size())

    def flrn(self):
        return super().normalize().as_state()

    def flatten(self):
        """ flattening when self is state/multiset of states/multisets """
        pairs = [ (self(*x), *x) for x in self.sp.iter_all() ]
        return convex_sum(pairs)

    def mean(self):
        """ Assuming that the elements of the domain are real numbers,
        the validity of the inclusion into the reals is computed """
        return self >= Predicate.fromfun(lambda x: x, self.sp)

    def variance(self, pred=None):
        """Variance of a predicate/random variable; if no predicate argument
        is given, it is assumed that there is an inclusion mapping
        from the space of self to the real numbers.

        """
        if not pred:
            pred = Predicate.fromfun(lambda x: x, self.sp)
        val = self.expectation(pred)
        p = pred - val * truth(pred.sp)
        return self.expectation(p & p)

    def st_deviation(self, pred=None):
        return math.sqrt(self.variance(pred))

    def covariance(self, pred1=None, pred2=None):
        """Covariance, for predicates with self as shared state.  If no
            predicates are provided, the identities are used.

        """
        if not pred1:
            pred1 = Predicate.fromfun(lambda x: x, self.sp)
        if not pred2:
            pred2 = Predicate.fromfun(lambda x: x, self.sp)
        val1 = self >= pred1
        val2 = self >= pred2
        p1 = pred1 - val1 * truth(pred1.sp)
        p2 = pred2 - val2 * truth(pred2.sp)
        return self >= (p1 & p2)

    def joint_covariance(self, pred1=None, pred2=None, mask=None):
        """joint covariance, for predicates on pred1 defined on the part
           self.MM(mask) and pred2 on the complemented part of
           self. If no mask is provided, a binary mask is used, with
           corresponding identity predicates.

        """
        sp = self.sp
        if mask is None:
            mask1 = [1, 0]
        else:
            mask1 = mask
        mask2 = [1 - i for i in mask1]
        sp1 = sp.MM(*mask1)
        sp2 = sp.MM(*mask2)
        if not pred1:
            pred1 = Predicate.fromfun(lambda x: x, sp1)
        if not pred2:
            pred2 = Predicate.fromfun(lambda x: x, sp2)
        val1 = self.MM(*mask1) >= pred1
        val2 = self.MM(*mask2) >= pred2
        p1 = pred1 - val1 * truth(sp1)
        p2 = pred2 - val2 * truth(sp2)
        return self >= ((proj(sp, mask1) << p1) & (proj(sp, mask2) << p2))

    def correlation(self, pred1=None, pred2=None):
        """ Correlation, for predicates with self as shared state """
        if not pred1:
            pred1 = Predicate.fromfun(lambda x: x, self.sp)
        if not pred2:
            pred2 = Predicate.fromfun(lambda x: x, self.sp)
        std1 = self.st_deviation(pred1)
        std2 = self.st_deviation(pred2)
        return self.covariance(pred1, pred2) / (std1 * std2)

    def joint_correlation(self, pred1=None, pred2=None, mask=None):
        """ see joint_covariance for explanation """
        if mask is None:
            mask1 = [1, 0]
        else:
            mask1 = mask
        mask2 = [1 - i for i in mask1]
        if not pred1:
            pred1 = Predicate.fromfun(lambda x: x, self.sp.MM(*mask1))
        if not pred1:
            pred1 = Predicate.fromfun(lambda x: x, self.sp.MM(*mask2))
        std1 = self.MM(*mask1).st_deviation(pred1)
        std2 = self.MM(*mask2).st_deviation(pred2)
        return self.joint_covariance(pred1, pred2, mask) / (std1 * std2)

    def MM(self, *mask):
        """Mask is a list of 0 and 1 whose length is the same as the dimension
        of the space. Marginal keeps those components of the state
        where there is 1 in the mask.

        """
        return super().MM(*mask).as_state()

    def as_chan(self):
        """ Turns state into channel 1 -> dom """
        return Channel(self.array, Space(), self.sp)

    def __truediv__(self, pred):
        """ Conditioning, written as: / """
        if self.sp != pred.sp:
            raise ValueError('Space mismatch in conditioning')
        try:
            return State(pred.array * self.array, self.sp).normalize()
        except NormalizationError as e:
            raise NormalizationError(
                "Conditioning failed: {}".format(e)) from None

    def get_value(self, *args):
        """ Allows states to be used as function dom -> reals, for elements
        from its sample space as arguments """
        return self.as_pred()(*args)

    def __call__(self, *args):
        """ Performs the function call for get_value. """
        return self.get_value(*args)

    def argmax(self):
        """ Maximum a posteriori probability (MAP): returns the list
        of pairs (r,x) where the probability r is maximal.
        """
        index = np.unravel_index(np.argmax(self.array), self.array.shape)
        maxprob = self.array[index]
        items = [self.sp[i][index[i]] for i in range(len(self.sp))]
        return (maxprob, items)

    def info_content(self):
        """ Information content """
        return Predicate.fromfun(lambda x: -math.log2(self(x)) if self(x) != 0 
                                 else 0, 
                                 self.sp)

    def entropy(self):
        """ Shannon entropy """
        return self >= self.info_content()

    def sample(self, N=1):
        """ sample N times, producing a multiset of size N of samples """
        ar = self.array
        sp = self.sp
        sam = np.random.multinomial(N, ar.reshape(sp.size()))
        return State(sam.reshape(ar.shape), sp)

    def sample_elt(self):
        """ sample a single element from the underlying space """
        sam = self.sample()
        return sam.argmax()[1][0]

    def plot(self, skiplabels=None):
        """Plots a state as bar chart, but works only for one and two
        dimensional states without too many elements. Use with care.

        """
        if len(self.sp) == 1:
            self.plot1(skiplabels)
        elif len(self.sp) == 2:
            self.plot2(skiplabels)
        else:
            raise ValueError('plots are only supported for dimensions 1 or 2')

    def plot1(self, skiplabels=None):
        """ Plots one dimensional states; is fragile. """
        xs = self.sp[0].list
        # print( xs )
        ys = [self(x) for x in xs]
        # print( ys )
        plt.rcParams["figure.figsize"] = (10, 5)
        if skiplabels:
            labels = [""] * len(xs)
            for i in range(math.floor(len(xs) / skiplabels)):
                labels[skiplabels * i] = xs[skiplabels * i]
        else:
            labels = xs
        # round float label entries to two decimals
        labels = [float("{0:.2f}".format(x)) if isinstance(x, float) else x
                  for x in labels]
        #print("Skips: ", skiplabels, labels)
        plt.subplots()
#        plt.xticks(xs, labels, rotation=45)
        #plt.bar(xs, ys, align="center", width = 1 / (0.12 * len(xs)))
        plt.bar(xs, ys, align="center", width=1 / (0.7 * len(xs)))
        #plt.bar(xs, ys, align="center", width = 1 / (0.12 * len(xs)))
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")

    def plot2(self, skiplabels=None):
        """ Plots one dimensional states; is fragile. """
        # handling of labels is unclear; links:
        # https://matplotlib.org/gallery/mplot3d/3d_bars.html
        # https://stackoverflow.com/questions/9433240/python-matplotlib-3d-bar-plot-adjusting-tick-label-position-transparent-b
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        xs = self.sp[0].list
        ys = self.sp[1].list
        lx = len(xs)
        ly = len(ys)
        zs = [self(x, y) for x in xs for y in ys]
        print(xs)
        # xpos = [1,1,1,2,3] # x coordinates of each bar
        # ypos = [1,2,3,2,3] # y coordinates of each bar
        xpos = []
        ypos = []
        for i in range(lx):
            xpos += [xs[lx - i - 1]] * ly
            ypos += ys
        print(xpos)
        print(ypos)
        num_elements = len(xpos)
        zpos = [0] * num_elements  # z coordinates of each bar
        # dx = np.ones(10)  # width of each bar
        # dy = np.ones(10)  # depth of each bar
        # dz = [1, 2, 3, 4, 5]
        # dz = [10,2,3,4,5,6,7,8,9] # height of each bar

        ticksx = np.arange(0.2, 5, 1)
        plt.xticks(ticksx, xs)

        ticksy = np.arange(0.2, 5, 1)
        plt.yticks(ticksy, ys)

        ax1.bar3d(xpos, ypos, zpos, 0.5, 0.5, zs)
        # ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
        # ax1.bar3d(xs, ys, zs, xs, ys, zs, color='#00ceaa')
        plt.draw()
        plt.pause(0.001)
        input("Press [enter] to continue.")


def uniform_state(sp):
    """ uniform state on sample space sp """
    size = sp.size()
    array = np.zeros(size)
    array[:] = 1.0 / size
    return State(array, sp)


class Predicate(Channel):
    """Collection of real values on a sample space sp, where the values
    are given by a matrix array. It is not enforced that the entries
    are between zero and one, so Predicate can also be used for
    observables.

    """

    def __init__(self, array, sp):
        super().__init__(array, sp, Space())
        self.sp = self.dom

    @classmethod
    def fromfun(cls, fun, dom):
        """Creates a Predicate from a function dom -> reals that
        represents the matrix / array.
        """
        def f(*args):
            return lambda: fun(*args)
        return super().fromfun(f, dom, Space()).as_pred()

    def __repr__(self):
        """ Prints a predicate as x1 : r1 | ... | xn : rn
        with value ri for element xi of the sample space.
        """
        return " | ".join(
            "{}: {:{spec}}".format(
                ",".join([str(d) for d in self.sp.get(*indices)]),
                self.array[indices],
                spec=float_format_spec)
            for indices in np.ndindex(*self.sp.shape)
            # the next line ensures that only non-zero items are printed;
            # comment it out if zero items are relevant
            if self.array[indices] != 0
        )
            

    def __matmul__(self, other):
        """ Parallel composition / product of predicates, written as: @ """
        return super().parcomp(other).as_pred()

    def __pow__(self, r):
        """ Iterated conjunction (not product, like in the superclass) """
        return Predicate(np.power(self.array, r), self.sp)

    def __mod__(self, state):
        """ Action of predicate self on states, written as: % """
        ar = self.array * state.array
        return State(ar, self.sp)

    def __call__(self, *args):
        """ Make it possible to use predicate as function dom -> reals. """
        return super().__call__(*args).as_scalar()

    def normalize(self):
        return super().normalize().as_pred()

    def argmax(self):
        """ Maximum a posteriori probability (MAP); same as for states """
        index = np.unravel_index(np.argmax(self.array), self.array.shape)
        maxprob = self.array[index]
        items = [self.sp[i][index[i]] for i in range(len(self.sp))]
        return (maxprob, items)


def truth(sp):
    """ Predicate that is always 1 on sample space sp """
    array = np.ones(sp.shape)
    return Predicate(array, sp)


def falsum(sp):
    """ Predicate that is always 0 on sample space sp """
    array = np.zeros(sp.shape)
    return Predicate(array, sp)


def point_pred(point, sp):
    """ Singleton / Dirac predicate for the element called point the
    sample space sp
    """
    if not isinstance(point, tuple):
        point = (point,)
    array = np.zeros(sp.shape)
    array[sp.get_index(*point)] = 1
    return Predicate(array, sp)


