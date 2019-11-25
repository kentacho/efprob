"""Test functions based on illustrations and exercises from 
the book (in progress):

Bart Jacobs, Structured Probabilistic Reasoning

Draft available from:

http://www.cs.ru.nl/B.Jacobs/PAPERS/ProbabilisticReasoning.pdf

Instruction: (install and) run "pytest -v" on this file.

One can also copy code fragements from this file for own experiments
and variations.

"""

from efprob import *
from builtins import *
from hmm import *


##############################################################
#
# Chapter 1. Collections and Channels
#
##############################################################

def test_wet_grass():
    #
    # Spaces
    #
    A = Space("A", ['a', '~a'])
    B = Space("B", ['b', '~b'])
    C = Space("C", ['c', '~c'])
    D = Space("D", ['d', '~d'])
    E = Space("E", ['e', '~e'])
    #
    # State and channels
    #
    wi = flip(0.6,A)
    sp = chan_fromstates([flip(0.2,B),flip(0.75,B)], A)
    ra = chan_fromstates([flip(0.8,C),flip(0.1,C)], A)
    wg = chan_fromstates([flip(0.95,D),flip(0.9,D),flip(0.8,D),flip(0,D)],B @ C)
    sr = chan_fromstates([flip(0.7,E),flip(0,E)], C)
    #
    # Predictions about Bayesian network
    #
    assert (sp >> wi) == State([21/50, 29/50],B) 
    assert (ra >> wi) == State([13/25, 12/25],C) 
    assert (sr * ra >> wi) == State([91/250, 159/250],E) 
    assert (wg >> ((sp @ ra) >> (copy(A) >> wi))) \
        == State([1399/2000, 601/2000],D) 
    #
    # Exercise
    #
    assert ((wg @ sr) >> ((idn(B) @ copy(C)) >> ((sp @ ra) >> (copy(A) >> wi)))) \
        == State([30443/100000, 39507/100000, 5957/100000, 24093/100000], D @ E)
    #
    # Wetness inferences (from Directed Models chapter)
    #
    joint = ((idn(A) @ idn(B) @ wg @ idn(C) @ sr) \
             * (idn(A) @ copy(B) @ copy(C,3)) \
             * (idn(A) @ sp @ ra) \
             * copy(A,3)) >> wi
    sr_pred = point_pred('e', E)
    #
    # Sprinkler probability, given a slippery road
    #
    assert (sp >> (wi / (ra << (sr << sr_pred)))) \
        == State([63/260, 197/260],B)
    assert (joint / (truth(A @ B @ D @ C) @ sr_pred)).MM(0,1,0,0,0) \
        == State([63/260, 197/260],B)
    #
    # Wet grass probability, given a slippery road
    #
    assert (wg >> (((sp @ ra) >> (copy(A) >> wi)) / (truth(B) @ (sr << sr_pred)))) \
        == State([4349/5200, 851/5200], D)
    assert (joint / (truth(A @ B @ D @ C) @ sr_pred)).MM(0,0,1,0,0) \
        == State([4349/5200, 851/5200], D)        



##############################################################
#
# Chapter 2. Predicates and Observables
#
##############################################################

def test_validity():
    #
    # Coin toss with win/loose reward and expected outcome
    #
    s = flip(0.3)
    rv = Predicate([100, -50], bool_sp)
    assert (s >= rv) == -5
    #
    # Expected sum of two/three dices
    #
    two_sum = pred_fromfun(lambda x,y: x+y, pips_sp @ pips_sp)
    assert np.isclose(dice @ dice >= two_sum, 7)
    three_sum = pred_fromfun(lambda x,y,z: x+y+z, pips_sp ** 3)
    assert np.isclose(dice @ dice @ dice >= three_sum, 21/2)


def test_evenish():
    #
    # Validity and conditioning for a dice with a fuzzy predicate
    #
    evenish = Predicate([1/5, 9/10, 1/10, 9/10, 1/10, 4/5], pips_sp)
    assert (dice >= evenish) == 1/2
    assert (dice / evenish) == State([1/15, 3/10, 1/30, 3/10, 1/30, 4/15], 
                                     pips_sp)


def test_burlargy_alarm():
    #
    # A burglarty-alarm example due to Pearl, with crossover influence
    #
    A = Space("A", ['a','~a'])
    B = Space("B", ['b','~b'])
    w = State([0.000095,0.009999,0.000005,0.989901], A @ B)
    assert w.MM(0,1) == State([0.0001, 0.9999], B)
    p = Predicate([0.8, 0.2], A)
    assert np.isclose((w >= (p @ truth(B))), 0.206056)
    assert (w / (p @ truth(B))) \
        == State([0.00036883, 
                  0.03882043, 
                  4.85e-06, 
                  0.9608], A @ B)
    assert (w / (p @ truth(B))).MM(0,1) == State([0.00037368, 
                                                  0.99962], B)


def test_two_urns_draw():
    #
    # Two urns with white and black balls, and a draw from either of them.
    #
    B = Space("Balls", ['W', 'B'])
    c = chan_fromstates([State([2/9,11/9],B), State([5/11,6/11],B)], coin_sp)
    assert (coin(0.5) / (c << point_pred('W',B))) \
        == State([22/67, 45/67], coin_sp)


def test_taxi_cab():
    C = Space("Colours", ['G', 'B'])
    w = flip(0.85, C)
    c = chan_fromstates([flip(0.8, C), flip(0.2,C)], C)
    assert (w / (c << point_pred('B', C))) == State([17/29, 12/29], C)


def test_blood_medicine():
    B = Space(None, ['H', 'L'])
    M = range_sp(3)
    b = chan_fromstates([flip(2/3,B), flip(7/9,B), flip(5/8,B)], M)
    prior = State([3/20, 9/20, 2/5], M)
    assert (b >> prior) == State([7/10, 3/10], B)
    p12 = Predicate([0,1,1], M)
    assert (b >> (prior / p12)) == State([12/17, 5/17], B)
    q1 = Predicate([1,0], B)
    print( b << q1 )
    print( prior >= b << q1 )
    assert (prior / (b << q1)) == State([1/7, 1/2, 5/14], M)
    q2 = Predicate([0.95,0.05], B)
    assert (prior / (b << q2)) == State([0.143382, 0.49632, 0.360294], M)


def test_capture_recapture():
    N = 20
    fish_sp = Space(None, [10 * i for i in range(2, 31)])
    prior = uniform_state(fish_sp)
    chan = chan_fromklmap(lambda d: binomial(N, N/d), fish_sp, range_sp(N+1))
    #(chan >> prior).plot()
    posterior = prior / (chan << point_pred(5, range_sp(N+1)))
    #
    # Expected number after catching 5 marked
    #
    assert posterior.expectation() == 116.49192983579051
    #
    # Expected number after catching 10 marked
    #
    assert (prior / (chan << point_pred(10, range_sp(N+1)))).expectation() \
        == 47.481088166925645
    #posterior.plot()


def test_disease_test():
    disease_sp = Space(None, ['d', '~d'])
    prior = flip(1/100, disease_sp)
    test_sp = Space(None, ['p', 'n'])
    test_pred = Predicate([1,0], test_sp)
    sensitivity = chan_fromstates([flip(9/10,test_sp), flip(1/20,test_sp)], 
                                  disease_sp)
    #
    # Prediction
    #
    assert (sensitivity >> prior) == State([117/2000, 1883/2000], test_sp)
    #
    # Explanation
    #
    assert (prior / (sensitivity << test_pred)) \
        == State([18/117, 99/117], disease_sp)


def test_resignations():
    #
    # Exercise
    #
    S = Space("Store", ['A', 'B', 'C'])
    G = Space("Gender", ['M', 'F'])
    w = State([50/225, 75/225, 100/225], S)
    c = chan_fromstates([State([0.5,0.5],G),
                         State([0.4,0.6],G),
                         State([0.3,0.7],G)], S)
    assert (w / (c << point_pred('F',G))) == State([5/28, 9/28, 14/28], S)


def test_coin_parameter():
    N = 100
    prior = discretized_uniform(0, 1, N)
    chan = chan_fromklmap(lambda r: flip(r), prior.sp, bool_sp)
    assert (chan >> prior) == State([99/200, 101/200], bool_sp)
    observations = [0,1,1,1,0,0,1,1]
    s = prior
    #s.plot(10)
    for ob in observations:
        pred = yes_pred if ob==1 else no_pred
        s = s / (chan << pred)
    #s.plot(10)
    # 
    # learned coin
    #
    assert (chan >> s) == State([3/5, 2/5], bool_sp)
    #
    # Expected value
    #
    assert s.expectation() == 0.6000000167832008


def test_asia_visit():
    A = Space("asia", ['a', '~a'])
    S = Space("smoking", ['s', '~s'])
    T = Space("tuberculosis", ['t', '~t'])
    E = Space("either", ['e', '~e'])
    L = Space("cancer", ['l', '~l'])
    X = Space("xray", ['x', '~x'])
    D = Space("dyspnoea", ['d', '~d'])
    B = Space("bronchitis", ['b', '~b'])
    #
    # Initial states
    #
    asia = flip(0.01, A) 
    smoking = flip(0.5, S) 
    #
    # Channels
    #
    tub = chan_fromstates([flip(0.05,T), flip(0.01,T)], A)
    either = chan_fromstates([flip(1,E), flip(1,E), flip(1,E), flip(0,E)], 
                             L @ T)
    lung = chan_fromstates([flip(0.1,L), flip(0.01,L)], S)
    xray = chan_fromstates([flip(0.98,X), flip(0.05,X)], E)
    dysp = chan_fromstates([flip(0.9,D), flip(0.7,D), flip(0.8,D), flip(0.1,D)],
                       B @ E)
    bronc = chan_fromstates([flip(0.6,B), flip(0.3,B)], S)
    #
    # Add wires from internal nodes to the outside, so that the 8 outgoing 
    # wires are respectively:
    # 1. smoking
    # 2. broncchitis
    # 3. lung 
    # 4. dyspnoea
    # 5. either
    # 6. xray
    # 7. tuberculosis
    # 8. asia
    #
    asia_joint = ((idn(S @ B @ L) @ dysp @ idn(E) @ xray @ idn(T @ A)) \
                  * (idn(S @ B) @ swap(B, L) @ copy(E,3) @ idn(T @ A)) \
                  * (idn(S @ B @ B @ L) @ either @ idn(T @ A)) \
                  * (idn(S) @ copy(B) @ copy(L) @ copy(T) @ idn(A)) \
                  * (idn(S) @ bronc @ lung @ tub @ idn(A)) \
                  * (copy(S,3) @ copy(A))) \
                  >> smoking @ asia
    #
    # Likelihood of lung cancer, given no bronchitis
    #
    B_pred = point_pred('b', B)
    assert (lung >> (smoking / (bronc << ~B_pred))) \
        == State([0.042727, 0.95727], L)
    assert ((asia_joint/(truth(S) @ ~B_pred @ truth(L @ D @ E @ X @ T @ A))). \
            MM(0,0,1,0,0,0,0,0)) == State([0.042727, 0.95727], L)
    #
    # likelihood of smoking, given positive xray, computed in many
    # different ways
    #
    X_pred = point_pred('x', X)
    assert ((smoking @ asia) / \
            ((lung @ tub) << (either << (xray << X_pred)))).MM(1,0) \
            == State([0.68775, 0.312246], S)
    assert (asia_joint / (truth(S @ B @ L @ D @ E) @ X_pred @ truth(T @ A))). \
        MM(1,0,0,0,0,0,0,0) == State([0.68775, 0.312246], S)
    assert (proj(S @ D @ X, [1,0,0]) >> \
            ((((idn(S) @ dysp @ idn(X)) * \
               (idn(S @ B) @ idn(E) @ xray) * \
               (idn(S @ B) @ copy(E)) * \
               (idn(S @ B) @ either) * \
               (idn(S @ B @ L) @ tub) * \
               (idn(S @ B @ L) @ asia.as_chan()) * \
               (idn(S @ B) @ lung) * \
               (idn(S) @ bronc @ idn(S)) * \
               copy(S,3)) >> smoking) / (proj(S @ D @ X, [0,0,1]) << X_pred))) \
               == State([0.68775, 0.312246], S)
    assert (proj(S @ B @ E @ X, [1,0,0,0]) >> \
            ((((idn(S @ B @ E) @ xray) * \
               (idn(S @ B) @ copy(E)) * \
               (idn(S @ B) @ either) * \
               (idn(S @ B @ L) @ tub) * \
               (idn(S @ B @ L) @ asia.as_chan()) * \
               (idn(S @ B) @ lung) * \
               (idn(S) @ bronc @ idn(S)) * \
               copy(S,3)) >> smoking) \
             / (proj(S @ B @ E @ X, [0,0,0,1]) << X_pred))) \
             == State([0.68775, 0.312246], S)
    assert (proj(S @ B @ E @ E, [1,0,0,0]) >> \
            ((((idn(S @ B) @ copy(E)) * \
               (idn(S @ B) @ either) * \
               (idn(S @ B @ L) @ tub) * \
               (idn(S @ B @ L) @ asia.as_chan()) * \
               (idn(S @ B) @ lung) * \
               (idn(S) @ bronc @ idn(S)) * \
               copy(S,3)) >> smoking) / (proj(S @ B @ E @ E, [0,0,0,1]) << \
                                         (xray << X_pred)))) \
                                         == State([0.68775, 0.312246], S)
    assert (proj(S @ B @ E, [1,0,0]) >> \
            ((((idn(S @ B) @ either) * \
               (idn(S @ B @ L) @ tub) * \
               (idn(S @ B @ L) @ asia.as_chan()) * \
               (idn(S @ B) @ lung) * \
               (idn(S) @ bronc @ idn(S)) * \
               copy(S,3)) >> smoking) / (proj(S @ B @ E, [0,0,1]) << \
                                         (xray << X_pred)))) \
                                         == State([0.68775, 0.312246], S)
    assert (proj(S @ B @ L @ T, [1,0,0,0]) >> \
            ((((idn(S @ B @ L) @ tub) * \
               (idn(S @ B @ L) @ asia.as_chan()) * \
               (idn(S @ B) @ lung) * \
               (idn(S) @ bronc @ idn(S)) * \
               copy(S,3)) >> smoking) / (proj(S @ B @ L @ T, [0,0,1,1]) << \
                                         (either << \
                                          (xray << X_pred))))) \
                                          == State([0.68775, 0.312246], S)
    assert (proj(S @ B @ L @ A, [1,0,0,0]) >> \
            ((((idn(S @ B @ L) @ asia.as_chan()) * \
               (idn(S @ B) @ lung) * \
               (idn(S) @ bronc @ idn(S)) * \
               copy(S,3)) >> smoking) / (proj(S @ B @ L @ A, [0,0,1,1]) << \
                                         ((idn(L) @ tub) << \
                                          (either << \
                                           (xray << X_pred)))))) \
                                           == State([0.68775, 0.312246], S)
    assert (proj(S @ B @ L, [1,0,0]) >> \
            ((((idn(S @ B) @ lung) * \
               (idn(S) @ bronc @ idn(S)) * \
               copy(S,3)) >> smoking) / (proj(S @ B @ L, [0,0,1]) << \
                                         ((idn(L) @ asia.as_chan()) << \
                                          ((idn(L) @ tub) << \
                                           (either << \
                                            (xray << X_pred))))))) \
                                            == State([0.68775, 0.312246], S)

    assert (proj(S @ B @ S, [1,0,0]) >> \
            ((((idn(S) @ bronc @ idn(S)) * \
               copy(S,3)) >> smoking) / (proj(S @ B @ S, [0,0,1]) << \
                                         (lung << \
                                          ((idn(L) @ asia.as_chan()) << \
                                           ((idn(L) @ tub) << \
                                            (either << \
                                             (xray << X_pred)))))))) \
                                             == State([0.68775, 0.312246], S)
    assert (proj(S @ S @ S, [1,0,0]) >> \
            ((copy(S,3) >> smoking) / (proj(S @ S @ S, [0,0,1]) << \
                                       (lung << \
                                        ((idn(L) @ asia.as_chan()) << \
                                         ((idn(L) @ tub) << \
                                          (either << \
                                           (xray << X_pred)))))))) \
                                           == State([0.68775, 0.312246], S)
    assert (smoking / (lung << \
                       ((idn(L) @ asia.as_chan()) << \
                        ((idn(L) @ tub) << \
                         (either << \
                          (xray << X_pred)))))) \
                          == State([0.68775, 0.312246], S)



def test_coin_dice_variance():
    s = flip(0.3)
    rv = Predicate([100, -50], s.sp)
    assert (s >= rv) == -5
    assert s.variance(rv) == 4725
    assert dice.variance() == 70/24 


def test_list_covariance():
    F = range_sp(5)
    a = Predicate([5, 10, 15, 20, 25], F)
    b = Predicate([10, 8, 10, 15, 12], F)
    assert (uniform_state(F) >= a) == 15
    assert (uniform_state(F) >= b) == 11
    assert uniform_state(F).covariance(a,b) == 11
    assert uniform_state(F).variance(a) == 50 
    # computed outcome is: 5.6000000000000005
    assert np.isclose(uniform_state(F).variance(b), 5.6)
    assert np.isclose(uniform_state(F).correlation(a,b), 0.65737)


def test_joint_variance_covariance():
    X = Space("X", [1,2])
    Y = Space("Y", [1,2,3])
    t = State([1/4, 1/4, 0, 0, 1/4, 1/4], X @ Y)
    assert t.MM(1,0).mean() == 3/2
    assert t.MM(0,1).mean() == 2
    assert t.MM(1,0).mean() == 3/2
    assert t.joint_covariance() == 1/4
    assert np.isclose(t.joint_correlation(), 1/2 * math.sqrt(2))



##############################################################
#
# Chapter 3. Directed Graphical Models
#
##############################################################

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


def test_wheather_play():
    Outlook = Space("Outlook", ['s', 'o', 'r'])
    Temp = Space("Temperature", ['h', 'm', 'c'])
    Humidity = Space("Humidity", ['h', 'n'])
    Windy = Space("Windiness", ['t', 'f'])
    Play = Space("Playing", ['y', 'n'])
    # joint domain
    S = Outlook @ Temp @ Humidity @ Windy @ Play
    # 
    # Empirical distribution
    #
    table = 1/14  *  point_state(('s', 'h', 'h', 'f', 'n'), S) \
            + 1/14 * point_state(('s', 'h', 'h', 't', 'n'), S) \
            + 1/14 * point_state(('o', 'h', 'h', 'f', 'y'), S) \
            + 1/14 * point_state(('r', 'm', 'h', 'f', 'y'), S) \
            + 1/14 * point_state(('r', 'c', 'n', 'f', 'y'), S) \
            + 1/14 * point_state(('r', 'c', 'n', 't', 'n'), S) \
            + 1/14 * point_state(('o', 'c', 'n', 't', 'y'), S) \
            + 1/14 * point_state(('s', 'm', 'h', 'f', 'n'), S) \
            + 1/14 * point_state(('s', 'c', 'n', 'f', 'y'), S) \
            + 1/14 * point_state(('r', 'm', 'n', 'f', 'y'), S) \
            + 1/14 * point_state(('s', 'm', 'n', 't', 'y'), S) \
            + 1/14 * point_state(('o', 'm', 'h', 't', 'y'), S) \
            + 1/14 * point_state(('o', 'h', 'n', 'f', 'y'), S) \
            + 1/14 * point_state(('r', 'm', 'h', 't', 'n'), S) 
    #
    # Play state via marginalisation
    #
    prior_play = table.MM(0,0,0,0,1)
    assert prior_play == State([9/14, 5/14], Play)
    #
    # Channels via disintegration
    #
    cO = table[[1,0,0,0,0] : [0,0,0,0,1]]
    cT = table[[0,1,0,0,0] : [0,0,0,0,1]]
    cH = table[[0,0,1,0,0] : [0,0,0,0,1]]
    cW = table[[0,0,0,1,0] : [0,0,0,0,1]]
    #
    # Outlook channel
    #
    assert cO('y') == State([2/9, 4/9, 3/9], Outlook)
    assert cO('n') == State([3/5, 0, 2/5], Outlook)
    #
    # Temperature channel
    #
    assert cT('y') == State([2/9, 4/9, 3/9], Temp)
    assert cT('n') == State([2/5, 2/5, 1/5], Temp)
    #
    # Humidity channel
    #
    assert cH('y') == State([1/3, 2/3], Humidity)
    assert cH('n') == State([4/5, 1/5], Humidity)
    #
    # Windy channel
    #
    assert cW('y') == State([1/3, 2/3], Windy)
    assert cW('n') == State([3/5, 2/5], Windy)
    c = tuple_chan(cO, cT, cH, cW)
    #
    # Pulled-back point predicate
    #
    assert (c << point_pred(('s', 'c', 'h', 't'), c.cod)) \
        == Predicate([2/243, 36/625], Play)
    assert np.isclose(prior_play >= c << point_pred(('s', 'c', 'h', 't'),c.cod),
                      4277 / 165375)
    assert (prior_play / (c << point_pred(('s', 'c', 'h', 't'), c.cod))) \
        == State([125/611, 486/611], Play)
    assert c.dagger(prior_play)('s', 'c', 'h', 't') \
        == State([125/611, 486/611], Play)


def test_joint_state_factorisation():
    A = Space(None, ['a', '~a'])
    B = Space(None, ['b', '~b'])
    C = Space(None, ['c', '~c'])
    D = Space(None, ['d', '~d'])
    joint = State([0.04, 0.18, 0.06, 0.02,
                   0.04, 0.18, 0.06, 0.02,
                   0.024, 0.018, 0.036, 0.002,
                   0.096, 0.072, 0.144, 0.008], A @ C @ D @ B)
    #
    # Marginal 
    #
    s = joint.MM(1,0,0,1)
    assert s == State([0.2, 0.4, 0.3, 0.1], A @ B)
    #
    # Extracted channels
    #
    f = joint[ [0,1,0,0] : [1,0,0,0] ]
    assert f('a') == State([0.5,0.5],C)
    assert f('~a') == State([0.2,0.8],C)
    g = joint[ [0,0,1,0] : [0,0,0,1] ]
    assert g('b') == State([0.4,0.6],D)
    assert g('~b') == State([0.9,0.1],D)
    #
    # The original joint state can be reconstructed, and thus has the
    # claimed shape
    #
    assert joint == (idn(A) @ f @ g @ idn(B)) >> ((copy(A) @ copy(B)) >> s)


def test_barbers_burglary_alarm():
    B = Space("Burglary", ['b', '~b'])
    E = Space("Earthquake", ['e', '~e'])
    A = Space("Alarm", ['a', '~a'])
    wB = flip(0.01, B)
    wE = flip(0.000001, E)
    alarm = chan_fromstates([flip(0.9999,A), 
                             flip(0.99,A), 
                             flip(0.99,A), 
                             flip(0.0001,A)], B @ E)
    p = Predicate([0.7,0.3],A)
    s = State([0.7,0.3],A)
    #
    # Burglary probability with Pearl's rule
    #
    assert ((wB @ wE) / (alarm << p)).MM(1,0) == State([0.0228947, 0.9771], B)
    #
    # Burglary probability with Jeffrey's rule
    #
    assert (alarm.dagger(wB @ wE) >> s).MM(1,0) == State([0.69303, 0.306968], B)
    #
    # Point evidence gives equal outcome (for Pearl and Jeffrey)
    # 
    assert ((wB @ wE) / (alarm << point_pred('a',A))).MM(1,0) \
        == (alarm.dagger(wB @ wE) >> point_state('a',A)).MM(1,0)


def test_compentence_experience():
    C = Space("Competence", ['c', '~c'])
    E = Space("Experience", ['e', '~e'])
    w = State([4/10, 1/10, 1/10, 4/10], C @ E)
    #
    # A priori competence is uniform
    #
    assert w.MM(1,0) == State([1/2,1/2], C)
    #
    # Observing experience increases competence
    #
    assert (w / (proj2(C,E) << point_pred('e',E))).MM(1,0) \
        == State([4/5,1/5], C)
    #
    # Incompetence surprise, incorporated via adaptation
    #
    r = State([1/8,7/8], C)
    w1 = proj1(C,E).dagger(w) >> r
    assert w1 == State([1/10, 1/40, 7/40, 7/10], C @ E)
    #
    # Experience after surprise is below half
    #
    assert (w1 / (proj2(C,E) << point_pred('e',E))).MM(1,0) \
        == State([4/11,7/11], C)


def test_whitworth_race():
    R = Space("Contenders", ['A', 'B', 'C'])
    T = range_sp(2)
    w = State([2/11, 4/11, 5/11], R)
    #
    # (Deterministic) channel R --> T, giving partition of R into A and B,C
    # 
    f = chan_fromstates([point_state(1,T), 
                         point_state(0,T), 
                         point_state(0,T)], R)
    #
    # Adapt to info that A has 50% chance 
    #
    w1 = f.dagger(w) >> State([1/2,1/2],T)
    assert w1 == State([1/2, 2/9, 5/18], R)


def test_pearl_burglary_alarm():
    A = Space("A", ['a','~a'])
    B = Space("B", ['B','~b'])
    w = State([1/200, 7/500, 1/1000, 98/100], A @ B)
    p = Predicate([0.8,0.2],A)
    assert (w / (p @ truth(B))).MM(0,1) == State([21/1057, 1036/1057], B)
    c = w[ [1,0] : [0,1] ]
    d = c.dagger(w.MM(0,1))
    assert (d >> State([4/5,1/5],A)) == State([19639/93195, 73556/93195], B)


def test_jeffrey_colors():
    C = Space("Colours", ['g', 'b', 'v'])
    S = Space("Sold", ['s', '~s'])
    w = State([3/25, 9/50, 3/25, 9/50, 8/25, 2/25], C @ S)
    v = State([0.7, 0.25, 0.05], C)
    p = v.as_pred()
    #
    # Pearl's update, in three ways
    #
    assert (w / (p @ truth(S))).MM(0,1) == State([26/61, 35/61], S)
    #
    # c : C --> S
    # d : S --> v
    #
    c = w[ [0,1] : [1,0] ]
    d = w[ [1,0] : [0,1] ]
    assert (w.MM(0,1) / (d << p)) == State([26/61, 35/61], S)
    assert (c >> (w.MM(1,0) / p)) == State([26/61, 35/61], S)
    #
    # Jeffrey's adaptation, in two ways
    #
    assert (d.dagger(w.MM(0,1)) >> v) == State([21/50, 29/50], S)
    assert (c >> v) == State([21/50, 29/50], S)
    assert d.dagger(w.MM(0,1)) == c
    #
    # Statement about original formulation
    #
    assert tuple_chan(idn(C),c) >> v \
        == State([14/50, 21/50, 1/10, 3/20, 1/25, 1/100], C @ S)


##############################################################
#
# Chapter 4. Learning of States and Channels
#
##############################################################


def test_missing_data_no_prior():
    #
    # Example from from Jensen-Nielsen 6.2.1
    #
    P = Space("Pregnancy", ['p', '~p'])
    B = Space("Blood test", ['b', '~b'])
    U = Space("Urine test", ['u', '~u'])
    #
    # Learn missing data by inserting uniform distributions at holes
    #
    table = (uniform_state(P) @ point_state('b',B) @ point_state('u',U)) \
            + (point_state('p',P) @ point_state('~b',B) @ point_state('u',U)) \
            + (point_state('p',P) @ point_state('b',B) @ uniform_state(U)) \
            + (point_state('p',P) @ point_state('b',B) @ point_state('~u',U)) \
            + (uniform_state(P) @ point_state('~b',B) @ uniform_state(U))
    assert table[[1,0,0]:[0,0,0]].as_state() == State([4/5,1/5], P)
    assert table[[0,1,0]:[1,0,0]]('p') == State([5/8, 3/8], B)
    assert table[[0,1,0]:[1,0,0]]('~p') == State([1/2, 1/2], B)
    assert table[[0,0,1]:[1,0,0]]('p') == State([9/16, 7/16], U)
    assert table[[0,0,1]:[1,0,0]]('~p') == State([3/4, 1/4], U)
    #
    # Exercise: learn via a uniform Bayesian network
    #
    wP = uniform_state(P)
    cB = uniform_chan(P,B)
    cU = uniform_chan(P,U)
    joint = tuple_chan(idn(P), cB, cU) >> wP
    # 
    # Update the joint state, with truth predicate at holes
    #
    joint1 = joint / (truth(P) @ point_pred('b',B) @ point_pred('u',U))
    joint2 = joint / (point_pred('p',P) @ point_pred('~b',B) @ point_pred('u',U))
    joint3 = joint / (point_pred('p',P) @ point_pred('b',B) @ truth(U))
    joint4 = joint / (point_pred('p',P) @ point_pred('b',B) @ point_pred('~u',U))
    joint5 = joint / (truth(P) @ point_pred('~b',B) @ truth(U))
    #
    # Add these states up and disintegrate the 
    #
    new_joint = joint1 + joint2 + joint3 + joint4 + joint5
    new_wP = new_joint[ [1,0,0] : [0,0,0] ].as_state()
    new_cB = new_joint[ [0,1,0] : [1,0,0] ]
    new_cU = new_joint[ [0,0,1] : [1,0,0] ]
    #
    # This gives the same outcomes as before
    #
    assert new_cB('p') == table[[0,1,0]:[1,0,0]]('p')
    assert new_cB('~p') == table[[0,1,0]:[1,0,0]]('~p')
    assert new_cU('p') == table[[0,0,1]:[1,0,0]]('p')
    assert new_cU('~p') == table[[0,0,1]:[1,0,0]]('~p')
    assert new_wP == State([4/5,1/5], P)


def test_missing_data_with_prior():
    #
    # Example from from Darwiche 17.3.1
    #
    A = Space("A", ['a', '~a'])
    B = Space("B", ['b', '~b'])
    C = Space("C", ['c', '~c'])
    D = Space("D", ['d', '~d'])
    uA = Predicate([0.5,0.5],A)
    uC = Predicate([0.5,0.5],C)
    uD = Predicate([0.5,0.5],D)
    b = point_pred('b',B)
    nb = point_pred('~b',B)
    c = point_pred('c',C)
    nc = point_pred('~c',C)
    d = point_pred('d',D)
    nd = point_pred('~d',D)
    #
    # Bayesian network from Darwiche, Figure 17.3
    #
    wA = flip(0.2,A)
    f = chan_fromstates([flip(0.75,B),flip(0.1,B)], A)
    g = chan_fromstates([flip(0.5,C),flip(0.25,B)], A)
    h = chan_fromstates([flip(0.2,D),flip(0.7,D)], B)
    joint = ((idn(A) @ idn(B) @ swap(D,C)) \
             * (idn(A) @ idn(B) @ h @ idn(C)) \
             * (idn(A) @ (copy(B) * f) @ g) \
             * copy(A,3)) >> wA
    joint1 = joint / \
             (truth(A) @ point_pred('b',B) @ point_pred('~c',C) @ truth(D))
    joint2 = joint / \
             (truth(A) @ point_pred('b',B) @ truth(C) @ point_pred('~d',D))
    joint3 = joint / \
             (truth(A) @ point_pred('~b',B) @ point_pred('c',C) @ point_pred('d',D))
    # 
    # Update the joint state, with truth predicate at holes
    #
    new_joint = 0.2 * joint1 + 0.4 * joint2 + 0.4 * joint3 
    new_wA = new_joint.MM(1,0,0,0)
    new_f = new_joint[ [0,1,0,0] : [1,0,0,0] ]
    new_g = new_joint[ [0,0,1,0] : [1,0,0,0] ]
    new_h = new_joint[ [0,0,0,1] : [0,1,0,0] ]
    assert new_wA == State([3571/8487, 4916/8487], A)
    assert new_f('a') == State([3157/3571, 414/3571], B)
    assert new_f('~a') == State([2419/6145, 3726/6145], B)
    assert new_g('a') == State([1521/3571, 2050/3571], C)
    assert new_g('~a') == State([819/1229, 410/1229], C)
    assert new_h('b') == State([1/15, 14/15], D)
    assert new_h('~b') == State([1, 0], D)
    #
    # Exercise
    #
    assert joint1.MM(1,0,0,1) == State([1/9, 4/9, 4/45, 16/45], A @ D)


#
# Todo: HMMs
#



def test_multiple_state_validities():
    s1 = flip(1/2)
    s2 = flip(1/4)
    s3 = flip(3/8)
    p = Predicate([1,2], s1.sp)
    #
    # Point-data validities
    #
    assert Mval_point(s1, p) == 1/8
    assert Mval_point(s2, p) == 9/64
    assert Mval_point(s3, p) == 75/512
    c = chan_fromstates([State([1/3,1/3,1/3], range_sp(3)),
                         State([1/2,1/3,1/6], range_sp(3))], s1.sp)
    q = Predicate([0,3,6], range_sp(3))
    #
    # M-validity and state/predicate transformation
    #
    assert np.isclose(Mval_point(c >> s1, q), 1/110592)
    assert Mval_point(s1, c << q) == 1/32


def test_non_maximality_for_non_point_data():
    X = range_sp(2)
    w = State([1/4,3/4], X)
    p1 = Predicate([1/2,3/4], X)
    p2 = Predicate([1,1/2], X)
    P = Predicate([1,1], Space("", [p1,p2]))
    assert Mval(w, P) == 55/128
    assert Mlrn(w, P) == State([16/55, 39/55], X)
    assert np.isclose(Mval(Mlrn(w, P), P), 10579/24200)
    v = State([1/2,1/2], X)
    assert Mval(v, P) == 15/32
    #
    # Multiple-state conditioning with P does not give highest
    # M-validity of P
    #
    assert Mval(Mlrn(w, P), P) < Mval(v, P)


def test_bag_learnin():
    #
    # Backtrack EM example from Norvig-Russell, fig. 20.10
    #
    T = range_sp(2)
    prior = flip(0.6, T)
    F = Space("Flavor", ['c', 'l'])
    W = Space("Wrapper", ['r', 'g'])
    H = Space("Holes", ['h', '~h'])
    f0 = chan_fromstates([flip(0.6,F), flip(0.4,F)], T)
    w0 = chan_fromstates([flip(0.6,W), flip(0.4,W)], T)
    h0 = chan_fromstates([flip(0.6,H), flip(0.4,H)], T)
    e = tuple_chan(f0, w0, h0)
    assert e(0) == State([216/1000, 144/1000, 144/1000, 96/1000,
                          144/1000, 96/1000, 96/1000, 64/1000], F @ W @ H)
    assert e(1) == State([64/1000, 96/1000, 96/1000, 144/1000, 
                          96/1000, 144/1000, 144/1000, 216/1000], F @ W @ H)
    d = e.dagger(prior)
    assert d('c', 'r', 'h') == State([81/97, 16/97], T)
    assert d('c', 'r', '~h') == State([27/39, 12/39], T)
    assert d('c', 'g', 'h') == State([27/39, 12/39], T)
    assert d('c', 'g', '~h') == State([1/2, 1/2], T)
    assert d('l', 'r', 'h') == State([27/39, 12/39], T)
    assert d('l', 'r', '~h') == State([1/2, 1/2], T)
    assert d('l', 'g', 'h') == State([1/2, 1/2], T)
    assert d('l', 'g', '~h') == State([12/39, 27/39], T)
    data = Predicate([273, 93, 104, 90, 79, 100, 94, 167], F @ W @ H)
    # 
    # M-learning
    bags = Mlrn_point_chan(prior, e, data)
    assert bags == State([30891/50440, 19549/50440], T)
    assert bags == d >> data.flrn()
    #
    # Learned channels, exactly as on p.730
    #
    dd = d.dagger(data.flrn())
    f1 = dd.MM(1,0,0)
    w1 = dd.MM(0,1,0)
    h1 = dd.MM(0,0,1)
    assert f1(0) == State([0.66841, 0.33159], F)
    assert f1(1) == State([0.388695, 0.61130], F)
    assert w1(0) == State([0.64831, 0.35169], W)
    assert w1(1) == State([0.381748, 0.61825], W)
    assert h1(0) == State([0.655847, 0.34415], H)
    assert h1(1) == State([0.382740, 0.61726], H)
    #
    # Log likelihoods, as in Russel-Norvig
    #
    # Quote: The log likelihood of the data increases from about -2044
    # initially to about -2021 after the first iteration ... That is,
    # the update improves the likelihood itself by a factor e^23 ~
    # 10^10. By the tenth iteration the learned model is a better vit
    # that the original model (L = -1982.214). Thereafter, progress
    # becomes very slow.
    #
    # Before learning
    #
    assert log_Mval_point(e >> prior, data) == -2044.260364580929
    #
    # After learning
    #
    assert log_Mval_point(tuple_chan(f1,w1,h1) >> bags, data) \
        == -2021.0262390280006
    #
    # Without decomposing the double dagger
    #
    assert log_Mval_point(dd >> bags, data) == -1979.3601270422814


def test_four_animals():
    #
    # Rao example from DempsterLR77
    Y = range_sp(4)
    N = 100
    w = discretized_uniform(0, 1, N)
    e = chan_fromklmap(lambda r: State([1/2 + 1/4 * r, 
                                    1/4 * (1 - r),
                                    1/4 * (1 - r),
                                    1/4 * r],Y), w.sp, Y)
    data = Predicate([125, 18, 20, 34], Y)
    #
    # M-learning of r
    #
    for i in range(100):
        w = Mlrn_point_chan(w, e, data)
    #
    # average as reported in the book
    #
    assert np.isclose(w.expectation(), 0.6268214581)
    #
    # C-learning of r
    N = 1000
    w = discretized_uniform(0, 1, N)
    e = chan_fromklmap(lambda r: State([1/2 + 1/4 * r, 
                                        1/4 * (1 - r),
                                        1/4 * (1 - r),
                                        1/4 * r],Y), w.sp, Y)
    for i in range(100):
        w = Clrn_point_chan(w, e, data)
    #
    # average as reported in the book
    #
    assert np.isclose(w.expectation(), 0.6267803732)


def test_shape_classification():
    Y = range_sp(3)
    Z = range_sp(2)
    N = 100
    w = discretized_uniform(0, 1, N)
    X = w.sp
    e = chan_fromklmap(lambda r: State([1/4, 
                                        1/4 * (1 + r),
                                        1/4 * (2 - r)],Y), X, Y)
    d = chan_fromstates([point_state(0,Z),
                         point_state(0,Z),
                         point_state(1,Z)], Y)
    Zdata = Predicate([63, 37], Z)
    #
    # Initially
    #
    assert (e >> w) == State([0.25, 0.37375, 0.37625], Y)
    assert w.expectation() == 0.495
    #
    # 10 steps of M-learning
    #
    for i in range(10):
        w = Mlrn_point_chan(w, d * e, Zdata)
    assert (e >> w) == State([0.25, 0.37501, 0.37499], Y)
    assert np.isclose(w.expectation(), 0.50003)
    #
    # 25 steps of M-learning
    #
    w = discretized_uniform(0, 1, N)
    for i in range(25):
        w = Mlrn_point_chan(w, d * e, Zdata)
    assert (e >> w) == State([0.25, 0.376436, 0.373564], Y)
    assert np.isclose(w.expectation(), 0.50574)
    #
    # 100 steps of M-learning
    #
    w = discretized_uniform(0, 1, N)
    for i in range(100):
        w = Mlrn_point_chan(w, d * e, Zdata)
    assert (e >> w) == State([0.25, 0.37934, 0.37066], Y)
    assert np.isclose(w.expectation(), 0.51736)
    #
    # 250 steps of M-learning
    #
    w = discretized_uniform(0, 1, N)
    for i in range(250):
        w = Mlrn_point_chan(w, d * e, Zdata)
    assert (e >> w) == State([0.25, 0.379977, 0.370023], Y)
    assert np.isclose(w.expectation(), 0.51991)
    #
    # 10 steps of C-learning
    #
    w = discretized_uniform(0, 1, N)
    for i in range(10):
        w = Clrn_point_chan(w, d * e, Zdata)
    assert (e >> w) == State([0.25, 0.379741, 0.370259], Y)
    assert np.isclose(w.expectation(), 0.51896)
    #
    # 25 steps of C-learning
    #
    w = discretized_uniform(0, 1, N)
    for i in range(25):
        w = Clrn_point_chan(w, d * e, Zdata)
    assert (e >> w) == State([0.25, 0.379896, 0.370104], Y)
    assert np.isclose(w.expectation(), 0.51958)
    #
    # 60 steps of C-learning
    #
    w = discretized_uniform(0, 1, N)
    for i in range(60):
        w = Clrn_point_chan(w, d * e, Zdata)
    assert (e >> w) == State([0.25, 0.379957, 0.370043], Y)
    assert np.isclose(w.expectation(), 0.51983)


def test_coin_classification():
    #
    # Coin multisets from DoB08, see also slides version
    #  https://ibug.doc.ic.ac.uk/media/uploads/documents/expectation_maximization-1.pdf
    # and elaborate explanation:
    # http://karlrosaen.com/ml/notebooks/em-coin-flips/
    #
    C = Space(None, ['H', 'T'])
    X = range_sp(2)
    u2 = uniform_state(X)
    e = chan_fromstates([flip(3/5, C), flip(1/2, C)], X)
    mss = [ Predicate([5,5], C), 
            Predicate([9,1], C), 
            Predicate([8,2], C), 
            Predicate([4,6], C), 
            Predicate([7,3], C) ]
    #
    # C-Learning from all multisets separately
    #
    assert Clrn_point_chan(u2, e, mss[0]) == State([0.449149, 0.550851], X)
    assert Clrn_point_chan(u2, e, mss[1]) == State([0.804986, 0.195014], X)
    assert Clrn_point_chan(u2, e, mss[2]) == State([0.733467, 0.266533], X)
    assert Clrn_point_chan(u2, e, mss[3]) == State([0.352156, 0.647844], X)
    assert Clrn_point_chan(u2, e, mss[4]) == State([0.647215, 0.352785], X)
    #
    # M-Learning from all multisets separately
    #
    assert Mlrn_point_chan(u2, e, mss[0]) == State([0.494949, 0.505051], X)
    assert Mlrn_point_chan(u2, e, mss[1]) == State([0.535354, 0.464646], X)
    assert Mlrn_point_chan(u2, e, mss[2]) == State([0.525253, 0.474747], X)
    assert Mlrn_point_chan(u2, e, mss[3]) == State([0.484848, 0.515152], X)
    assert Mlrn_point_chan(u2, e, mss[4]) == State([0.515152, 0.484848], X)
    #
    # The learning algorithm in a single function
    #
    def clrn_chan(stat, chan, point_data_list):
        N = len(point_data_list)
        if N == 0:
            raise Exception('Learning requires non-empty list of data')
        joints = [ 1/N * (Clrn_point_chan(stat, chan, d) @ d.flrn())
                   for d in point_data_list ]
        joint = functools.reduce(lambda x,y: x + y, joints)
        dom_size = len(chan.dom)
        cod_size = len(chan.cod)
        m1 = dom_size * [1] + cod_size * [0]
        m2 = dom_size * [0] + cod_size * [1]
        return ( joint.MM(*m1), joint[ m2 : m1 ] )
    #
    # Learned state and channel, as in example
    #
    lrn = clrn_chan(uniform_state(range_sp(2)), e, mss)
    assert lrn[0] == State([0.597395, 0.402605], X)
    assert lrn[1](0) == State([0.713012, 0.286988], C)
    assert lrn[1](1) == State([0.581339, 0.418661], C)
    #
    # Learned state and channel, after 10 iterations
    #
    lrn = (uniform_state(range_sp(2)), e)
    for i in range(10):
        lrn = clrn_chan(*lrn, mss)
    assert lrn[0] == State([0.537636, 0.462364], X)
    assert lrn[1](0) == State([0.789933, 0.210067], C)
    assert lrn[1](1) == State([0.508914, 0.491086], C)


def test_learn_along_channel():
    #
    # Exercise
    #
    X = Space("X", ['v', 'w'])
    Y = Space("Y", ['a', 'b', 'c'])
    u = uniform_state(X)
    e = chan_fromstates([State([1/3, 2/3, 0], Y), 
                         State([1/8, 1/2, 3/8], Y)], X)
    ap = point_pred('a', Y)
    bp = point_pred('b', Y)
    cp = point_pred('c', Y)
    #
    # pulled back point-predicates
    #
    assert (e << ap) == Predicate([1/3, 1/8], X)
    assert (e << bp) == Predicate([2/3, 1/2], X)
    assert (e << cp) == Predicate([0, 3/8], X)
    #
    # M-validity along e, in various ways (with rounding corrections)
    #
    assert ((u >= e << ap) * (u >= e << bp) ** 2 * (u >= e << cp)) \
        == 539 / 36864 - 0.000000000000000005
    assert Mval(e >> u, Predicate([1,2,1], Space("", [ap, bp, cp]))) \
        == 539 / 36864 - 0.000000000000000005
    assert Mval_point(e >> u, Predicate([1,2,1], Space("", ['a', 'b', 'c']))) \
        == 539 / 36864 - 0.000000000000000005
    assert Mval(u, Predicate([1,2,1], Space("", [e << ap, e << bp, e << cp]))) \
        == 539 / 36864 - 0.000000000000000005
    assert Mval_chan(u, e, Predicate([1,2,1], Space("", [ap, bp, cp]))) \
        == 539 / 36864 - 0.000000000000000005
    assert Mval_point_chan(u, e, Predicate([1,2,1], Space("", ['a', 'b', 'c']))) \
        == 539 / 36864 - 0.000000000000000005
    #
    # C-validity along e, in various ways
    #
    assert (u >= (e << ap) & (e << bp) & (e << bp) & (e << cp)) \
        == 3/512
    assert (u >= (e << ap) & (e << bp) ** 2 & (e << cp)) \
        == 3/512
    assert Cval(u, Predicate([1,2,1], Space("", [e << ap, e << bp, e << cp]))) \
        == 3/512
    assert Cval_chan(u, e, Predicate([1,2,1], Space("", [ap, bp, cp]))) \
        == 3/512
    assert Cval_point_chan(u, e, Predicate([1,2,1], Space("", ['a', 'b', 'c']))) \
        == 3/512
    #
    # M-learning along e
    #
    assert (1/4 * (u / (e << ap)) + 1/2 * (u / (e << bp)) + 1/4 * (u / (e << cp))) \
        == State([36/77, 41/77], X)
    assert Mlrn(u, Predicate([1,2,1], Space("", [e << ap, e << bp, e << cp]))) \
        == State([36/77, 41/77], X)
    Mu = Mlrn_chan(u, e, Predicate([1,2,1], Space("", [ap, bp, cp])))
    assert Mu == State([36/77, 41/77], X)
    #
    # M-validity after M-learning along e
    #
    assert Mval_chan(Mu, e, Predicate([1,2,1], Space("", [ap, bp, cp]))) \
        == 133476771 / 8999178496 + 0.000000000000000003
    #
    # C-learning along e
    #
    assert u / ((e << ap) & (e << bp) ** 2 & (e << cp)) \
        == State([0,1], X)
    assert Clrn(u, Predicate([1,2,1], Space("", [e << ap, e << bp, e << cp]))) \
        == State([0,1], X)
    Cu = Clrn_chan(u, e, Predicate([1,2,1], Space("", [ap, bp, cp])))
    assert Cu == State([0,1], X)
    #
    # C-validity after C-learning along e
    #
    assert (Cu >= ((e << ap) & (e << bp) ** 2 & (e << cp))) == 3/256
    assert Cval_chan(Cu, e, Predicate([1,2,1], Space("", [ap, bp, cp]))) \
        == 3/256 


def test_coursera():
    #
    # Exercise adapted from:
    #
    # https://www.coursera.org/lecture/bayesian-methods-in-machine-learning/example-em-for-discrete-mixture-e-step-hTye6
    #
    # https://www.coursera.org/lecture/bayesian-methods-in-machine-learning/example-em-for-discrete-mixture-m-step-8ee72
    #
    X = range_sp(2)
    Y = range_sp(3)
    data = Predicate([3,2,5], Y)
    alpha = 0.5
    beta = 0.5
    gamma = 0.5
    c = chan_fromstates([State([alpha, 1-alpha, 0], Y),
                         State([0, 1-beta, beta], Y)], X)
    prior = State([gamma, 1-gamma], X)
    d = c.dagger(prior)
    dd = d.dagger(data.flrn())
    post = d >> data.flrn()
    assert post == State([2/5, 3/5], X)
    assert Mlrn_point_chan(prior, c, data) == State([2/5, 3/5], X)
    assert dd(0) == State([3/4, 1/4, 0], Y)
    assert dd(1) == State([0, 1/6, 5/6], Y)
    assert dd >> post == State([3/10, 1/5, 1/2], Y)
    assert (dd.dagger(post) >> data.flrn()) == post
    #
    # Correctly claimed at Coursera: 
    #
    # alpha = 3/4
    #
    # but incorrectly claimed: 
    #
    # beta = 6/7
    # gamma = 4/11
    #










