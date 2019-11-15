"""
Test functions based on illustrations and exercises from 
the book (in progress):

Bart Jacobs, Structured Probabilistic Reasoning
Draft available from:

http://www.cs.ru.nl/B.Jacobs/PAPERS/ProbabilisticReasoning.pdf

"""

from efprob import *
from builtins import *



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
    rv = Predicate([100, -50], bool_sp)
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


def test_joint_var_cor():
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



