#
# EfProb extension for Markov Decision Processes
#
# EfProb:
# Channel-based library for discrete probability calculations
#
# Copyright, 2020:
#   Bart Jacobs, Radboud University Nijmegen
#
from efprob import *
from builtins import *

class Mdp:
    def __init__(self, transition_channel_list, reward):
        """ Reward is a predicate on sp and is used as observable.
        All transition channels in the list must be of type sp -> sp
        """
        self.reward = reward
        self.sp = reward.sp
        if transition_channel_list == []:
            raise ValueError("MDP must have at least one transition channel")
        self.arguments_number = len(transition_channel_list)
        if not all([c.dom == self.sp for c in transition_channel_list]) \
           or \
           not all([c.cod == self.sp for c in transition_channel_list]):
            raise ValueError("Transitions channels must all have the same domain and codomain")
        self.transition_channel_list = transition_channel_list

    def __repr__(self):
        return "Markov Decision Process with:\n* space: " + \
            repr(self.sp) + \
            "\n* " + repr(self.arguments_number) + " transition channels" + \
            "\n* reward: " + repr(self.reward)

    def channel_from_policy(self, policy):
        """ Return a single channel that follows the choice of actions in
        the policy. The policy can be channel or a function sp -> range_sp(n) 
        where n is self.arguments_number.
        """
        if isinstance(policy, Channel):
            def aux(y): return self.transition_channel_list[y]
            return chan_fromklmap(lambda x: 
                                  convex_sum([(policy(x)(*y), aux(*y)(x))
                                              for y in policy(x).sp.iter_all()]),
                                  self.sp, self.sp)
        else:
            return chan_fromklmap(lambda x:
                                  self.transition_channel_list[policy(x)](x),
                                  self.sp, self.sp)

    def return_operator(self, gamma):
        """ The return operator, from predicates to predicates, taking
        the maximum over all actions 
        """
        def aux(pred):
            return [c << pred for c in self.transition_channel_list]
        return lambda pred: pred_fromfun(lambda x: self.reward(x) + \
                                         gamma * max([p(x) for p in aux(pred)]),
                                         self.sp)
            
    def policy_operator(self, gamma, policy):
        """ The operator from predicates to predicates associated with a
        specific policy
        """
        c = self.channel_from_policy(policy)
        return lambda pred: self.reward + gamma * (c << pred)

    def greedy(self, gamma, iterations=100):
        """ The (deterministic) greedy policy obtained after a number of
        iterations of the return operator
        """ 
        R = self.return_operator(gamma)
        p = falsum(self.sp)
        for i in range(iterations):
            p = R(p)
        preds = [ c << p for c in self.transition_channel_list ]
        def aux(x):
            m = preds[0](x)
            max_position = 0
            for i in range(self.arguments_number - 1):
                mi = preds[i+1](x)
                if mi > m:
                    m = mi
                    max_position = i+1
            return max_position
        return aux
