"""
Helper functions
"""

import operator
import functools
import math


def _prod(iterable):
    return functools.reduce(operator.mul, iterable, 1)

#
# real number version of the multinomial function that sends
# (r_1, ..., r_n) to the fraction
#
# (sum r_i)! / prod (r_i!)
#
# This version uses the Gamma function instead of the factorial,
# using that Gamma(n) = (n-1)!
#


def _multinomial_real(nums):
    facs = map(special.gamma, map(lambda x: 1+x, nums))
    return math.floor(special.gamma(1 + sum(nums))
                      / functools.reduce(lambda x, y: x*y, facs))


#
# Functions for masks
#

#
# super_mask and sub_mask must have the same length. The result
# removes all entries from sub_mask which are 0 in super_mask
#
def mask_restrict(super_mask, sub_mask):
    if len(super_mask) != len(sub_mask):
        raise Exception('Length mismatch in mask restriction')
    result = []
    for i in range(len(super_mask)):
        if super_mask[i] == 1:
            result = result + [sub_mask[i]]
    return result

#
# Add two masks pointwise, throwing an error if there are two 1's at a
# particular position.
#


def mask_sum(mask1, mask2):
    n = len(mask1)
    if len(mask2) != n:
        raise Exception('Length mismatch in mask summation')
    sum_mask = [mask1[i] + mask2[i] for i in range(n)]
    # check disjointness after summation
    if any([i > 1 for i in sum_mask]):
        raise Exception('Non-disjoint masks in mask summation')
    return sum_mask

#
# Repeated sum of masks
#


def mask_summation(mask_list):
    return reduce(lambda m1, m2: mask_sum(m1, m2), mask_list)

#
# Use mask as filter for a list
#


def mask_filter(ls, mask):
    if len(ls) != len(mask):
        raise Exception('Length mismatch in mask filtering')
    return [ls[i] for i in range(len(ls)) if mask[i] != 0]

#
# Check disjointness of two masks: there are no two 1's at a
# particular position.
#


def mask_disjointness(mask1, mask2):
    n = len(mask1)
    if len(mask2) != n:
        raise Exception('Length mismatch in mask disjointness')
    return any([mask1[i] + mask2[i] <= 1 for i in range(n)])
