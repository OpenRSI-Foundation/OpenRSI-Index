"""An explicitly incomplete high-temperature local expansion, not a solution."""
import math


def pressure(K):
    return math.log(2) + 1.5 * K * K
