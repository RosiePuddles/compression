from math import log2
from time import time

def XORsum(k, s) -> int:
    out = 0
    for bit_shift in s:
        out ^= k << bit_shift
    return out


def bitRep(e: int, l: int = 8) -> str:
    return ''.join(('{0:0' + str(l) + 'b}').format(e, 'b'))
