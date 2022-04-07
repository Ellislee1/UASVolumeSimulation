import numpy as np
import numba as nb

@nb.njit(nopython=True)
def getDistance(p1, p2):
    p1 = complex(p1[0],p1[1])
    p2 = complex(p2[0],p2[1])

    return abs(p1-p2)