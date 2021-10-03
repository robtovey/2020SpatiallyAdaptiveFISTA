'''
Created on 4 Jun 2020

@author: rob
'''
from numba import jit, prange
from math import sqrt as c_sqrt
__params = {'nopython':True, 'parallel':False, 'fastmath':True, 'cache':True}
__pparams = __params.copy(); __pparams['parallel'] = True
EPS, FTYPE = 1e-25, 'float64'


@jit(**__params)
def _blur_coarse(CM, coarse_map, dx, r, N):
    for I in range(CM.shape[0]):
        for J in range(CM.shape[1]):
            for i in range(max(0, I - N), min(coarse_map.shape[0], I + N)):
                for j in range(max(0, J - N), min(coarse_map.shape[1], J + N)):
                    distance = c_sqrt((i - I) ** 2 + (j - J) ** 2) * dx - c_sqrt(2) * dx
                    if distance < r:
                        CM[I, J, 0] = min(CM[I, J, 0], coarse_map[i, j, 0])
                        CM[I, J, 1] = max(CM[I, J, 1], coarse_map[i, j, 1])


@jit(**__params)
def _merge2D(old, old_DM, OLD, OLD_DM, new, new_DM):
    i, I, j = 0, 0, 0
    while i < old.size and I < OLD.size:
        x0, y0, h = old_DM[i, 0], old_DM[i, 1], old_DM[i, 2]
        X0, Y0, H = OLD_DM[I, 0], OLD_DM[I, 1], OLD_DM[I, 2]

        if x0 < X0:
            new[j] = old[i]
            new_DM[j, 0], new_DM[j, 1], new_DM[j, 2] = x0, y0, h
            i += 1
        elif X0 < x0 or Y0 < y0:
            new[j] = OLD[I]
            new_DM[j, 0], new_DM[j, 1], new_DM[j, 2] = X0, Y0, H
            I += 1
        elif y0 < Y0 or h < H:
            new[j] = old[i]
            new_DM[j, 0], new_DM[j, 1], new_DM[j, 2] = x0, y0, h
            i += 1
        elif H < h:
            new[j] = OLD[I]
            new_DM[j, 0], new_DM[j, 1], new_DM[j, 2] = X0, Y0, H
            I += 1
        else:  # merge both records
            new[j] = old[i] + OLD[I]
            new_DM[j, 0], new_DM[j, 1], new_DM[j, 2] = x0, y0, h
            i += 1; I += 1
        j += 1

    while i < old.size:
        new[j] = old[i]
        new_DM[j, 0], new_DM[j, 1], new_DM[j, 2] = old_DM[i, 0], old_DM[i, 1], old_DM[i, 2]
        i += 1; j += 1
    while I < OLD.size:
        new[j] = OLD[I]
        new_DM[j, 0], new_DM[j, 1], new_DM[j, 2] = OLD_DM[I, 0], OLD_DM[I, 1], OLD_DM[I, 2]
        I += 1; j += 1

    return j

