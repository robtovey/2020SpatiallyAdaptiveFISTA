'''
Created on 9 Apr 2020

@author: Rob
'''
from numba import jit, prange, i4, i8, f8
from numpy import array, sqrt
_PARALLEL = True
__params = {'nopython':True, 'parallel':False, 'fastmath':True, 'cache':True, 'boundscheck':False}
__pparams = __params.copy(); __pparams['parallel'] = True
__cparams = __params.copy(); __cparams['cache'] = False

_compiled = {}

@jit(['F(F,F,F,F)'.replace('F', 'f8')], **__params)
def len_intersect(x0, x1, y0, y1): return max(min(x1, y1) - max(x0, y0), 0)


@jit(**__cparams)
def _discretise1D(value, arr, dof_map, level, out):
    if arr.shape[0] == 0:
        out[:] = value
        return

    i = 2  # skip root and first child
    while i < dof_map.shape[0]:
        if dof_map[i, 0] == level + 1:
            break
        i += 1

    scale, N = 2 ** ((level - 1) / 2), out.shape[0] // 2
    _discretise1D(value - scale * arr[0], arr[1:i], dof_map[1:i],
                     level + 1, out[:N])
    _discretise1D(value + scale * arr[0], arr[i:], dof_map[i:],
                     level + 1, out[N:])


@jit(**__cparams)
def _from_discrete1D(arr, dof_map, level, out):
    if arr.shape[0] == 1:
        out[:] = 0
        return arr[0] * 2.** (1 - level)
    elif arr.shape[0] == 0:
        out[:] = 0
        return 0
    elif out.shape[0] == 0:
        value = 0
        for i in range(arr.size):
            value += arr[i]
        return value / arr.size * 2.** (1 - level)

    i = 2  # skip root and first child
    while i < dof_map.shape[0]:
        if dof_map[i, 0] == level + 1:
            break
        i += 1

    scale, N = 2 ** ((level - 1) / 2), arr.shape[0] // 2
    value0 = _from_discrete1D(arr[:N], dof_map[1:i], level + 1, out[1:i])
    value1 = _from_discrete1D(arr[N:], dof_map[i:], level + 1, out[i:])

    out[0] = scale * (-value0 + value1)
    return value0 + value1


@jit(**__cparams)
def _discretise2D(value, arr, dof_map, level, out):
    if arr.shape[0] == 0:
        out[:] = value
        return

    I = array([1, 1, 1, 1])
    i, j = 2, 1
    if dof_map.shape[0] > 1:
        while j < 4:
            if dof_map[i, 0] == level + 1:
                I[j] = i
                j += 1
            i += 1

    scale, N = 2 ** float(level - 1), out.shape[0] // 2
    _discretise2D(value + scale * (-arr[0, 0] - arr[0, 1] + arr[0, 2]),
                  arr[I[0]:I[1]], dof_map[I[0]:I[1]], level + 1, out[:N, :N])
    _discretise2D(value + scale * (-arr[0, 0] + arr[0, 1] - arr[0, 2]),
                  arr[I[1]:I[2]], dof_map[I[1]:I[2]], level + 1, out[:N, N:])
    _discretise2D(value + scale * (+arr[0, 0] - arr[0, 1] - arr[0, 2]),
                  arr[I[2]:I[3]], dof_map[I[2]:I[3]], level + 1, out[N:, :N])
    _discretise2D(value + scale * (+arr[0, 0] + arr[0, 1] + arr[0, 2]),
                  arr[I[3]:], dof_map[I[3]:], level + 1, out[N:, N:])


@jit(**__cparams)
def _from_discrete2D(arr, dof_map, level, out):
    if arr.size == 1:
        out[:] = 0
        return arr[0, 0] * 4.** (1 - level)
    elif arr.size == 0:
        out[:] = 0
        return 0.0
    elif out.shape[0] == 0:
        value = 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                value += arr[i, j]
        return value / arr.size * 4.** (1 - level)

    I = array([1, 1, 1, 1])
    i, j = 2, 1
    if dof_map.shape[0] > 1:
        while j < 4:
            if dof_map[i, 0] == level + 1:
                I[j] = i
                j += 1
            i += 1

    scale, N = 2. ** (level - 1), arr.shape[0] // 2
    value0 = _from_discrete2D(arr[:N, :N], dof_map[I[0]:I[1]], level + 1, out[I[0]:I[1]])
    value1 = _from_discrete2D(arr[:N, N:], dof_map[I[1]:I[2]], level + 1, out[I[1]:I[2]])
    value2 = _from_discrete2D(arr[N:, :N], dof_map[I[2]:I[3]], level + 1, out[I[2]:I[3]])
    value3 = _from_discrete2D(arr[N:, N:], dof_map[I[3]:], level + 1, out[I[3]:])

    out[0, 0] = scale * (-value0 - value1 + value2 + value3)
    out[0, 1] = scale * (-value0 + value1 - value2 + value3)
    out[0, 2] = scale * (+value0 - value1 - value2 + value3)
    return value0 + value1 + value2 + value3


def opt_jit(debug, *args, **kwargs):
    if len(args) > 0:
        args = [args[0].replace('F', 'float64').replace('I', 'int32')]
    else:
        args = None

    def decorator(func):
        if debug:
            return func
        else:
            return jit(args, **kwargs)(func)

    return decorator


def numba_compile(f, DEBUG=False):
    opts = {'nopython':True, 'parallel':False, 'fastmath':True, 'cache':False}
    if f in _compiled:
        return _compiled[f]
    elif f == '_FP_wave2leaf':
        F = opt_jit(DEBUG, 'void(F,F[:,:],I[:,:],I,F[:,:])',
                locals={'i':i4, 'j':i4, 'I':i8[:], 'scale':f8, 'v0':f8, 'v1':f8, 'v2':f8, 'v3':f8}, **opts)(_FP_wave2leaf)
    elif f == '_BP_leaf2wave':
        F = opt_jit(DEBUG, 'F(F[:],I[:,:],I,F[:,:])',
                locals={'i':i4, 'j':i4, 'I':i8[:], 'scale':f8,
                        'value0':f8, 'value1':f8, 'value2':f8, 'value3':f8}, **opts)(_BP_leaf2wave)
    else:
        F = jit(**opts)(globals()[f])
    _compiled[f] = F
    return F


@jit(['void(i4[:,:],F[:],F[:],F[:],F[:])'.replace('F', 'f8')], **__pparams)
def _gradF(dof_map, grid, L, R, DF):
    for i in prange(1, dof_map.shape[0]):
        j = dof_map[i]
        supp0, supp1 = j[1] * 2 ** float(1 - j[0]), (j[1] + 1) * 2 ** float(1 - j[0])
        midpoint = .5 * (supp0 + supp1)

        v = 0
        for k in range(L.size):
            v += R[k] / sqrt(L[k]) * (
                len_intersect(midpoint, supp1, grid[k], grid[k + 1])
                -len_intersect(supp0, midpoint, grid[k], grid[k + 1]))

        DF[i] = 2 ** ((j[0] - 1) / 2) * v


@jit(['F(F,F,F[:],F,F,F,F)'.replace('F', 'f8')], **__params)
def line_intersect_square(px, py, d, x0, y0, x1, y1):
    '''
    The line is p + t*d for t\in R
    The box is bound by the corners [x0,y0] and [x1,y1]

    There are 4 possible intersection points:
        (X0,y0), (X1,y1), (x0,Y0), (x1,Y1)

    We always return the area above (positive y) the line
    If we flip the x axis then we never change the area
    If we flip the y axis then we always change the area of integration
    Using this, we can reduce to the case x0<x1, X0<X1, y0<y1, Y0<Y1

    '''

    # TODO: deal with d[i]=0
    Y0 = py + (x0 - px) / d[0] * d[1]
    Y1 = py + (x1 - px) / d[0] * d[1]
    X0 = px + (y0 - py) / d[1] * d[0]
    X1 = px + (y1 - py) / d[1] * d[0]

    sign = False
    # WLOG X0<X1, Y0<Y1
    if X1 < X0:  # flip in Y-axis
        tmp = X1
        X1 = X0
        X0 = tmp
        Y0, Y1, sign = y1 + y0 - Y0, y1 + y0 - Y1, not sign

    if X1 < x0:  # Line passes above the square
        v = 0
    elif X0 > x1:  # Line passes below the square
        v = (x1 - x0) * (y1 - y0)
    elif X0 >= x0 and X1 <= x1:  # Line passes through top and bottom
        v = (y1 - y0) * (.5 * (X1 + X0) - x0)
    elif Y0 >= y0 and Y1 <= y1:  # Line passes through left and right
        v = (x1 - x0) * (y1 - .5 * (Y1 + Y0))
    elif X0 < x0:  # Line cuts top left corner
        v = .5 * (X1 - x0) * (y1 - Y0)
    else:  # Line cuts bottom right corner
        sign = not sign
        v = .5 * (x1 - X0) * (Y1 - y0)

    if sign:
        v = (x1 - x0) * (y1 - y0) - v

    return v


@jit(['void(F[:,:],I[:,:],b1[:],F[:],F[:,:],F[:],F[:,:])'.replace('F', 'f8').replace('I', 'i4')], **__pparams)
def _FP_leaf(value, dof_map, is_leaf, g, slope, centre, sino):
    for j in prange(sino.shape[0]):
        for i in range(dof_map.shape[0]):
            if is_leaf[i]:
                c = 2.**(1 - dof_map[i, 0])
                b0x, b0y = dof_map[i, 1] * c - centre[0], dof_map[i, 2] * c - centre[1]
                b1x, b1y = b0x + c, b0y + c
                midx, midy = b0x + .5 * c, b0y + .5 * c
                r = 2.**(-.5) * c

                t = midx * slope[j, 1] - midy * slope[j, 0]
                for k in range(g.size - 1):
                    if g[k] > t + r:
                        break
                    elif g[k + 1] > t - r:
                        p00, p01 = g[k] * slope[j, 1], -g[k] * slope[j, 0]
                        p10, p11 = g[k + 1] * slope[j, 1], -g[k + 1] * slope[j, 0]

                        sino[j, k] += (
                            value[i, 0] * abs(
                            line_intersect_square(p10, p11, slope[j], b0x, b0y, midx, midy)
                            -line_intersect_square(p00, p01, slope[j], b0x, b0y, midx, midy))
                            +value[i, 1] * abs(
                            line_intersect_square(p10, p11, slope[j], b0x, midy, midx, b1y)
                            -line_intersect_square(p00, p01, slope[j], b0x, midy, midx, b1y))
                            +value[i, 2] * abs(
                            line_intersect_square(p10, p11, slope[j], midx, b0y, b1x, midy)
                            -line_intersect_square(p00, p01, slope[j], midx, b0y, b1x, midy))
                            +value[i, 3] * abs(
                            line_intersect_square(p10, p11, slope[j], midx, midy, b1x, b1y)
                            -line_intersect_square(p00, p01, slope[j], midx, midy, b1x, b1y))
                        )


@jit(['void(F,F[:,:],I[:,:],I,F[:,:])'.replace('F', 'f8').replace('I', 'i4')], **__cparams)
def _FP_wave2leaf(value, wave, dof_map, level, out):
    scale = 2 ** float(level - 1)
    v0 = value + scale * (-wave[0, 0] - wave[0, 1] + wave[0, 2])  # [x,y]
    v1 = value + scale * (-wave[0, 0] + wave[0, 1] - wave[0, 2])  # [x+h,y]
    v2 = value + scale * (+wave[0, 0] - wave[0, 1] - wave[0, 2])  # [x,y+h]
    v3 = value + scale * (+wave[0, 0] + wave[0, 1] + wave[0, 2])  # [x+h,y+h]

    if wave.shape[0] == 1:
        out[0, 0], out[0, 1], out[0, 2], out[0, 3] = v0, v1, v2, v3
        return

    I = array([1, 1, 1, 1])
    i, j = 2, 1
    if dof_map.shape[0] > 1:
        while j < 4:
            if dof_map[i, 0] == level + 1:
                I[j] = i
                j += 1
            i += 1

    _FP_wave2leaf(v0, wave[I[0]:I[1]], dof_map[I[0]:I[1]], level + 1, out[I[0]:I[1]])
    _FP_wave2leaf(v1, wave[I[1]:I[2]], dof_map[I[1]:I[2]], level + 1, out[I[1]:I[2]])
    _FP_wave2leaf(v2, wave[I[2]:I[3]], dof_map[I[2]:I[3]], level + 1, out[I[2]:I[3]])
    _FP_wave2leaf(v3, wave[I[3]:], dof_map[I[3]:], level + 1, out[I[3]:])


@jit(**__params)
def _Radon_row(i, p00, p01, p10, p11, slope, b0x, midx, b1x, b0y, midy, b1y):
    if i == 0:
        return (
                # bottom left
                -abs(line_intersect_square(p10, p11, slope, b0x, b0y, midx, midy)
                    -line_intersect_square(p00, p01, slope, b0x, b0y, midx, midy))
                # top left
                -abs(line_intersect_square(p10, p11, slope, b0x, midy, midx, b1y)
                    -line_intersect_square(p00, p01, slope, b0x, midy, midx, b1y))
                # bottom right
                +abs(line_intersect_square(p10, p11, slope, midx, b0y, b1x, midy)
                    -line_intersect_square(p00, p01, slope, midx, b0y, b1x, midy))
                # top right
                +abs(line_intersect_square(p10, p11, slope, midx, midy, b1x, b1y)
                    -line_intersect_square(p00, p01, slope, midx, midy, b1x, b1y))
                )
    elif i == 1:
        return (
                # bottom left
                -abs(line_intersect_square(p10, p11, slope, b0x, b0y, midx, midy)
                    -line_intersect_square(p00, p01, slope, b0x, b0y, midx, midy))
                # top left
                +abs(line_intersect_square(p10, p11, slope, b0x, midy, midx, b1y)
                    -line_intersect_square(p00, p01, slope, b0x, midy, midx, b1y))
                # bottom right
                -abs(line_intersect_square(p10, p11, slope, midx, b0y, b1x, midy)
                    -line_intersect_square(p00, p01, slope, midx, b0y, b1x, midy))
                # top right
                +abs(line_intersect_square(p10, p11, slope, midx, midy, b1x, b1y)
                    -line_intersect_square(p00, p01, slope, midx, midy, b1x, b1y))
                )
    else:
        return (
                # bottom left
                +abs(line_intersect_square(p10, p11, slope, b0x, b0y, midx, midy)
                    -line_intersect_square(p00, p01, slope, b0x, b0y, midx, midy))
                # top left
                -abs(line_intersect_square(p10, p11, slope, b0x, midy, midx, b1y)
                    -line_intersect_square(p00, p01, slope, b0x, midy, midx, b1y))
                # bottom right
                -abs(line_intersect_square(p10, p11, slope, midx, b0y, b1x, midy)
                    -line_intersect_square(p00, p01, slope, midx, b0y, b1x, midy))
                # top right
                +abs(line_intersect_square(p10, p11, slope, midx, midy, b1x, b1y)
                    -line_intersect_square(p00, p01, slope, midx, midy, b1x, b1y)))


@jit(**__pparams)
def Radon_to_matrixT(A, dof_map, g, slope, centre):
    b0x, b0y, b1x, b1y = 0 - centre[0], 0 - centre[1], 1 - centre[0], 1 - centre[1]
    midx, midy = b0x + .5, b0y + .5
    r = 2.**(-.5)

    for i0 in range(slope.shape[0]):
        for i1 in range(g.size - 1):
            p00, p01 = g[i1] * slope[i0, 1], -g[i1] * slope[i0, 0]
            p10, p11 = g[i1 + 1] * slope[i0, 1], -g[i1 + 1] * slope[i0, 0]

            A[0, i0, i1] = (
                # bottom left
                +abs(line_intersect_square(p10, p11, slope[i0], b0x, b0y, midx, midy)
                    -line_intersect_square(p00, p01, slope[i0], b0x, b0y, midx, midy))
                # top left
                +abs(line_intersect_square(p10, p11, slope[i0], b0x, midy, midx, b1y)
                    -line_intersect_square(p00, p01, slope[i0], b0x, midy, midx, b1y))
                # bottom right
                +abs(line_intersect_square(p10, p11, slope[i0], midx, b0y, b1x, midy)
                    -line_intersect_square(p00, p01, slope[i0], midx, b0y, b1x, midy))
                # top right
                +abs(line_intersect_square(p10, p11, slope[i0], midx, midy, b1x, b1y)
                    -line_intersect_square(p00, p01, slope[i0], midx, midy, b1x, b1y))
            )
            A[1, i0, i1] = 0
            A[2, i0, i1] = 0

    for j in prange(1, dof_map.shape[0]):
        c = 2.**(1 - dof_map[j, 0])
        scale = 1 / c
        b0x, b0y = dof_map[j, 1] * c - centre[0], dof_map[j, 2] * c - centre[1]
        b1x, b1y = b0x + c, b0y + c
        midx, midy = b0x + .5 * c, b0y + .5 * c
        r = 2.**(-.5) * c

        for i in range(3):  # for each type of wavelet
            for i0 in range(slope.shape[0]):  # for each projection
                t = midx * slope[i0, 1] - midy * slope[i0, 0]  # for each pixel
                for i1 in range(g.size - 1):
                    if g[i1] > t + r:
                        A[3 * j + i, i0, i1] = 0
                    elif g[i1 + 1] > t - r:
                        p00, p01 = g[i1] * slope[i0, 1], -g[i1] * slope[i0, 0]
                        p10, p11 = g[i1 + 1] * slope[i0, 1], -g[i1 + 1] * slope[i0, 0]

                        A[3 * j + i, i0, i1] = scale * _Radon_row(i, p00, p01, p10, p11, slope[i0],
                                                                  b0x, midx, b1x, b0y, midy, b1y)
                    else:
                        A[3 * j + i, i0, i1] = 0


@jit(**__params)
def Radon_to_sparse_matrix(data, indcs, ptr, dof_map, g, slope, centre):
    sz = g.size - 1
    J = 0  # current index of matrix

    # constant pixel
    b0x, b0y, b1x, b1y = 0 - centre[0], 0 - centre[1], 1 - centre[0], 1 - centre[1]
    midx, midy = b0x + .5, b0y + .5
    ptr[0] = J
    for i0 in range(slope.shape[0]):
        for i1 in range(g.size - 1):
            p00, p01 = g[i1] * slope[i0, 1], -g[i1] * slope[i0, 0]
            p10, p11 = g[i1 + 1] * slope[i0, 1], -g[i1 + 1] * slope[i0, 0]

            data[J] = (
                # bottom left
                +abs(line_intersect_square(p10, p11, slope[i0], b0x, b0y, midx, midy)
                    -line_intersect_square(p00, p01, slope[i0], b0x, b0y, midx, midy))
                # top left
                +abs(line_intersect_square(p10, p11, slope[i0], b0x, midy, midx, b1y)
                    -line_intersect_square(p00, p01, slope[i0], b0x, midy, midx, b1y))
                # bottom right
                +abs(line_intersect_square(p10, p11, slope[i0], midx, b0y, b1x, midy)
                    -line_intersect_square(p00, p01, slope[i0], midx, b0y, b1x, midy))
                # top right
                +abs(line_intersect_square(p10, p11, slope[i0], midx, midy, b1x, b1y)
                    -line_intersect_square(p00, p01, slope[i0], midx, midy, b1x, b1y))
            )
            indcs[J] = i0 * sz + i1
            J += 1
    ptr[1] = J  # empty row
    ptr[2] = J  # empty row

    # wavelet pixels
    for j in range(1, dof_map.shape[0]):
        c = 2.**(1 - dof_map[j, 0])
        scale = 1 / c
        b0x, b0y = dof_map[j, 1] * c - centre[0], dof_map[j, 2] * c - centre[1]
        b1x, b1y = b0x + c, b0y + c
        midx, midy = b0x + .5 * c, b0y + .5 * c
        r = 2.**(-.5) * c

        for i in range(3):  # for each wavelet type
            ptr[3 * j + i] = J  # start of new row
            for i0 in range(slope.shape[0]):
                t = midx * slope[i0, 1] - midy * slope[i0, 0]
                for i1 in range(g.size - 1):
                    if g[i1] > t + r:
                        break
                    elif g[i1 + 1] > t - r:
                        p00, p01 = g[i1] * slope[i0, 1], -g[i1] * slope[i0, 0]
                        p10, p11 = g[i1 + 1] * slope[i0, 1], -g[i1 + 1] * slope[i0, 0]

                        data[J] = scale * _Radon_row(i, p00, p01, p10, p11, slope[i0],
                                                      b0x, midx, b1x, b0y, midy, b1y)
                        indcs[J] = i0 * sz + i1
                        J += 1
                        if J >= data.size:
                            raise MemoryError()

    ptr[3 * dof_map.shape[0]] = J  # end of data arrays


@jit(**__params)
def Radon_update_sparse_matrix(o_data, o_indcs, o_ptr, refine,
        n_data, n_indcs, n_ptr, dof_map, g, slope, centre):
    sz = g.size - 1
    J, JJ = 0  # current row/index of new matrix

    for j in range(refine.shape[0]):  # current row of old matrix

        if refine[j] == 0:  # copy old row to new
            for i in range(3):
                n_ptr[3 * J + i] = JJ  # row J starts at index JJ
                for jj in range(o_ptr[3 * j + i], o_ptr[3 * j + i + 1]):
                    n_data[JJ] = o_data[jj]
                    n_indcs[JJ] = o_indcs[jj]
                    JJ += 1
                    if JJ >= n_data.size:
                        raise MemoryError()
            J += 1  # move to new row in new matrix

        elif refine[j] > 0:  # create new rows
            # wavelet pixels
            c = 2.**(1 - dof_map[J, 0])
            scale = 1 / c
            r = 2.**(-.5) * c

            for _ in range(4):  # one pixel splits into 4 consecutive ones
                b0x, b0y = dof_map[J, 1] * c - centre[0], dof_map[J, 2] * c - centre[1]
                b1x, b1y = b0x + c, b0y + c
                midx, midy = b0x + .5 * c, b0y + .5 * c

                for i in range(3):  # for each wavelet type
                    n_ptr[3 * J + i] = JJ  # start of new row
                    for i0 in range(slope.shape[0]):
                        t = midx * slope[i0, 1] - midy * slope[i0, 0]
                        for i1 in range(g.size - 1):
                            if g[i1] > t + r:
                                break
                            elif g[i1 + 1] > t - r:
                                p00, p01 = g[i1] * slope[i0, 1], -g[i1] * slope[i0, 0]
                                p10, p11 = g[i1 + 1] * slope[i0, 1], -g[i1 + 1] * slope[i0, 0]

                                n_data[JJ] = scale * _Radon_row(i, p00, p01, p10, p11, slope[i0],
                                                              b0x, midx, b1x, b0y, midy, b1y)
                                n_indcs[JJ] = i0 * sz + i1
                                JJ += 1
                                if JJ >= n_data.size:
                                    raise MemoryError()
                J += 1  # move to new row in new matrix

    n_ptr[3 * J] = JJ  # end of data arrays


@jit(['void(F[:,:],I[:,:],b1[:],F[:],F[:,:],F[:],F[:,:],F[:])'.replace('F', 'f8').replace('I', 'i4')], **__pparams)
def _BP_leaf(sino, dof_map, is_leaf, g, slope, centre, wave, residue):
    for i in prange(dof_map.shape[0]):
        if is_leaf[i]:
            c = 2.**(1 - dof_map[i, 0])
            b0x, b0y = dof_map[i, 1] * c - centre[0], dof_map[i, 2] * c - centre[1]
            b1x, b1y = b0x + c, b0y + c
            midx, midy = b0x + .5 * c, b0y + .5 * c
            r = 2.**(-.5) * c

            value0, value1, value2, value3 = 0, 0, 0, 0
            for j in range(slope.shape[0]):
                t = midx * slope[j, 1] - midy * slope[j, 0]
                for k in range(g.size - 1):
                    if g[k] > t + r:
                        break
                    elif g[k + 1] > t - r:
                        p00, p01 = g[k] * slope[j, 1], -g[k] * slope[j, 0]
                        p10, p11 = g[k + 1] * slope[j, 1], -g[k + 1] * slope[j, 0]

                        value0 += sino[j, k] * abs(
                            line_intersect_square(p10, p11, slope[j], b0x, b0y, midx, midy)
                            -line_intersect_square(p00, p01, slope[j], b0x, b0y, midx, midy))
                        value1 += sino[j, k] * abs(
                            line_intersect_square(p10, p11, slope[j], b0x, midy, midx, b1y)
                            -line_intersect_square(p00, p01, slope[j], b0x, midy, midx, b1y))
                        value2 += sino[j, k] * abs(
                            line_intersect_square(p10, p11, slope[j], midx, b0y, b1x, midy)
                            -line_intersect_square(p00, p01, slope[j], midx, b0y, b1x, midy))
                        value3 += sino[j, k] * abs(
                            line_intersect_square(p10, p11, slope[j], midx, midy, b1x, b1y)
                            -line_intersect_square(p00, p01, slope[j], midx, midy, b1x, b1y))

            wave[i, 0] = (-value0 - value1 + value2 + value3) / c
            wave[i, 1] = (-value0 + value1 - value2 + value3) / c
            wave[i, 2] = (+value0 - value1 - value2 + value3) / c
            residue[i] = value0 + value1 + value2 + value3
        else:
            residue[i] = -1


@jit(['F(F[:],I[:,:],I,F[:,:])'.replace('F', 'f8').replace('I', 'i4')], **__cparams)
def _BP_leaf2wave(residue, dof_map, level, wave):
    if wave.shape[0] == 1: # leaf
        return residue[0]

    I = array([1, 1, 1, 1])
    i, j = 2, 1
    if dof_map.shape[0] > 1:
        while j < 4:
            if dof_map[i, 0] == level + 1:
                I[j] = i
                j += 1
            i += 1

    scale = 2 ** float(level - 1)
    value0 = _BP_leaf2wave(residue[I[0]:I[1]], dof_map[I[0]:I[1]], level + 1, wave[I[0]:I[1]])
    value1 = _BP_leaf2wave(residue[I[1]:I[2]], dof_map[I[1]:I[2]], level + 1, wave[I[1]:I[2]])
    value2 = _BP_leaf2wave(residue[I[2]:I[3]], dof_map[I[2]:I[3]], level + 1, wave[I[2]:I[3]])
    value3 = _BP_leaf2wave(residue[I[3]:], dof_map[I[3]:], level + 1, wave[I[3]:])

    wave[0, 0] = scale * (-value0 - value1 + value2 + value3)
    wave[0, 1] = scale * (-value0 + value1 - value2 + value3)
    wave[0, 2] = scale * (+value0 - value1 - value2 + value3)
    return value0 + value1 + value2 + value3


@jit(['void(F[:],F[:],F,F)'.replace('F', 'f8')], **__params)
def _smallgrad(df, u, a, eps):
    for i in prange(df.size):
        if u[i] > eps:  # u>0
            df[i] += a
        elif u[i] < -eps:  # u<0
            df[i] -= a
        elif df[i] > a:  # u=0, df>a
            df[i] -= a
        elif df[i] < -a:  # u=0, df<-a
            df[i] += a
        else:
            df[i] = 0



###################################################################################################
# # Copyright (C) 2010  Alex Opie  <lx_op@orcon.net.nz>
# #
# # This program is free software; you can redistribute it and/or modify it
# # under the terms of the GNU General Public License as published by
# # the Free Software Foundation; either version 3 of the License, or (at
# # your option) any later version.
# #
# # This program is distributed in the hope that it will be useful, but
# # WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the GNU
# # General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with this program; see the file COPYING.  If not, see
# # <http://www.gnu.org/licenses/>.


import numpy as np


def phantom (n=256, p_type='Modified Shepp-Logan', ellipses=None):
    """
     phantom (n = 256, p_type = 'Modified Shepp-Logan', ellipses = None)

    Create a Shepp-Logan or modified Shepp-Logan phantom.

    A phantom is a known object (either real or purely mathematical)
    that is used for testing image reconstruction algorithms.  The
    Shepp-Logan phantom is a popular mathematical model of a cranial
    slice, made up of a set of ellipses.  This allows rigorous
    testing of computed tomography (CT) algorithms as it can be
    analytically transformed with the radon transform (see the
    function `radon').

    Inputs
    ------
    n : The edge length of the square image to be produced.

    p_type : The type of phantom to produce. Either
      "Modified Shepp-Logan" or "Shepp-Logan".  This is overridden
      if `ellipses' is also specified.

    ellipses : Custom set of ellipses to use.  These should be in
      the form
          [[I, a, b, x0, y0, phi],
           [I, a, b, x0, y0, phi],
           ...]
      where each row defines an ellipse.
      I : Additive intensity of the ellipse.
      a : Length of the major axis.
      b : Length of the minor axis.
      x0 : Horizontal offset of the centre of the ellipse.
      y0 : Vertical offset of the centre of the ellipse.
      phi : Counterclockwise rotation of the ellipse in degrees,
            measured as the angle between the horizontal axis and
            the ellipse major axis.
      The image bounding box in the algorithm is [-1, -1], [1, 1],
      so the values of a, b, x0, y0 should all be specified with
      respect to this box.

    Output
    ------
    P : A phantom image.

    Usage example
    -------------
      import matplotlib.pyplot as pl
      P = phantom ()
      pl.imshow (P)

    References
    ----------
    Shepp, L. A.; Logan, B. F.; Reconstructing Interior Head Tissue
    from X-Ray Transmissions, IEEE Transactions on Nuclear Science,
    Feb. 1974, p. 232.

    Toft, P.; "The Radon Transform - Theory and Implementation",
    Ph.D. thesis, Department of Mathematical Modelling, Technical
    University of Denmark, June 1996.

    """

    if (ellipses is None):
        ellipses = _select_phantom (p_type)
    elif (np.size (ellipses, 1) != 6):
        raise AssertionError ("Wrong number of columns in user phantom")

    # Blank image
    p = np.zeros ((n, n))

    # Create the pixel grid
    ygrid, xgrid = np.mgrid[-1:1:(1j * n), -1:1:(1j * n)]

    for ellip in ellipses:
        I = ellip [0]
        a2 = ellip [1] ** 2
        b2 = ellip [2] ** 2
        x0 = ellip [3]
        y0 = ellip [4]
        phi = ellip [5] * np.pi / 180  # Rotation angle in radians

        # Create the offset x and y values for the grid
        x = xgrid - x0
        y = ygrid - y0

        cos_p = np.cos (phi)
        sin_p = np.sin (phi)

        # Find the pixels within the ellipse
        locs = (((x * cos_p + y * sin_p) ** 2) / a2
              +((y * cos_p - x * sin_p) ** 2) / b2) <= 1

        # Add the ellipse intensity to those pixels
        p [locs] += I

    return p


def _select_phantom (name):
    if (name.lower () == 'shepp-logan'):
        e = _shepp_logan ()
    elif (name.lower () == 'modified shepp-logan'):
        e = _mod_shepp_logan ()
    else:
        raise ValueError ("Unknown phantom type: %s" % name)

    return e


def _shepp_logan ():
    #  Standard head phantom, taken from Shepp & Logan
    return [[   2, .69, .92, 0, 0, 0],
            [-.98, .6624, .8740, 0, -.0184, 0],
            [-.02, .1100, .3100, .22, 0, -18],
            [-.02, .1600, .4100, -.22, 0, 18],
            [ .01, .2100, .2500, 0, .35, 0],
            [ .01, .0460, .0460, 0, .1, 0],
            [ .02, .0460, .0460, 0, -.1, 0],
            [ .01, .0460, .0230, -.08, -.605, 0],
            [ .01, .0230, .0230, 0, -.606, 0],
            [ .01, .0230, .0460, .06, -.605, 0]]


def _mod_shepp_logan ():
    #  Modified version of Shepp & Logan's head phantom,
    #  adjusted to improve contrast.  Taken from Toft.
    return [[   1, .69, .92, 0, 0, 0],
            [-.80, .6624, .8740, 0, -.0184, 0],
            [-.20, .1100, .3100, .22, 0, -18],
            [-.20, .1600, .4100, -.22, 0, 18],
            [ .10, .2100, .2500, 0, .35, 0],
            [ .10, .0460, .0460, 0, .1, 0],
            [ .10, .0460, .0460, 0, -.1, 0],
            [ .10, .0460, .0230, -.08, -.605, 0],
            [ .10, .0230, .0230, 0, -.606, 0],
            [ .10, .0230, .0460, .06, -.605, 0]]
###################################################################################################
