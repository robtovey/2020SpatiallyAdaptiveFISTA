'''
Created on 4 Jun 2020

@author: rob
'''
from numba import jit, prange
from math import sqrt as c_sqrt
_PARALLEL = True
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

__params = __params.copy(); __params['cache'] = False
__pparams = __pparams.copy(); __pparams['cache'] = False; __pparams['parallel'] = _PARALLEL
# def jit(*args, **kwargs): return lambda a: a


def ker2func1(base_eval, base_grad, base_int):

    ##################################################
    # r(t) = sum y_j ker_j(t)
    ##################################################
    # Pixel-wise functions
    @jit(**__params)
    def eval_int(y, p0, p1):
        ''' v = int_{p0}^{p1} r(t) dt '''
        v = 0
        for j in range(y.size):
            v += y[j] * base_int(j, p0, p1)
        return v

    @jit(**__params)
    def eval_mid(y, p0, p1):
        ''' v = r((p0+p1)/2) '''
        p = .5 * (p0 + p1)
        v = 0
        for j in range(y.size):
            v += y[j] * base_eval(j, p)
        return v

    @jit(**__params)
    def eval_grad(y, p):
        ''' v[i] = \partial_i r(p) '''
        v = 0
        for j in range(y.size):
            v += y[j] * base_grad(j, p)
        return v

    ##################################################
    # mesh functions
    @jit(**__pparams)
    def int_mesh(v, y, dof_map):
        ''' v[i] = \int_i0^i1 r(t) dt '''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_int(y, dof_map[i, 0], dof_map[i, 0] + dof_map[i, 1])

    @jit(**__pparams)
    def av_mesh(v, y, dof_map):
        ''' v[i] = \int_i0^i1 r(t) dt / (i1-i0)'''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_int(y, dof_map[i, 0], dof_map[i, 0] + dof_map[i, 1]) / dof_map[i, 1]

    @jit(**__pparams)
    def eval_mesh(v, y, dof_map):
        ''' v[i] = r(.5*(i0+i1)) '''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_mid(y, dof_map[i, 0], dof_map[i, 0] + dof_map[i, 1])

    @jit(**__pparams)
    def grad_mesh(v, y, dof_map):
        ''' v[i] = dr(.5*(i0+i1)) '''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_grad(y, dof_map[i, 0] + .5 * dof_map[i, 1])

    @jit(**__pparams)
    def max_abs(v, y, dof_map, s):
        '''
        if |d^2r| < s then estimate |r|_\infty on each mesh-point
        v[:,0] is discrete maximum, v[:,1] is continuous maximum
        '''
        for i in prange(dof_map.shape[0]):
            p0, p1 = dof_map[i, 0], dof_map[i, 0] + dof_map[i, 1]
            h = .5 * dof_map[i, 1]
            pmid = p0 + h
            v[i] = abs(eval_mid(y, p0, p1)) + h * abs(eval_grad(y, pmid)) + .5 * h * h * s

    return {'eval_int':eval_int, 'eval_mid':eval_mid, 'eval_grad':eval_grad,
            'int_mesh':int_mesh, 'av_mesh':av_mesh, 'eval_mesh':eval_mesh,
            'grad_mesh':grad_mesh, 'max_abs':max_abs}


def ker2func2(base_eval, base_grad, base_int):
    gradx, grady = base_grad

    ##################################################
    # r(t) = sum y_j ker_j(t)
    ##################################################
    # Pixel-wise functions
    @jit(**__params)
    def eval_int(y, px0, py0, px1, py1):
        ''' v = int_{p0}^{p1} r(t) dt '''
        v = 0
        for j in range(y.size):
            v += y[j] * base_int(j, px0, py0, px1, py1)
        return v

    @jit(**__params)
    def eval_mid(y, px0, py0, px1, py1):
        ''' v = r((p0+p1)/2) '''
        px, py = .5 * (px0 + px1), .5 * (py0 + py1)
        v = 0
        for j in range(y.size):
            v += y[j] * base_eval(j, px, py)
        return v

    @jit(**__params)
    def eval_gradx(y, px, py):
        ''' v[i] >= |\partial_i r(p)| '''
        v = 0
        for j in range(y.size):
            v += y[j] * gradx(j, px, py)
        return v

    @jit(**__params)
    def eval_grady(y, px, py):
        ''' v[i] >= |\partial_i r(p)| '''
        v = 0
        for j in range(y.size):
            v += y[j] * grady(j, px, py)
        return v

    ##################################################
    # mesh functions
    @jit(**__pparams)
    def int_mesh(v, y, dof_map):
        ''' v[i] = \int_i0^i1 r(t) dt '''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_int(y, dof_map[i, 0], dof_map[i, 1],
                        dof_map[i, 0] + dof_map[i, 2], dof_map[i, 1] + dof_map[i, 2])

    @jit(**__pparams)
    def av_mesh(v, y, dof_map):
        ''' v[i] = \int_i0^i1 r(t) dt / (i1-i0)'''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_int(y, dof_map[i, 0], dof_map[i, 1],
                            dof_map[i, 0] + dof_map[i, 2],
                            dof_map[i, 1] + dof_map[i, 2]) / dof_map[i, 2]

    @jit(**__pparams)
    def eval_mesh(v, y, dof_map):
        ''' v[i] = r(.5*(i0+i1)) '''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_mid(y, dof_map[i, 0], dof_map[i, 1],
                            dof_map[i, 0] + dof_map[i, 2],
                            dof_map[i, 1] + dof_map[i, 2])

    @jit(**__pparams)
    def grad_mesh(v, y, dof_map):
        ''' v[i] = dr(.5*(i0+i1)) '''
        for i in prange(dof_map.shape[0]):
            v[i, 0] = eval_gradx(y, dof_map[i, 0] + .5 * dof_map[i, 2],
                                dof_map[i, 1] + .5 * dof_map[i, 2])
            v[i, 1] = eval_grady(y, dof_map[i, 0] + .5 * dof_map[i, 2],
                                dof_map[i, 1] + .5 * dof_map[i, 2])

    @jit(**__pparams)
    def max_abs(v, y, dof_map, s):
        '''
        if |d^2r| < s then estimate |r|_\infty on each mesh-point
        v[:,0] is discrete maximum, v[:,1] is continuous maximum
        '''
        for i in prange(dof_map.shape[0]):
            px0, px1 = dof_map[i, 0], dof_map[i, 0] + dof_map[i, 2]
            py0, py1 = dof_map[i, 1], dof_map[i, 1] + dof_map[i, 2]
            h = c_sqrt(.5) * dof_map[i, 2]
            pmidx, pmidy = .5 * (px0 + px1), .5 * (py0 + py1)
            dx, dy = eval_gradx(y, pmidx, pmidy), eval_grady(y, pmidx, pmidy)
            v[i] = abs(eval_mid(y, px0, py0, px1, py1)) + h * c_sqrt(dx * dx + dy * dy) + .5 * h * h * s

    return {'eval_int':eval_int, 'eval_mid':eval_mid, 'eval_grad':[eval_gradx, eval_grady],
            'int_mesh':int_mesh, 'av_mesh':av_mesh, 'eval_mesh':eval_mesh,
            'grad_mesh':grad_mesh, 'max_abs':max_abs}


def ker2meshfunc2(base_eval, base_grad, base_int, dx, r):
    gradx, grady = base_grad

    ##################################################
    # r(t) = sum y_j ker_j(t)
    ##################################################
    # Pixel-wise functions
    @jit(**__params)
    def eval_int(y, px0, py0, px1, py1):
        ''' v = int_{p0}^{p1} r(t) dt '''
        jm0, jM0 = int((px0 - r) / dx), int((px1 + r) / dx) + 1
        jm0, jM0 = max(jm0, 0), min(jM0, y.shape[0])
        jm1, jM1 = int((py0 - r) / dx), int((py1 + r) / dx) + 1
        jm1, jM1 = max(jm1, 0), min(jM1, y.shape[1])

        v = 0
        for j0 in range(jm0, jM0):
            for j1 in range(jm1, jM1):
                v += y[j0, j1] * base_int(j0, j1, px0, py0, px1, py1)
        return v

    @jit(**__params)
    def eval_mid(y, px0, py0, px1, py1):
        ''' v = r((p0+p1)/2) '''
        jm0, jM0 = int((px0 - r) / dx), int((px1 + r) / dx) + 1
        jm0, jM0 = max(jm0, 0), min(jM0, y.shape[0])
        jm1, jM1 = int((py0 - r) / dx), int((py1 + r) / dx) + 1
        jm1, jM1 = max(jm1, 0), min(jM1, y.shape[1])

        px, py = .5 * (px0 + px1), .5 * (py0 + py1)
        v = 0
        for j0 in range(jm0, jM0):
            for j1 in range(jm1, jM1):
                v += y[j0, j1] * base_eval(j0, j1, px, py)
        return v

    @jit(**__params)
    def eval_gradx(y, px, py):
        ''' v[i] >= |\partial_i r(p)| '''
        jm0, jM0 = int((px - r) / dx), int((px + r) / dx) + 1
        jm0, jM0 = max(jm0, 0), min(jM0, y.shape[0])
        jm1, jM1 = int((py - r) / dx), int((py + r) / dx) + 1
        jm1, jM1 = max(jm1, 0), min(jM1, y.shape[1])

        v = 0
        for j0 in range(jm0, jM0):
            for j1 in range(jm1, jM1):
                v += y[j0, j1] * gradx(j0, j1, px, py)
        return v

    @jit(**__params)
    def eval_grady(y, px, py):
        ''' v[i] >= |\partial_i r(p)| '''
        jm0, jM0 = int((px - r) / dx), int((px + r) / dx) + 1
        jm0, jM0 = max(jm0, 0), min(jM0, y.shape[0])
        jm1, jM1 = int((py - r) / dx), int((py + r) / dx) + 1
        jm1, jM1 = max(jm1, 0), min(jM1, y.shape[1])

        v = 0
        for j0 in range(jm0, jM0):
            for j1 in range(jm1, jM1):
                v += y[j0, j1] * grady(j0, j1, px, py)
        return v

    ##################################################
    # mesh functions
    @jit(**__pparams)
    def int_mesh(v, y, dof_map):
        ''' v[i] = \int_i0^i1 r(t) dt '''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_int(y, dof_map[i, 0], dof_map[i, 1],
                        dof_map[i, 0] + dof_map[i, 2], dof_map[i, 1] + dof_map[i, 2])

    @jit(**__pparams)
    def av_mesh(v, y, dof_map):
        ''' v[i] = \int_i0^i1 r(t) dt / (i1-i0)'''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_int(y, dof_map[i, 0], dof_map[i, 1],
                            dof_map[i, 0] + dof_map[i, 2],
                            dof_map[i, 1] + dof_map[i, 2]) / dof_map[i, 2]

    @jit(**__pparams)
    def eval_mesh(v, y, dof_map):
        ''' v[i] = r(.5*(i0+i1)) '''
        for i in prange(dof_map.shape[0]):
            v[i] = eval_mid(y, dof_map[i, 0], dof_map[i, 1],
                            dof_map[i, 0] + dof_map[i, 2],
                            dof_map[i, 1] + dof_map[i, 2])

    @jit(**__pparams)
    def grad_mesh(v, y, dof_map):
        ''' v[i] = dr(.5*(i0+i1)) '''
        for i in prange(dof_map.shape[0]):
            v[i, 0] = eval_gradx(y, dof_map[i, 0] + .5 * dof_map[i, 2],
                                dof_map[i, 1] + .5 * dof_map[i, 2])
            v[i, 1] = eval_grady(y, dof_map[i, 0] + .5 * dof_map[i, 2],
                                dof_map[i, 1] + .5 * dof_map[i, 2])

    @jit(**__pparams)
    def max_abs(v, y, dof_map, s):
        '''
        if |d^2r| < s then estimate |r|_\infty on each mesh-point
        v[:,0] is discrete maximum, v[:,1] is continuous maximum
        '''
        for i in prange(dof_map.shape[0]):
            px0, px1 = dof_map[i, 0], dof_map[i, 0] + dof_map[i, 2]
            py0, py1 = dof_map[i, 1], dof_map[i, 1] + dof_map[i, 2]
            pmidx, pmidy = .5 * (px0 + px1), .5 * (py0 + py1)
            h = c_sqrt(.5) * dof_map[i, 2]

            v0, vx, vy = 0, 0, 0
            jm0, jM0 = int((px0 - r) / dx), int((px1 + r) / dx) + 1
            jm0, jM0 = max(jm0, 0), min(jM0, y.shape[0])
            jm1, jM1 = int((py0 - r) / dx), int((py1 + r) / dx) + 1
            jm1, jM1 = max(jm1, 0), min(jM1, y.shape[1])
            for j0 in range(jm0, jM0):
                for j1 in range(jm1, jM1):
                    v0 += y[j0, j1] * base_eval(j0, j1, pmidx, pmidy)
                    vx += y[j0, j1] * gradx(j0, j1, pmidx, pmidy)
                    vy += y[j0, j1] * grady(j0, j1, pmidx, pmidy)

            v[i] = abs(v0) + h * c_sqrt(vx * vx + vy * vy) + .5 * h * h * s

    return {'eval_int':eval_int, 'eval_mid':eval_mid, 'eval_grad':[eval_gradx, eval_grady],
            'int_mesh':int_mesh, 'av_mesh':av_mesh, 'eval_mesh':eval_mesh,
            'grad_mesh':grad_mesh, 'max_abs':max_abs}


def ker2mat1(base_eval, base_grad, base_int, IP):

    ##################################################
    # foward projection:
    #     A(x,mesh)_j = integral(ker_j * x, mesh)
    ##################################################
    @jit(**__pparams)
    def fwrd(V, arr, dof_map):
        for j in prange(V.size):
            v = 0
            for i in range(dof_map.shape[0]):
                v += arr[i] * base_int(j, dof_map[i, 0], dof_map[i, 0] + dof_map[i, 1])
            V[j] = v

    ##################################################
    # backward projection:
    #     r = A.T(y) = sum y_j ker_j is a function
    # These methods return mesh functions
    ##################################################
    # v = int_{p0}^{p1} r(t) dt / (p1-p0)
    @jit(**__pparams)
    def bwrd(V, y, dof_map):
        for i in prange(dof_map.shape[0]):
            p0 = dof_map[i, 0]
            p1 = p0 + dof_map[i, 1]
            scale = 1 / dof_map[i, 1]
            v = 0
            for j in range(y.size):
                v += y[j] * base_int(j, p0, p1) * scale
            V[i] = v

    ##################################################
    # utilities
    # The matrix such that (Ax)_j = \int x(t) ker_j(t) dt
    @jit(**__pparams)
    def to_matrix(A, dof_map, flag):
        if flag == 1:  # evaluated at midpoints
            for j in prange(A.shape[0]):
                for i in range(A.shape[1]):
                    A[j, i] = base_eval(j, dof_map[i, 0] + .5 * dof_map[i, 1])
        elif flag == 2:  # grad evaluated at midpoints
            for j in prange(A.shape[0]):
                for i in range(A.shape[1]):
                    A[j, i] = base_grad(j, dof_map[i, 0] + .5 * dof_map[i, 1])
        else:  # exact discrete forward matrix
            for j in prange(A.shape[0]):
                for i in range(A.shape[1]):
                    A[j, i] = base_int(j, dof_map[i, 0], dof_map[i, 0] + dof_map[i, 1])

    # the matrix of A^T assuming it is a map A^Ty = \sum y_j ker_j projected onto a mesh
    @jit(**__pparams)
    def to_matrixT(A, dof_map):
        for i in prange(A.shape[0]):
            p0, p1 = dof_map[i, 0], dof_map[i, 0] + dof_map[i, 1]
            scale = 1 / dof_map[i, 1]
            for j in range(A.shape[1]):
                A[i, j] = base_int(j, p0, p1) * scale

    # the matrix of AA^T assuming no mesh
    @jit(**__pparams)
    def to_matrix2(AA):
        for i in prange(AA.shape[0]):
            for j in range(AA.shape[1]):
                AA[i, j] = IP(i, j)

    return {'fwrd':fwrd, 'bwrd':bwrd, 'to_matrix':to_matrix,
            'to_matrixT':to_matrixT, 'to_matrix2':to_matrix2, }


def ker2mat2(base_eval, base_grad, base_int, IP):
#     gradx, grady = base_grad

    ##################################################
    # foward projection:
    #     A(x,mesh)_j = integral(ker_j * x, mesh)
    ##################################################
    @jit(**__pparams)
    def fwrd(V, arr, dof_map):
        for j in prange(V.size):
            v = 0
            for i in range(dof_map.shape[0]):
                v += arr[i] * base_int(j, dof_map[i, 0], dof_map[i, 1],
                                       dof_map[i, 0] + dof_map[i, 2],
                                       dof_map[i, 1] + dof_map[i, 2])
            V[j] = v

    ##################################################
    # backward projection:
    #     r = A.T(y) = sum y_j ker_j is a function
    # These methods return mesh functions
    ##################################################
    # v = int_{p0}^{p1} r(t) dt / (p1-p0)
    @jit(**__pparams)
    def bwrd(V, y, dof_map):
        for i in prange(dof_map.shape[0]):
            h = dof_map[i, 2]
            px0, py0 = dof_map[i, 0], dof_map[i, 1]
            px1, py1 = px0 + h, py0 + h
            scale = 1 / (h * h)
            v = 0
            for j in range(y.size):
                v += y[j] * base_int(j, px0, py0, px1, py1) * scale
            V[i] = v

    ##################################################
    # utilities
    # The matrix such that (Ax)_j = \int x(t) ker_j(t) dt
    @jit(**__pparams)
    def to_matrix(A, dof_map, flag):
        if flag == 1:  # evaluated at midpoints
            for j in prange(A.shape[0]):
                for i in range(A.shape[1]):
                    h = .5 * dof_map[i, 2]
                    A[j, i] = base_eval(j, dof_map[i, 0] + h, dof_map[i, 1] + h)
#         elif flag == 2:  # grad evaluated at midpoints
#             for j in prange(A.shape[0]):
#                 for i in range(A.shape[1]):
#                     h = .5 * dof_map[i, 2]
#                     A[j, i, 0] = gradx(j, dof_map[i, 0] + h, dof_map[i, 1] + h)
#                     A[j, i, 1] = grady(j, dof_map[i, 0] + h, dof_map[i, 1] + h)
        else:  # exact discrete forward matrix
            for j in prange(A.shape[0]):
                for i in range(A.shape[1]):
                    A[j, i] = base_int(j, dof_map[i, 0], dof_map[i, 1],
                                       dof_map[i, 0] + dof_map[i, 2],
                                       dof_map[i, 1] + dof_map[i, 2])

    # the matrix of A^T assuming it is a map A^Ty = \sum y_j ker_j projected onto a mesh
    @jit(**__pparams)
    def to_matrixT(A, dof_map):
        for i in prange(A.shape[0]):
            px0, py0 = dof_map[i, 0], dof_map[i, 1]
            px1, py1 = px0 + dof_map[i, 2], py0 + dof_map[i, 2]
            scale = 1 / (dof_map[i, 2] * dof_map[i, 2])
            for j in range(A.shape[1]):
                A[i, j] = base_int(j, px0, py0, px1, py1) * scale

    # the matrix of AA^T assuming no mesh
    @jit(**__pparams)
    def to_matrix2(AA):
        for i in prange(AA.shape[0]):
            for j in range(AA.shape[1]):
                AA[i, j] = IP(i, j)

    return {'fwrd':fwrd, 'bwrd':bwrd, 'to_matrix':to_matrix,
            'to_matrixT':to_matrixT, 'to_matrix2':to_matrix2, }


def ker2meshmat2(base_eval, base_grad, base_int, IP, dx, r):

    ##################################################
    # foward projection:
    #     A(x,mesh)_j = integral(ker_j * x, mesh)
    ##################################################
    @jit(**__pparams)
    def fwrd(V, arr, dof_map, coarse_map, stride):
        for J0 in prange(V.shape[0] // stride):
            v = 0.
            for j0 in range(stride * J0, min(stride * (J0 + 1), V.shape[0])):
                for j1 in range(V.shape[1]):
                    v = 0.
                    cm0, cm1 = int(j0 * dx * coarse_map.shape[0]), int(j1 * dx * coarse_map.shape[1])
                    cm0, cm1 = min(cm0, coarse_map.shape[0] - 1), min(cm1, coarse_map.shape[1] - 1)
                    im, iM = coarse_map[cm0, cm1, 0], coarse_map[cm0, cm1, 1]

                    for i in range(im, iM):
                        v += arr[i] * base_int(j0, j1,
                                               dof_map[i, 0], dof_map[i, 1],
                                               dof_map[i, 0] + dof_map[i, 2],
                                               dof_map[i, 1] + dof_map[i, 2])

                    V[j0, j1] = v

    ##################################################
    # backward projection:
    #     r = A.T(y) = sum y_j ker_j is a function
    # These methods return mesh functions
    ##################################################
    # v = int_{p0}^{p1} r(t) dt / (p1-p0)
    @jit(**__pparams)
    def bwrd(V, y, dof_map):
        for i in prange(dof_map.shape[0]):
            h = dof_map[i, 2]
            px0, py0 = dof_map[i, 0], dof_map[i, 1]
            px1, py1 = px0 + h, py0 + h
            scale = 1 / (h * h)

            jm0, jM0 = int((px0 - r) / dx), int((px1 + r) / dx) + 1
            jm0, jM0 = max(jm0, 0), min(jM0, y.shape[0])
            jm1, jM1 = int((py0 - r) / dx), int((py1 + r) / dx) + 1
            jm1, jM1 = max(jm1, 0), min(jM1, y.shape[1])

            v = 0
            for j0 in range(jm0, jM0):
                for j1 in range(jm1, jM1):
                    v += y[j0, j1] * base_int(j0, j1, px0, py0, px1, py1) * scale
            V[i] = v

    ##################################################
    # utilities
    # The matrix such that (Ax)_j = \int x(t) ker_j(t) dt
    @jit(**__pparams)
    def to_matrix(A, dof_map, coarse_map, stride, flag):
        if flag == 1:  # evaluated at midpoints
            for J0 in prange(A.shape[0] // stride):
                for j0 in range(stride * J0, min(stride * (J0 + 1), A.shape[0])):
                    for j1 in range(A.shape[1]):
                        cm0, cm1 = int(j0 * dx * coarse_map.shape[0]), int(j1 * dx * coarse_map.shape[1])
                        cm0, cm1 = min(cm0, coarse_map.shape[0] - 1), min(cm1, coarse_map.shape[1] - 1)
                        jm, jM = coarse_map[cm0, cm1, 0], coarse_map[cm0, cm1, 1]
                        for i in range(jm, jM):
                            h = .5 * dof_map[i, 2]
                            A[j0, j1, i] = base_eval(j0, j1, dof_map[i, 0] + h, dof_map[i, 1] + h)
        else:  # exact discrete forward matrix
            for J0 in prange(A.shape[0] // stride):
                for j0 in range(stride * J0, min(stride * (J0 + 1), A.shape[0])):
                    for j1 in range(A.shape[1]):
                        cm0, cm1 = int(j0 * dx * coarse_map.shape[0]), int(j1 * dx * coarse_map.shape[1])
                        cm0, cm1 = min(cm0, coarse_map.shape[0] - 1), min(cm1, coarse_map.shape[1] - 1)
                        jm, jM = coarse_map[cm0, cm1, 0], coarse_map[cm0, cm1, 1]

                        for i in range(jm, jM):
                            A[j0, j1, i] = base_int(j0, j1,
                                                    dof_map[i, 0], dof_map[i, 1],
                                                   dof_map[i, 0] + dof_map[i, 2],
                                                   dof_map[i, 1] + dof_map[i, 2])

    # the matrix of A^T assuming it is a map A^Ty = \sum y_j ker_j projected onto a mesh
    @jit(**__pparams)
    def to_matrixT(A, dof_map):
        for i in prange(A.shape[0]):
            px0, py0 = dof_map[i, 0], dof_map[i, 1]
            px1, py1 = px0 + dof_map[i, 2], py0 + dof_map[i, 2]
            scale = 1 / (dof_map[i, 2] * dof_map[i, 2])

            jm0, jM0 = int((px0 - r) / dx), int((px1 + r) / dx) + 1
            jm0, jM0 = max(jm0, 0), min(jM0, A.shape[1])
            jm1, jM1 = int((py0 - r) / dx), int((py1 + r) / dx) + 1
            jm1, jM1 = max(jm1, 0), min(jM1, A.shape[2])

            for j0 in range(jm0, jM0):
                for j1 in range(jm1, jM1):
                    A[i, j0, j1] = base_int(j0, j1, px0, py0, px1, py1) * scale

    @jit(**__params)
    def to_sparse_matrix(data, scaled, indcs, ptr, dof_map, data_sz):
        II = 0  # current index of matrix
        for I in range(dof_map.shape[0]):
            # Initialise parameters constant for each new row
            h = dof_map[I, 2]
            scale, N = 1 / (h * h), int((h + r) / dx) + 1

            px0, py0 = dof_map[I, 0], dof_map[I, 1]
            jx, jy = int(px0 / dx), int(py0 / dx)
            jm0, jM0 = max(0, jx - N), min(data_sz, jx + N + 1)
            jm1, jM1 = max(0, jy - N), min(data_sz, jy + N + 1)

            ptr[I] = II  # row I starts at index II
            for j0 in range(jm0, jM0):
                for j1 in range(jm1, jM1):  # create new columns
                    data[II] = base_int(j0, j1, px0, py0, px0 + h, py0 + h)
                    scaled[II] = data[II] * scale
                    indcs[II] = j0 * data_sz + j1
                    II += 1

        ptr[dof_map.shape[0]] = II  # end of data arrays

    @jit(**__params)
    def update_sparse_matrix(o_data, o_indcs, o_ptr, refine,
                         n_data, n_scale, n_indcs, n_ptr, dof_map, data_sz):
        I, II = 0, 0  # current row/index of new matrix
        for i in range(refine.shape[0]):  # current row of old matrix

            if refine[i] == 0:  # copy old row to new
                # old scale equals new scale
                scale = 1 / (dof_map[I, 2] * dof_map[I, 2])
                n_ptr[I] = II  # row I starts at index II
                for ii in range(o_ptr[i], o_ptr[i + 1]):
                    n_data[II] = o_data[ii]
                    n_scale[II] = o_data[ii] * scale
                    n_indcs[II] = o_indcs[ii]
                    II += 1
                I += 1  # move to new row in new matrix

            elif refine[i] > 0:  # create new rows
                # Initialise parameters constant for each new row
                h = dof_map[I, 2]
                scale, N = 1 / (h * h), int((h + r) / dx) + 1

                for _ in range(4):  # one pixel splits into 4
                    px0, py0 = dof_map[I, 0], dof_map[I, 1]
                    jx, jy = int(px0 / dx), int(py0 / dx)
                    jm0, jM0 = max(0, jx - N), min(data_sz, jx + N + 1)
                    jm1, jM1 = max(0, jy - N), min(data_sz, jy + N + 1)

                    n_ptr[I] = II  # row I starts at index II
                    for j0 in range(jm0, jM0):
                        for j1 in range(jm1, jM1):  # create new columns
                            n_data[II] = base_int(j0, j1, px0, py0, px0 + h, py0 + h)
                            n_scale[II] = n_data[II] * scale
                            n_indcs[II] = j0 * data_sz + j1
                            II += 1
                    I += 1  # move to new row in new matrix

        n_ptr[I] = II  # end of data arrays

    @jit(**__pparams)
    def sp_matvec(data, indcs, ptr, x, out):
        for i in prange(ptr.size - 1):
            v = 0
            for j in range(ptr[i], ptr[i + 1]):
                v += x[indcs[j]] * data[j]
            out[i] = v

    # the matrix of AA^T assuming no mesh
    @jit(**__pparams)
    def to_matrix2(AA):
        for i0 in prange(AA.shape[0]):
            for i1 in prange(AA.shape[1]):
                for j0 in range(AA.shape[2]):
                    for j1 in range(AA.shape[3]):
                        AA[i0, i1, j0, j1] = IP(i0, i1, j0, j1)

    return {'fwrd':fwrd, 'bwrd':bwrd, 'to_matrix':to_matrix,
            'to_matrixT':to_matrixT, 'to_matrix2':to_matrix2,
            'to_sparse_matrix':to_sparse_matrix, 'sp_matvec':sp_matvec,
            'update_sparse_matrix':update_sparse_matrix}


def _test_op(A):
    from lasso2 import sparse_mesh, sparse_func, norm
    FS, d = sparse_mesh(5, A.dim), (2.** -5) ** A.dim
    F = lambda x: A * x
    B = lambda y: x.copy(A.T(y, FS))
    from algorithms import Inner
    from numpy import random

    x = sparse_func(0, FS)
    y = F(x)

    def rand(x):
        if hasattr(x, 'x'):
            x.x[...] = random.rand(*x.x.shape)
        else:
            x[...] = random.rand(*x.shape)
        return x

    M, MT = A._to_matrix(FS), A._to_matrixT(FS)
    mat_test = lambda x: (F(x), M.dot(x.ravel()))
    matT_test = lambda y: (B(y).ravel(), MT.dot(y))
    mat_match = [mat_test(rand(x)) for _ in range(10)] + [matT_test(rand(y)) for _ in range(10)]

    adjoint_test = lambda x, y: (Inner(F(x), y), FS.inner(x, B(y)))
    adjoint_match = [adjoint_test(rand(x), rand(y)) for _ in range(10)]

#     print('Discrete matrix match: %.2e' % abs(M - d * MT.T).max())
#     print('Operator = matrix:     %.2e' % max(abs(s[0] - s[1]).max() for s in mat_match))
#     print('<Ax,y>=<x,A^*y>:       %.2e' % max(abs(s[0] - s[1]).max() for s in adjoint_match))
#     print('Operator norm:         %.2e > 0' % (A._norm - norm(M.dot(MT), 2) ** .5))

    assert abs(M - d * MT.T).max() < 1e-10
    assert max(abs(s[0] - s[1]).max() for s in mat_match) < 1e-10
    assert max(abs(s[0] - s[1]).max() for s in adjoint_match) < 1e-10
    assert A._norm > norm(M.dot(MT), 2) ** .5 - 1e-10

