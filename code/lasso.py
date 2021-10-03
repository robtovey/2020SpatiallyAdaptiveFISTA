'''
Created on 26 May 2020

@author: Rob Tovey

This module implements a sparse grid of the domain [0,1]^d up to d=2.
A pixel is defined as a tuple:
    [x1, ..., xd, h]
which corresponds to the cube
    [x1,x1+h]*[...]*[xd,xd+h]

Pixel division is done by insertion, coarsening is not implemented.
A coarse mesh indexing is stored to allow more efficient searching
and plotting etc. within the mesh.

A forward model of Gaussian bluring and tomography is also implemented.
'''
from numpy import (empty, array, zeros, isscalar, log, pi, sqrt, log10,
    log2, arange, concatenate, linspace, prod)
from scipy.sparse import csr_matrix
from numpy.linalg import norm
from matplotlib import pyplot as plt
from sparse_bin import jit, FTYPE, _blur_coarse
from sparse_ops import func, kernelMap, op, ker2meshmat2, ker2meshfunc2, sparse_matvec
from math import (exp as c_exp, erf as c_erf, sin as c_sin, cos as c_cos, sqrt as c_sqrt, floor)
from algorithms import (stopping_criterion, faster_FISTA, _makeVid,
    FB, shrink, shrink_vec, FISTA)
from adaptive_spaces import sparse_function as sparse_func, sparse_FS as sparse_mesh, adaptive_func as Array


class const_plus_func(Array):

    def __init__(self, const, func):
        Array.__init__(self, [const, func], isFEM=False)
        self.c, self.f = const, func

    def asarray(self): return (self.c, self.f.asarray())

    def __L(self, other):
        if isscalar(other):
            return (other, other)
        else:
            assert len(other) == 2
            return other

    @property
    def size(self): return 1 + self.f.size

    @property
    def c(self): return self.x[0]

    @property
    def f(self): return self.x[1]

    @property
    def shape(self): return (self.size,)

    def update(self, x, FS=None):
        assert len(x) == 2
        self.x[0] = x[0]
        self.x[1].update(x[1], FS)

    def __add__(self, other): return self.copy([self.x[i] + o for i, o in enumerate(self.__L(other))])

    def __sub__(self, other): return self.copy([self.x[i] - o for i, o in enumerate(self.__L(other))])

    def __mul__(self, other):
        assert isscalar(other)
        return self.copy((self.x[0] * other, self.x[1] * other))

    def __truediv__(self, other): return self * (1 / other)

    def __rtruediv__(self, other): raise NotImplementedError

    def __iadd__(self, other):
        other = self.__L(other)
        for i in range(2):
            self.x[i] += other[i]

    def __isub__(self, other):
        other = self.__L(other)
        for i in range(2):
            self.x[i] -= other[i]

    def __imul__(self, other):
        assert isscalar(other)
        self.x *= other

    def __itruediv__(self, other): self *= (1 / other)

    def __neg__(self): return self.copy((-self.x[0], -self.x[1]))

    __radd__ = __add__
#     __rsub__ = __sub__
    __rmul__ = __mul__

    def discretise(self, level): return self.f.discretise(level) + self.c

    def from_discrete(self, arr): return self.copy(0, self.f.from_discrete(arr))

    def plot(self, level, *args, ax=None, **kwargs):
        kwargs['background'] = kwargs.get('background', 0) + self.c
        self.f.plot(level, *args, ax=ax, **kwargs)

    def copy(self, like=None):
        if like is None:
            c, f = self.c, self.f.copy()
        else:
            assert len(like) == 2
            c, f = like[0], self.f.copy(like[1])
        return const_plus_func(c, f)


class kernelMap1D(kernelMap):

    def __init__(self, ker='Gaussian', x=None, Norm=-1, **kwargs):
        '''
        Represents a linear map such that:
            (A\mu)_i = \int k(x,p_i)mu(x)dx
        whenever mu is a sparse_func1D element.
        '''
        assert x is not None
        X = array(x, dtype=FTYPE).copy().reshape(-1)

        if ker == 'Gaussian':
            sigma = kwargs['sigma']
            tmp0, tmp1 = 1 / sqrt(2 * pi * sigma ** 2), 1 / sigma ** 2
            p = array([tmp0, tmp1, sqrt(tmp1 / 2), tmp0 * sqrt(pi / (2 * tmp1)), tmp0 ** 2 * sqrt(pi / tmp1)], dtype=FTYPE)

            @jit(nopython=True, fastmath=True)
            def base_eval(i, x): return p[0] * c_exp(-.5 * p[1] * (x - X[i]) ** 2)

            @jit(nopython=True, fastmath=True)
            def base_grad(i, x): return -p[1] * (x - X[i]) * base_eval(i, x)

            @jit(nopython=True, fastmath=True)
            def base_int(i, x0, x1): return p[3] * (c_erf(p[2] * (x1 - X[i])) - c_erf(p[2] * (x0 - X[i])))

            @jit(nopython=True, fastmath=True)
            def IP(i, j):
                v = p[4]
                if i != j:
                    v *= c_exp(-.25 * p[1] * (X[i] - X[j]) ** 2)
                return v

            assert abs(X[1:] - X[:-1]).ptp() < 1e-15
            dx = abs(X[1] - X[0])
            C = ((1, sqrt(1 + sqrt(pi) * sigma / dx), (1 + sqrt(2 * pi) * sigma / dx)),
                 (c_exp(-.5), sqrt(sqrt(2 / c_exp(1)) + sqrt(pi) / 2 * sigma / dx), 2 / c_exp(.5) + 2 * sigma / dx),
                 (2 / c_exp(.5), sqrt(4 / c_exp(1) + 11 * sqrt(pi) / 4 * sigma / dx), 4 / c_exp(.5) + sqrt(8 * pi) * sigma / dx))

            def mat_norm(k):
                ''' |A|_{L^2->C^k}'''
                scale = p[0] / (1 if k == 0 else sigma ** k)
                return scale * min(C[k][0] * sqrt(X.shape[0]), C[k][1], C[k][2])

            def func_norm(y, k):
                ''' |Ay|_{C^k}'''
                scale = p[0] / (1 if k == 0 else sigma ** k)
                return scale * min(C[k][0] * abs(y).sum(),
                                   C[k][1] * norm(y.reshape(-1)),
                                   C[k][2] * abs(y).max())

            __params = (sigma, p, C)

        elif ker == 'Fourier':
            __params = tuple()

            @jit(nopython=True, fastmath=True)
            def base_eval(i, x): return c_cos(x * X[i])

            @jit(nopython=True, fastmath=True)
            def base_grad(i, x): return -X[i] * c_sin(x * X[i])

            @jit(nopython=True, fastmath=True)
            def base_int(i, x0, x1): return (c_sin(x1 * X[i]) - c_sin(x0 * X[i])) / X[i]

            @jit(nopython=True, fastmath=True)
            def IP(i, j):
                x = X[i] + X[j]
                v = c_sin(x) / x
                if i != j:
                    x = X[i] - X[j]
                    v += c_sin(x) / x
                else:
                    v += 1
                return .5 * v

            def mat_norm(k):
                ''' |A|_{L^2->C^k}'''
                if k == 0:
                    v = X.size
                if k == 1:
                    v = (X ** 2).sum()
                elif k == 2:
                    v = (X ** 4).sum()
                return sqrt(v)

            def func_norm(y, k):
                ''' |Ay|_{C^k}'''
                if k == 0:
                    return abs(y).sum()
                elif k == 1:
                    return ((y * X) ** 2).sum() ** .5
                elif k == 2:
                    return (y * X ** 2).sum()

        kernelMap.__init__(self, 1, X, base_eval, base_grad, base_int, IP, mat_norm, func_norm, Norm=Norm)
        self.ker, self.__params = ker, __params


class kernelMap2D(kernelMap):

    def __init__(self, ker='Gaussian', x=None, Norm=-1, **kwargs):
        '''
        Represents a linear map such that:
            (A\mu)_i = \int k(x,p_i)mu(x)dx
        whenever mu is a sparse_func2D element.
        '''
        assert x is not None
        # x,y provided X[ij] = (x[i],y[j])
        # x is a list of 1D points, X[ij] = (x[i],x[j])
        # x is a list of vectors, X[ij] = (x[0][i],x[1][j])
        if 'y' in kwargs:
            y = kwargs.pop('y')
        elif isscalar(x[0]):
            y = x
        else:
            x, y = x[0], x[1]
        X = zeros((x.size, y.size, 2), dtype=FTYPE)
        X[..., 0] += array(x).reshape(-1, 1); X[..., 1] += array(y).reshape(1, -1)
        X.shape = -1, 2

        if ker == 'Gaussian':
            '''
            f(x,y) = t0*exp(-.5*(x^2+y^2)t1)
            df(x,y) = -t1*[x;y]*f(x,y)
            \int_x0^x1\int_y0^y1 f = t0*[sqrt(pi/(2t1)) erf(sqrt(t1/2)*x)]_x0^x1*[sqrt(pi/(2t1)) erf(sqrt(t1/2)*y)]_y0^y1
                                   = t0*pi/(2*t1) * [erf(sqrt(t1/2)*x)]_x0^x1 * [erf(sqrt(t1/2)*y)]_y0^y1
            \int f(x,y)f(x-X,y-Y) = t0^2*[sqrt(pi/t1)exp(-X^2*t1/4)]*[sqrt(pi/t1)exp(-Y^2*t1/4)]
                                  = t0^2*pi/t1 * exp(-t1/4*(X^2+Y^2))
            '''
            sigma = kwargs['sigma']
            t0, t1 = 1 / (2 * pi * sigma ** 2), 1 / sigma ** 2
            p = array([t0, t1, sqrt(t1 / 2), t0 * (pi / (2 * t1)), t0 ** 2 * (pi / t1)], dtype=FTYPE)
            # x^2/sigma^2 > 49 => exp(-x^2/2sigma^2) < 1e-10
            r, r2 = 7 * sigma, 49 * sigma ** 2

            @jit(nopython=True, fastmath=True)
            def EXP(t):
                if t < r2:
                    return p[0] * c_exp(-.5 * p[1] * t)
                else:
                    return 0

            @jit(nopython=True, fastmath=True)
            def ERF(t):
                if abs(t) < r:
                    return c_erf(p[2] * t)
                elif t > 0:
                    return 1
                else:
                    return -1

            @jit(nopython=True, fastmath=True)
            def base_eval(i, x, y): return EXP((x - X[i, 0]) * (x - X[i, 0]) + (y - X[i, 1]) * (y - X[i, 1]))

            @jit(nopython=True, fastmath=True)
            def gradx(i, x, y): return -p[1] * (x - X[i, 0]) * EXP((x - X[i, 0]) * (x - X[i, 0]) + (y - X[i, 1]) * (y - X[i, 1]))

            @jit(nopython=True, fastmath=True)
            def grady(i, x, y): return -p[1] * (y - X[i, 1]) * EXP((x - X[i, 0]) * (x - X[i, 0]) + (y - X[i, 1]) * (y - X[i, 1]))

            @jit(nopython=True, fastmath=True)
            def base_int(i, x0, y0, x1, y1):
                v = ERF(x1 - X[i, 0]) - ERF(x0 - X[i, 0])
                if v == 0:
                    return 0
                v *= ERF(y1 - X[i, 1]) - ERF(y0 - X[i, 1])
                return p[3] * v

            @jit(nopython=True, fastmath=True)
            def IP(i, j):
                v = p[4]
                if i != j:
                    v *= c_exp(-.25 * p[1] * ((X[i, 0] - X[j, 0]) ** 2 + (X[i, 1] - X[j, 1]) ** 2))
                return v

#             assert abs(X[1:] - X[:-1]).ptp() < 1e-15
            c, R = sigma / abs(x[1] - x[0]), int(sigma / abs(x[1] - x[0])) + 1
            C = ((1, sqrt(1 + sqrt(pi) * c) ** 2, (1 + sqrt(2 * pi) * c) ** 2),
                 (c_exp(-.5), sqrt(2 / c_exp(1) * R + sqrt(pi) * c + pi * c * c), 2 / c_exp(.5) * R + 4 * c + sqrt(2 * pi ** 3) * c * c),
                 (2 / c_exp(.5), sqrt(8 / c_exp(1) * R + 5.5 * sqrt(pi) * c + 5 * pi * c * c), 4 / c_exp(.5) + 4 * sqrt(2 * pi) * c + 6 * pi * c * c))

            def mat_norm(k):
                ''' |A|_{L^2->C^k}'''
                scale = p[0] / (1 if k == 0 else sigma ** k)
                return scale * min(C[k][0] * sqrt(X.shape[0]), C[k][1], C[k][2])

            def func_norm(y, k):
                ''' |Ay|_{C^k}'''
                scale = p[0] / (1 if k == 0 else sigma ** k)
                return scale * min(C[k][0] * abs(y).sum(),
                                   C[k][1] * norm(y.reshape(-1)),
                                   C[k][2] * abs(y).max())

            __params = (sigma, p, C)

        kernelMap.__init__(self, 2, X, base_eval, (gradx, grady), base_int, IP, mat_norm, func_norm, Norm=Norm)
        self.ker, self.__params = ker, __params


class kernelMapMesh2D(op):

    def __init__(self, ker='Gaussian', x=None, Norm=-1, _multicore_mat=True, **kwargs):
        '''
        Represents a linear map such that:
            (A\mu)_i = \int k(x,p_i)mu(x)dx
        whenever mu is a sparse_func2D element.
        '''
        assert x is not None
        X = array(x, dtype=FTYPE).copy().reshape(-1)

        assert ker == 'Gaussian'
        '''
        f(x,y) = t0*exp(-.5*(x^2+y^2)t1)
        df(x,y) = -t1*[x;y]*f(x,y)
        \int_x0^x1\int_y0^y1 f = t0*[sqrt(pi/(2t1)) erf(sqrt(t1/2)*x)]_x0^x1*[sqrt(pi/(2t1)) erf(sqrt(t1/2)*y)]_y0^y1
                               = t0*pi/(2*t1) * [erf(sqrt(t1/2)*x)]_x0^x1 * [erf(sqrt(t1/2)*y)]_y0^y1
        \int f(x,y)f(x-X,y-Y) = t0^2*[sqrt(pi/t1)exp(-X^2*t1/4)]*[sqrt(pi/t1)exp(-Y^2*t1/4)]
                              = t0^2*pi/t1 * exp(-t1/4*(X^2+Y^2))
        '''
        sigma = kwargs['sigma']
        t0, t1 = 1 / (2 * pi * sigma ** 2), 1 / sigma ** 2
        p = array([t0, t1, sqrt(t1 / 2), t0 * (pi / (2 * t1)), t0 ** 2 * (pi / t1)], dtype=FTYPE)
        # x^2/sigma^2 > 49 => exp(-x^2/2sigma^2) < 1e-10
        dx, r, r2 = abs(X[1] - X[0]), 7 * sigma, 49 * sigma ** 2

        @jit(nopython=True, fastmath=True)
        def EXP(x, y):
            t = x * x + y * y
            if t < r2:
                return p[0] * c_exp(-.5 * p[1] * t)
            else:
                return 0

        @jit(nopython=True, fastmath=True)
        def ERF(t):
            if abs(t) < r:
                return c_erf(p[2] * t)
            elif t > 0:
                return 1
            else:
                return -1

        @jit(nopython=True, fastmath=True)
        def base_eval(i0, i1, x, y): return EXP(x - X[i0], y - X[i1])

        @jit(nopython=True, fastmath=True)
        def gradx(i0, i1, x, y): return -p[1] * (x - X[i0]) * EXP(x - X[i0], y - X[i1])

        @jit(nopython=True, fastmath=True)
        def grady(i0, i1, x, y): return -p[1] * (y - X[i1]) * EXP(x - X[i0], y - X[i1])

        @jit(nopython=True, fastmath=True)
        def base_int(i0, i1, x0, y0, x1, y1):
            v = ERF(x1 - X[i0]) - ERF(x0 - X[i0])
            if v == 0:
                return 0
            v *= ERF(y1 - X[i1]) - ERF(y0 - X[i1])
            return p[3] * v

        @jit(nopython=True, fastmath=True)
        def IP(i0, i1, j0, j1):
            return p[4] * c_exp(-.25 * p[1] * ((X[i0] - X[j0]) * (X[i0] - X[j0])
                                              +(X[i1] - X[j1]) * (X[i1] - X[j1])))

#             assert abs(X[1:] - X[:-1]).ptp() < 1e-15
        c, R = sigma / dx, int(sigma / dx) + 1
        C = ((1, sqrt(1 + sqrt(pi) * c) ** 2, (1 + sqrt(2 * pi) * c) ** 2),
             (c_exp(-.5), sqrt(2 / c_exp(1) * R + sqrt(pi) * c + pi * c * c), 2 / c_exp(.5) * R + 4 * c + sqrt(2 * pi ** 3) * c * c),
             (2 / c_exp(.5), sqrt(8 / c_exp(1) * R + 5.5 * sqrt(pi) * c + 5 * pi * c * c), 4 / c_exp(.5) + 4 * sqrt(2 * pi) * c + 6 * pi * c * c))

        def mat_norm(k):
            ''' |A|_{L^2->C^k}'''
            scale = p[0] / (1 if k == 0 else sigma ** k)
            return scale * min(C[k][0] * X.size, C[k][1], C[k][2])

        def func_norm(y, k):
            ''' |Ay|_{C^k}'''
            scale = p[0] / (1 if k == 0 else sigma ** k)
            return scale * min(C[k][0] * abs(y).sum(),
                               C[k][1] * norm(y.reshape(-1)),
                               C[k][2] * abs(y).max())

        from sparse_bin import prange

        @jit(nopython=True, parallel=True, fastmath=True)
        def max_abs(v, y, dof_map, s):
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
                        scale = y[j0, j1] * base_eval(j0, j1, pmidx, pmidy)
                        v0 += scale
                        vx -= p[1] * (pmidx - X[j0]) * scale
                        vy -= p[1] * (pmidy - X[j1]) * scale
                v[i] = abs(v0) + h * c_sqrt(vx * vx + vy * vy) + .5 * h * h * s

        __params = {'sigma':sigma, 'p':p, 'C':C, 'dx':dx, 'r':r}

        op.__init__(self, self.fwrd, self.bwrd, out_sz=X.shape[0] ** 2, norm=Norm)
        self.x, self.dim = X, 2
        self.ker, self.__params = ker, __params

        self._bin = ker2meshmat2(base_eval, (gradx, grady), base_int, IP, dx, r)
        self._func_bin = ker2meshfunc2(base_eval, (gradx, grady), base_int, dx, r)
        self._bin.update({'base_eval':base_eval, 'base_grad':(gradx, grady), 'base_int':base_int, 'norm':mat_norm})
        self._func_bin.update({'norm':func_norm, 'max_abs':max_abs})

        if Norm < 0:
            tmp = empty((self.x.size,) * 4, dtype=FTYPE)
            self._bin['to_matrix2'](tmp)
            self._norm = norm(tmp.reshape(self.x.size ** 2, -1), 2) ** .5
            print('norm: ', self._norm)
        self.__gnorm = {}
        self._buf = {'DM':None}
        self._multicore_mat = _multicore_mat

    def _blur_coarse(self, coarse_map):
        CM = coarse_map.copy()
        dx = 1 / coarse_map.shape[0]
        r = self.__params['r']
        _blur_coarse(CM, coarse_map, dx, r, int(r / dx) + 1)
        return CM

    def _fat_dof_map(self, dof_map):
        fat_DM = dof_map.copy()
        fat_DM[:,:-1] -= self.__params['r']
        fat_DM[:, -1] += 2 * self.__params['r']
        return fat_DM

    def update(self, FS, refine=None):
        DM = FS.dof_map
        FS = FS.FS if hasattr(FS, 'FS') else FS
        self._buf['DM'] = DM
        self._buf['CM'] = self._blur_coarse(FS.coarse_map)

        if DM[:, -1].max() == 0:
            self._buf['bwrd_mat'] = csr_matrix((DM.shape[0], self.x.size ** 2), dtype=FTYPE)
            self._buf['fwrd_mat'] = csr_matrix((self.x.size ** 2, DM.shape[0]), dtype=FTYPE)
#             self._buf['bwrd_mat'] = self._to_matrixT(FS)
#             self._buf['fwrd_mat'] = self._buf['bwrd_mat'].T * DM[:, -1:].T ** (self.dim)
        else:

            total_sz = ((DM[:, -1] + self.__params['r']) / self.__params['dx']).astype('int32') + 1
            total_sz = ((2 * total_sz + 1) ** 2).sum()
            ptr, indcs = empty(DM.shape[0] + 1, dtype='int32'), empty(total_sz, dtype='int32')
            data, scaled = empty(total_sz, dtype=FTYPE), empty(total_sz, dtype=FTYPE)
            if self._buf.get('data', None) is None or refine is None:
                self._bin['to_sparse_matrix'](data, scaled, indcs, ptr, DM, self.x.size)
            else:
                self._bin['update_sparse_matrix'](self._buf['data'], self._buf['indcs'], self._buf['ptr'], refine,
                                                  data, scaled, indcs, ptr, DM, self.x.size)
            total_sz = ptr.max()
            data, scaled, indcs = data[:total_sz], scaled[:total_sz], indcs[:total_sz]
            self._buf['fwrd_mat'] = csr_matrix((data, indcs, ptr), shape=(DM.shape[0], self.x.size ** 2)).T.tocsr()
            self._buf['fwrd_mat'].sort_indices()
            self._buf['bwrd_mat'] = csr_matrix((scaled, indcs, ptr), shape=(DM.shape[0], self.x.size ** 2))
            self._buf['bwrd_mat'].sort_indices()
            self._buf['data'], self._buf['indcs'], self._buf['ptr'] = data, indcs, ptr

    def fwrd(self, m, FS=None):
        if isinstance(m, Array):
            DM = m.dof_map
            CM = m.coarse_map
            FS = m.FS
            m = m.ravel()
        elif FS is not None:
            DM = FS.dof_map
            CM = FS.coarse_map
        else:
            out = empty((self.x.size,) * 2, dtype=FTYPE)
            # discrete measure
            for j0 in range(out.shape[0]):
                for j1 in range(out.shape[1]):
                    out[j0, j1] = sum(m[i, 0] * self._bin['base_eval'](j0, j1, *m[i, 1:]) for i in range(m.shape[0]))
            return out

        if DM is self._buf['DM']:
            if self._multicore_mat:
                out = empty(self.x.size ** 2, dtype=FTYPE)
                self._bin['sp_matvec'](self._buf['fwrd_mat'].data, self._buf['fwrd_mat'].indices,
                                       self._buf['fwrd_mat'].indptr, m.reshape(-1), out)
                return out
            else:
                return self._buf['fwrd_mat'].dot(m)
        else:
            out = empty((self.x.size,) * 2, dtype=FTYPE)
            CM = self._blur_coarse(CM)
            stride = max(self.x.size // CM.shape[0], 1)
            self._bin['fwrd'](out, m, DM, CM, stride)
        return out.reshape(-1)

    def bwrd(self, s, FS=None):
        if FS is None:
            return func(s.reshape(self.x.size, self.x.size), self._func_bin)
        DM = FS.dof_map
        if DM is self._buf['DM']:
            if self._multicore_mat:
                out = empty(DM.shape[0], dtype=FTYPE)
                self._bin['sp_matvec'](self._buf['bwrd_mat'].data, self._buf['bwrd_mat'].indices,
                                       self._buf['bwrd_mat'].indptr, s.reshape(-1), out)
                return out
            else:
                return self._buf['bwrd_mat'].dot(s)
        else:
            out = empty(DM.shape[0], dtype=FTYPE)
            self._bin['bwrd'](out, s.reshape(self.x.size, -1), DM)
            return out

    def _to_matrix(self, FS, flag=0):
        if flag == 2 and self.dim > 1:
            raise NotImplementedError
        DM = FS.dof_map
        if DM is self._buf['DM']:
            CM = self._buf['CM']
        else:
            CM = self._blur_coarse(FS.coarse_map)
        stride = max(self.x.size // CM.shape[0], 1)

        out = zeros((self.x.size, self.x.size, DM.shape[0]), dtype=FTYPE)
        self._bin['to_matrix'](out, DM, CM, stride, flag)
        return out.reshape(-1, DM.shape[0])

    def _to_matrixT(self, FS):
        DM = FS.dof_map
        out = zeros((DM.shape[0], self.x.size, self.x.size), dtype=FTYPE)
        self._bin['to_matrixT'](out, DM)
        return out.reshape(DM.shape[0], -1)

    def gnorm(self, k):
        if k not in self.__gnorm:
            self.__gnorm[k] = self._bin['norm'](k)
        return self.__gnorm[k]


def Lasso(A, data, w, adaptive=True, iters=None, prnt=True, plot=True,
          vid=None, scale=True, stop=1e-10, algorithm='Greedy'):
    if iters is None:
        iters = 10000
    if isscalar(iters):
        iters = iters, 25

    if type(adaptive) is bool:
        adaptive = 10000,  # default maximum number of pixels
    elif isscalar(adaptive):
        adaptive = (adaptive,)
    if len(adaptive) < 2:
        adaptive = adaptive[0], True  # default is adaptive

    if A.dim == 1:
        fine_mesh = sparse_mesh(10, 1)
        mesh = sparse_mesh(1 if adaptive[1] else int(log2(adaptive[0] - 1)) + 1, 1)
    elif A.dim == 2:
        fine_mesh = sparse_mesh(8, 2)
        mesh = sparse_mesh(1 if adaptive[1] else int(log2(sqrt(adaptive[0] - 1))) + 1, 2)
    else:
        raise NotImplementedError
    if not adaptive[1]:
        adaptive = mesh.size, False
    recon = sparse_func(0, mesh)
    A.update(mesh)

    D = data.reshape(-1)
    if type(scale) is float:
        w = w * scale
    elif scale:
        w = w * A.T(D).max_abs(fine_mesh.dof_map).max()  # w=1 corresponds to 0 exact solution
    pms = {'eps_max':mesh.H.max(), 'eps_min':mesh.H.min(), 'i':0, 'dof':mesh.size,
       'gap':0, 'gap0':0, 'residual':(0, 0), 'mass':0,
       'E':0, 'F':0, 'F0':0, 'Emin':norm(D) ** 2 / 2, 'Fmax':0, 'F0max':0,
       'gamma':1, 'delta':1, 'delta_crit':1, 'thresh':0, 'thresh0':0, 'max':1,
       'supp_frac':1, 'supp_frac0':1, 'refineI':2, 'factor':1,
       'FS':mesh, 'backstop_gap':norm(D) ** 2 / 2, 'backstop_eps':0.75}

    def energy(u):
        refine(u, True)
        return [pms[t] for t in ('gap0', 'gap', 'E', 'F0', 'F', 'eps_min',
                                 'dof', 'thresh0', 'thresh')]

    def refine(u, dummy=False):
        # u is the current primal, r is the current dual
        # gamma*r is discrete dual feasible, delta*r is dual feasible
        # v is the optimal primal, s is the discretised optimal dual

        if u.dof_map is not u.FS.dof_map:
            u.update(FS=u.FS)
        mesh, DM = u.FS, u.dof_map
        r = A * u - D
        adj = A.T * r
        normU = mesh.norm(u, 1, DM)
        n_r, rD = norm(r), -(r * D).sum()
        Av_norm = [adj.norm(k) for k in range(3)]  # |A^*r|_{C^k}
        eps = pms['eps_max']

        const = abs(A.T(r, u)), adj.max_abs(DM)  # discrete/continuous maximum per pixel
        gamma = max(1e-10, min(rD / n_r ** 2, w / max(1e-10, const[0].max())))  # exactly w/|PA^*r|_inf \approx 1^-
        delta = max(1e-10, min(rD / n_r ** 2, w / max(1e-10, const[1].max())))  # approximately w/|A^*r|_inf \approx 1^-

        E = .5 * n_r ** 2 + w * normU  # current primal
        Estar = gamma * rD - .5 * (gamma * n_r) ** 2  # current discrete dual
        Estar0 = delta * rD - .5 * (delta * n_r) ** 2  # current continuous dual
        pms.update(Emin=min(E, pms['Emin']), Fmax=max(Estar, pms['Fmax']), F0max=max(Estar0, pms['F0max']))
        extrema = pms['Emin'], pms['Fmax'], pms['F0max']

        toGap = lambda t: (2 * max(0, t)) ** .5

        # discrete gap and threshold estimates
        gap = toGap(extrema[0] - extrema[1])  # error to discrete dual
        threshold = max(w - A.gnorm(0) * toGap(E - extrema[1]),  # primal estimate
                        (w - A.gnorm(0) * toGap(extrema[0] - Estar)) / gamma)  # dual estimate

        # continuous gap and threshold estimates
        gap0 = toGap(extrema[0] - extrema[2])  # error to continuous dual
        err0 = min(toGap(E - extrema[2]),  # |r-r^*|
                    # |r-r_{disc}| + |r_{disc}-r^*| = gap + sqrt(eps*|A^*r^*|_{C^1})
                    toGap(E - extrema[1]) + (eps * (Av_norm[1] + A.gnorm(1) * toGap(E - extrema[1])) * E / w))
        threshold0 = max(w - A.gnorm(0) * err0,  # primal estimate
                        (w - A.gnorm(0) * toGap(extrema[0] - Estar0)) / delta)  # dual estimate

        # (E-Estar) + (1-delta/gamma)Estar < 2 (E-Estar)
        # delta/gamma > 1 - (2-1)(E-Estar)/Estar
        delta_crit = gamma * (1 - 1 * (extrema[0] - Estar) / Estar)

        pms.update(gap=extrema[0] - extrema[1], gap0=extrema[0] - extrema[2],
                   residual=(n_r, abs(r).max()), mass=normU, E=E, F=Estar,
                   F0=Estar0, gamma=gamma, delta=delta, delta_crit=delta_crit,
                   thresh=threshold / w, thresh0=threshold0 / w, dof=const[0].size,
                   supp_frac=mesh.integrate(const[0] > threshold, DM),
                   supp_frac0=mesh.integrate(const[1] > threshold0, DM))

        if adaptive[1]:
            ref = zeros(u.size, dtype='int32')  # default no refine
            if u.size < adaptive[0] and delta < delta_crit:
                ref[const[1] > w / delta_crit] = +1  # select pixels to refine

            if pms['gap0'] < pms['backstop_gap']:
                ref[u.FS.H < pms['backstop_eps']] = 0
                pms['backstop_gap'] /= 2 ** A.dim
                pms['backstop_eps'] /= 2

            ref[const[1] < threshold0] = -1  # select pixels to coarsen

            if abs(ref).max() > .1:  # if any pixels are changed
                mesh.refine(ref)
                pms['eps_min'] = mesh.H.min()
                pms['eps_max'] = mesh.H.max()
                pms['Fmax'] = pms['F0max']  # When discretisation changes discrete dual value is invalidated

                A.update(mesh, ref)
#                 factor = A._buf['bwrd_mat'].power(2).sum(1).A1
#                 pms['factor'] = A.norm ** 2 / factor / factor.size

    def gradF(u):
        pms['i'] += 1
        if pms['i'] >= pms['refineI']:
            refine(u)
            pms['refineI'] = max(pms['refineI'] + 1, pms['refineI'] * 1.15)  # 20 refinements for each factor of 10 iterations
#             pms['refineI'] = max(pms['refineI'] + 1, pms['refineI'] * 1.30)  #  9 refinements for each factor of 10 iterations

        if u.dof_map is not u.FS.dof_map:
            u.update(FS=u.FS)

        if pms['factor'] is 1:
            return u.copy(A.T(A * u - D, u.FS))
        else:
            return u.copy(A.T(A * u - D, u.FS) * pms['factor'])

    def proxG(u, t):
        U = u.copy(0)
        if isscalar(pms['factor']):
            shrink(u.x.reshape(-1, 1), t * w * pms['factor'], U.x.reshape(-1, 1))
        else:
            shrink_vec(u.x.reshape(-1, 1), t * w * pms['factor'], U.x.reshape(-1, 1))
        return U

    if A.dim == 1:

        def doPlot(i, u, fig, plt):
            if pms.get('axes', None) is None:
                pms['axes'] = fig.subplots(2, 1)

            ax = pms['axes'][0]
            title = 'Iter=%d, dofs=%d, disc. gap=%.2e, cont. gap=%.2e' % (i, u.size, pms['gap'], pms['gap0'])
            if pms.get('recon plot', None) is None:
                pms['recon plot'] = u.plot(ax=ax, level=10, max_ticks=500, mass=True)
            else:
                u.plot(level=10, max_ticks=500, mass=True, ax=ax, update=pms['recon plot'])
            ax.set_title(title)
            ax.set_ylim(1.1 * min(0, pms['recon plot'].get_ydata().min()),
                        1.1 * max(1e-6, pms['recon plot'].get_ydata().max()))

            ax = pms['axes'][1]
            res = A.T((D - A * u) / w)
            high = res.discretise(fine_mesh.dof_map)
            low = res.discretise(u.dof_map)
            title = ('Est. violation=%.3f, exact violation=%.3f, disc. thresh.=%.2f, cont. thresh.=%.2f'
                     % ((1 / pms['delta'] - 1, abs(high).max() - 1, pms['thresh'], pms['thresh0'])))
            if pms.get('dual plot', None) is None:
                pms['dual plot'] = [None] * 5
                pms['dual plot'][0] = ax.plot(array([0, 1]), [0] * 2, 'r:', [0, 1], [0] * 2, 'r:')
                pms['dual plot'][1] = ax.plot(array([0, 1]), [0] * 2, 'r--', [0, 1], [0] * 2, 'r--')
                pms['dual plot'][2] = ax.plot(array([0, 1]), [1] * 2, 'r-', [0, 1], [-1] * 2, 'r-')
                pms['dual plot'][3] = ax.plot(fine_mesh.dof_map[:, 0] + .5 * fine_mesh.dof_map[:, 1], high, color='black')[0]
                pms['dual plot'][4] = u.copy(low).plot(10, ax=ax, max_ticks=0, color='blue')
            else:
                for j, s in enumerate(('thresh', 'thresh0')):
                    if pms[s] > 0:
                        for L in pms['dual plot'][j]:
                            L.set_alpha(1)
                        pms['dual plot'][j][0].set_ydata([pms[s]] * 2)
                        pms['dual plot'][j][1].set_ydata([-pms[s]] * 2)
                    else:
                        for L in pms['dual plot'][j]:
                            L.set_alpha(0)
                pms['dual plot'][3].set_ydata(high)
                u.copy(low).plot(10, ax=ax, update=pms['dual plot'][4], max_ticks=0, color='blue')
            ax.set_title(title)
            ax.set_ylim(1.1 * min(-1.1, pms['dual plot'][3].get_ydata().min()),
                        1.1 * max(1.1, pms['dual plot'][3].get_ydata().max()))

            if i < 2:
                fig.tight_layout()

    elif A.dim == 2:

        def doPlot(i, u, fig, plt):
            # TODO: remove plt
            if pms.get('axes', None) is None:
                pms['axes'] = fig.subplots(1, 3, gridspec_kw={'width_ratios': [3, 3, 2]})

            if pms.get('data plot', None) is None:
                pms['data plot'] = pms['axes'][0].imshow(data, origin='lower', extent=[0, 1, 0, 1],
                                                         vmin=0, vmax=data.max())
                pms['axes'][0].set_title('Data')

            ax, mesh = pms['axes'][1], u.copy(1 / u.dof_map[:, -1] ** 2);
            try:
                plt.sca(ax)
            except ValueError:
                return
            if pms.get('recon plot', None) is None:
                pms['recon plot'] = u.plot(level=7, ax=ax, mass=True, cmap='Blues')
#                 pms['recon mesh'] = mesh.plot(level=7, ax=ax, mass=False, cmap='Reds', alpha=.25)
                # TODO: include mesh heatmap?
                # TODO: what is the right scaling for vmax?
                ax.set_title('Reconstruction')
            else:
                u.plot(level=7, mass=True, ax=ax, update=pms['recon plot'])
                pms['recon plot'].set_clim(0, 1.5 * pms['recon plot'].get_array().max())
#                 mesh.plot(level=7, mass=False, ax=pms['recon mesh'], scale='log', update=True)
#                 pms['recon mesh'].set_clim(0, pms['recon mesh'].get_array().max())

            ii = stop_crit.iiter; I = stop_crit.I[:ii] + 1
            ax = pms['axes'][2]; plt.sca(ax)
            if pms.get('error plot', None) is None:
                pms['error plot'] = (ax.plot(I, stop_crit.extras[:ii, 1], '-', color='tab:red', label='Discrete PD gap')[0],
                                     ax.plot(I, stop_crit.extras[:ii, 0], '-', color='tab:blue', label='Continuous PD gap')[0],
                                     ax.plot(I, 1 - stop_crit.extras[:ii, 8], '--', color='tab:red', label='Discrete threshold')[0],
                                     ax.plot(I, 1 - stop_crit.extras[:ii, 7], '--', color='tab:blue', label='Continuous threshold')[0],
                                     ax.plot(I, stop_crit.extras[:ii, 6], '-', color='black', label='Pixels')[0])
                ax.set_yscale('log');  ax.set_xscale('log')
                ax.legend(loc='lower left')
                ax.set_xlabel('Iterations')
            else:
                for L in pms['error plot']:
                    L.set_xdata(I)
                pms['error plot'][0].set_ydata(stop_crit.extras[:ii, 1])
                pms['error plot'][1].set_ydata(stop_crit.extras[:ii, 0])
                pms['error plot'][2].set_ydata(1 - stop_crit.extras[:ii, 8])
                pms['error plot'][3].set_ydata(1 - stop_crit.extras[:ii, 7])
                pms['error plot'][4].set_ydata(stop_crit.extras[:ii, 6])
            ax.set_xlim(1, max(2, I[-1]))
            lims = [1e5, 0]
            for L in pms['error plot']:
                lims[0] = min(lims[0], L.get_ydata().min())
                lims[1] = max(lims[1], L.get_ydata().max())
            ax.set_ylim(
                    max(1e-6, 10. ** floor(log10(lims[0] + 1e-7))),
                    min(1e8, 10. ** floor(log10(lims[1] + 1) + 1))
                )

            if i < 2:
                fig.tight_layout()

    def custom_stop(i, *_, d=None, extras=None, **__): return ((extras[0 if adaptive[1] else 1] < stop) and i > 2)

    stop_crit = stopping_criterion(iters[0], custom_stop, frequency=iters[1], prnt=prnt, record=True,
                              energy=energy, vid=vid, fig=None if plot is False else _makeVid(stage=0, record=vid),
                              callback=doPlot if plot else lambda *args:None)

    if type(algorithm) is str:
        algorithm = algorithm, {}
    if algorithm[0] is 'Greedy':
        default = {'xi':0.95, 'S':1}
        default.update(algorithm[1])
        recon = faster_FISTA(recon, 1 / A.norm ** 2, gradF, proxG, stop_crit, **default)
    elif algorithm[0] is 'FISTA':
        default = {'a':10, 'restarting':False}
        default.update(algorithm[1])
        recon = FISTA(recon, 1 / A.norm ** 2, gradF, proxG, stop_crit, **default)
    elif algorithm[0] is 'FB':
        default = {'scale':2}
        default.update(algorithm[1])
        recon = FB(recon, default['scale'] / A.norm ** 2, gradF, proxG, stop_crit)

    return recon, stop_crit


class CBPArray:

    def __init__(self, *args, original=True, dx=None, scale=1):
        if len(args) > 3:
            x, original, dx = args[:3], args[3], args[4]
        elif len(args) == 3:
            x = args
        else:
            x = args[0]
#         Array.__init__(self, list(x), True)
        self.x = list(x)
        self.original = original
        self.dx = dx
        self.scale = scale

    def copy(self): return CBPArray([x.copy() for x in self.x], original=self.original, dx=self.dx, scale=self.scale)

    def to_original(self):
        if self.original:
            return self.copy()
        else:
            ap = .5 * (self.x[0] + self.x[1] + self.x[1])
            am = .5 * (-self.x[0] + self.x[1] + self.x[1])
            b = (self.dx / 2) * (self.x[1] - self.x[2])
            return CBPArray(ap, am, b, original=True, dx=self.dx)

    def to_cone(self):
        if self.original:
            a = self.x[0] - self.x[1]
            b = .5 * (self.x[0] + self.x[1] + (2 / self.dx) * self.x[2])
            c = .5 * (self.x[0] + self.x[1] - (2 / self.dx) * self.x[2])
            return CBPArray(a, b, c, original=False, dx=self.dx)
        else:
            return self.copy()

    def to_measure(self):
        arr = self.x if self.original else self.to_original().x
        x = arange(self.dx / 2, 1, self.dx)
        a = arr[0] - arr[1]
        ind = (abs(a) > 1e-20)
        out = zeros((ind.sum(), 2))
        out[:, 0] = a[ind]
        out[:, 1] = x[ind] + self.scale * arr[2][ind] / a[ind]
        return out

    @property
    def size(self): return self.x[0].size

    def asarray(self): return concatenate([a.reshape(-1, 1) for a in self.x], axis=1)

    def ravel(self): return concatenate([a.ravel() for a in self.x], axis=0)

    def update(self, *args):
        self.x = args if len(args) > 1 else args[0]

    def __neg__(self): return CBPArray([-xi for xi in self.x], original=self.original, dx=self.dx, scale=self.scale)

    def __add__(self, other): return CBPArray([xi + other.x[i] for i, xi in enumerate(self.x)], original=self.original, dx=self.dx, scale=self.scale)

    def __sub__(self, other): return CBPArray([xi - other.x[i] for i, xi in enumerate(self.x)], original=self.original, dx=self.dx, scale=self.scale)

    def __mul__(self, other): return CBPArray([xi * other for xi in self.x], original=self.original, dx=self.dx, scale=self.scale)

    def __truediv__(self, other): return CBPArray([xi / other for xi in self.x], original=self.original, dx=self.dx, scale=self.scale)

    def __rtruediv__(self, other): return CBPArray([other / xi for xi in self.x], original=self.original, dx=self.dx, scale=self.scale)

    def __iadd__(self, other):
        for i, xi in enumerate(self.x):
            xi += other.x[i]

    def __isub__(self, other):
        for i, xi in enumerate(self.x):
            xi -= other.x[i]

    def __imul__(self, other):
        for xi in self.x:
            xi *= other

    def __itruediv__(self, other):
        for xi in self.x:
            xi /= other

    __radd__, __rmul__ = __add__, __mul__


@jit
def sqrd(x): return x * x


@jit(nopython=True)
def CBPProx(a, b, c, A, B, C, h):
    '''
    min |a-A|^2 + |b-B|^2 + |c-C|^2 such that a,b \geq 0, |c|\leq h(a+b)
    '''
    h2 = h * h
    for i in range(a.size):
        #         ee = [-1, -1, -1, -1]
        e = -1
        CC = abs(C[i])

        # Case 1: |c| < h(a+b), remainder becomes trivial
        a0, b0 = max(A[i], 0), max(B[i], 0)
#         ee[0] = sqrd(a0 - A[i]) + sqrd(b0 - B[i]) + sqrd(min(CC, h * (a0 + b0)) - CC)
        if CC <= h * (a0 + b0):
            e = sqrd(a0 - A[i]) + sqrd(b0 - B[i])
            a[i], b[i], c[i] = a0, b0, CC
        # else, c = h(a+b)sign(C)
        # Case 2: a, b > 0
        a0 = ((h2 + 1) * A[i] - h2 * B[i] + h * CC) / (1 + 2 * h2)
        b0 = ((h2 + 1) * B[i] - h2 * A[i] + h * CC) / (1 + 2 * h2)
#         ee[1] = sqrd(max(0, a0) - A[i]) + sqrd(max(0, b0) - B[i]) + sqrd(h * (a0 + b0) - CC)
        if a0 >= 0 and b0 >= 0:
            e0 = sqrd(a0 - A[i]) + sqrd(b0 - B[i]) + sqrd(h * (a0 + b0) - CC)
            if e < 0 or e0 < e:
                e = e0
                a[i], b[i], c[i] = a0, b0, h * (a0 + b0)

        # Case 3: a=0, problem becomes 1D
        a0, b0 = 0, max(0, (B[i] + h * CC) / (1 + h2))
        e0 = sqrd(A[i]) + sqrd(b0 - B[i]) + sqrd(h * b0 - CC)
#         ee[2] = sqrd(a0 - A[i]) + sqrd(b0 - B[i]) + sqrd(h * (a0 + b0) - CC)
        if e < 0 or e0 < e:
            e = e0
            a[i], b[i], c[i] = 0, b0, h * b0

        # Case 4: b=0, same but flipped
        a0, b0 = max(0, (A[i] + h * CC) / (1 + h2)), 0
        e0 = sqrd(a0 - A[i]) + sqrd(B[i]) + sqrd(h * a0 - CC)
#         ee[3] = sqrd(a0 - A[i]) + sqrd(b0 - B[i]) + sqrd(h * (a0 + b0) - CC)
        if e < 0 or e0 < e:
            e = e0
            a[i], b[i], c[i] = a0, 0, h * a0

        if C[i] < 0:
            c[i] = -c[i]

#         if min(ee) - e > 1e-20:
#             raise

# # Checking 0 sub-derivative on the prox steps
#         o = 1e-15
#         if min(a[i], b[i]) < -o or abs(c[i]) > h * (a[i] + b[i]) + o:
#             raise
#
#         FLAG = abs(c[i]) > h * (a[i] + b[i]) - o
#         if a[i] > -o or FLAG:
#             if a[i] < A[i] - o:
#                 raise
#         elif abs(a[i] - A[i]) > o:
#             raise
#         if b[i] > -o or FLAG:
#             if b[i] < B[i] - o:
#                 raise
#         elif abs(b[i] - B[i]) > o:
#             raise
#
#         if FLAG:
#             if abs(c[i]) > CC + o:
#                 raise
#         elif abs(c[i] - C[i]) > o:
#             raise


def CBP(A, data, w, size=None, iters=None, prnt=True, plot=True, vid=None, stop=1e-10, algorithm='Greedy'):
    '''
    Solve the primal problem:
        min 1/2|Au-d|_2^2 + w|u|_1
    with the approximation
        Au \approx A(mesh)u_1 + A'(mesh)u_2
    where the energy becomes
        min 1/2|Au_1+A'u_2-d|_2^2 + w|u_1|_1 s.t. |u_2|\leq h/2|u_1|
    with its dual:
        min 1/2|v+d|_2^2 such that |A^*v|\leq w
    Related at optima by:
        v = Au-d

    Parameters
    ----------
    A : `numpy.ndarray`, (n,m) or an equivalent `op`-type object
        The matrix, or equivalent, object mapping from domain to image
    data : `numpy.ndarray`, (n,)
        The data array
    w : `float` > 0
        Positive L1 weighting
    size : `int`, optional
        Number of pixels (default 10^4).
    record : `bool`, optional
        If `True` then values such as the energy are recorded and returned
    plot : `dict`, optional
        Extra parameters to pass to the plotting function
    vid : `None` or `dict`, optional
        If `None` (default) then no video is recorded, otherwise must have a `filename`
        and `fps` key.
    iters : `int`, (`int`, `int`), or `None`, optional
        Number of iterations to compute. First integer is total number of iterations
        (default 10^4) and second is frequency of recording energy values (default
        every 25th iteration).
    '''
    if iters is None:
        iters = 10000
    if isscalar(iters):
        iters = iters, 25

    if size is None:
        size = 10000  # default, potential very large memory
    elif type(size) is not int:
        size = size[0]

    fine_mesh = sparse_mesh(10, 1)
    mesh = sparse_mesh(int(log2(size - 1)) + 1, 1)
    size = mesh.size

    D = data.reshape(-1)
    w = w * A.T(D).max_abs(fine_mesh.dof_map).max()  # w=1 corresponds to 0 exact solution

    pms = {'eps_max':mesh.H.max(), 'eps_min':mesh.H.min(), 'i':0, 'dof':mesh.size,
       'gap':0, 'gap0':0, 'residual':(0, 0), 'E': 0, 'F': 0, 'F0': 0, 'Emin': 1e16, 'E0min': 1e16, 'Fmax':-1e16, 'Fmax_d':-1e16, 'F0max':-1e16,
       'thresh':0, 'thresh0':0, 'FS':mesh}
    dx = pms['eps_min']

    oldA = A
    A, B = oldA._to_matrix(mesh, 1), oldA._to_matrix(mesh, 2)  # evaluation and gradient at midpoint
    n = norm(A.dot(A.T), 2), norm(B.dot(B.T), 2)
    scale = (n[0] / n[1]) ** .5
#     scale = 1
    B *= scale
#     print(norm(A.dot(A.T), 2) ** .5, norm(B.dot(B.T), 2) ** .5)
#     exit()
    myop = op(lambda x: A.dot(x.x[0] - x.x[1]) + B.dot(x.x[2]),
              lambda y: CBPArray(A.T.dot(y), A.T.dot(-y), B.T.dot(y), original=False, dx=dx),
              norm=1.1 * (norm(A.dot(A.T), 2) + norm(B.dot(B.T), 2)) ** .5)

    def energy(u):
        refine(u, True)
        return [pms[t] for t in ('gap0', 'gap', 'E', 'F0', 'F', 'eps_min',
                                 'dof', 'thresh0', 'thresh')]

    def refine(u, dummy=False):
        eps = dx

        # Discrete estimates
        '''
        If E(a,b,c) = .5(Aa-Ab+Bc-d)^2 + w(a+b) such that a,b>0, |c|<h/2(a+b)
        then the dual is
        F(z) = -<z,d> -.5|z|^2 such that |A^Tz| + h/2|B^Tz| < w
        with optimality condition
        z = Aa-Ab+Bc-d
        '''
        r = myop * u - D
        n_r = norm(r)
        h = eps / u.scale

        gamma = min(1, w / (abs(A.T.dot(r)) + (h / 2) * abs(B.T.dot(r))).max())
        E = .5 * n_r ** 2 + w * (u.x[0].sum() + u.x[1].sum())  # current discrete primal
        Estar = -gamma * (r * D).sum() - .5 * (gamma * n_r) ** 2  # current discrete dual

        tmp = pms['Emin'], pms['Fmax'], pms['gap']
        pms.update(Emin=min(E, pms['Emin']), Fmax=max(Estar, pms['Fmax']), gamma=gamma)
        pms['gap'] = pms['Emin'] - pms['Fmax']

        # Continuous estimates
        measure = u.to_measure()
        normU = abs(measure[:, 0]).sum()
        r = oldA * measure - D
        adj = oldA.T * r
        n_r = norm(r)
        Av_norm = [adj.norm(k) for k in range(3)]  # |A^*r|_{C^k}
#         const = oldA.T(r)._max_abs(mesh, Av_norm[2])
        const = abs(oldA.T(r, mesh)), adj.max_abs(mesh.dof_map)  # discrete/continuous maximum per pixel
        gamma = min(1, w / const[0].max())  # exactly w/|PA^*r|_inf \approx 1^-
        delta = min(1, w / const[1].max())  # approximately w/|A^*r|_inf \approx 1^-

        E = .5 * n_r ** 2 + w * normU  # current primal
        Estar = -gamma * (r * D).sum() - .5 * (gamma * n_r) ** 2  # current discrete dual
        Estar0 = -delta * (r * D).sum() - .5 * (delta * n_r) ** 2  # current continuous dual

        pms.update(E0min=min(E, pms['E0min']), Fmax_d=max(Estar, pms['Fmax_d']), F0max=max(Estar0, pms['F0max']))
        extrema = pms['E0min'], pms['Fmax_d'], pms['F0max']

        toGap = lambda t: (2 * max(0, t)) ** .5
#         # discrete gap and threshold estimates
#         gap = toGap(extrema[0] - extrema[1])  # error to discrete dual
#         threshold = min(w - A.gnorm(0) * toGap(E - extrema[1]),  # primal estimate
#                         (w - A.gnorm(0) * toGap(extrema[0] - Estar)) / gamma)  # dual estimate
#
#         # continuous gap and threshold estimates
#         gap0 = toGap(extrema[0] - extrema[2])  # error to continuous dual
#         err0 = min(toGap(E - extrema[2]),  # |r-r^*|
#                     # |r-r_{disc}| + |r_{disc}-r^*| = gap + sqrt(eps*|A^*r^*|_{C^1})
#                     toGap(E - extrema[1]) + (eps * (Av_norm[1] + A.gnorm(1) * toGap(E - extrema[1])) * E / w))
#         threshold0 = min(w - A.gnorm(0) * err0,  # primal estimate
#                         (w - A.gnorm(0) * toGap(extrema[0] - Estar0)) / delta)  # dual estimate
#
#         # (E-Estar) + (1-delta/gamma)Estar < 2 (E-Estar)
#         # delta/gamma > 1 - (2-1)(E-Estar)/Estar
#         delta_crit = gamma * (1 - 1 * (extrema[0] - Estar) / Estar)

        threshold = min(w - oldA.gnorm(0) * toGap(E - extrema[1]),  # primal estimate
                        (w - oldA.gnorm(0) * toGap(extrema[0] - Estar)) / gamma)  # dual estimate

        err0 = min(toGap(E - extrema[2]),
                    toGap(E - extrema[1]) + (eps * (Av_norm[1] + oldA.gnorm(1) * toGap(E - extrema[1])) * E / w))
        threshold0 = min(w - oldA.gnorm(0) * err0,  # primal estimate
                        (w - oldA.gnorm(0) * toGap(extrema[0] - Estar0)) / delta)  # dual estimate

        pms.update(gap0=extrema[0] - extrema[2],
                   residual=(n_r, abs(r).max()), E=E, F=Estar, F0=Estar0, delta=delta,
                   thresh=threshold / w, thresh0=threshold0 / w)

        return u

    def gradF(u):
        #             d/du \frac12|Au-D|^2_2 +w(u1+u2)= A.T*(Au-D) + w
        out = myop.T(myop(u) - D)  # data term
        out.x[0] += w  # L1 term
        out.x[1] += w

        return out

    def proxG(u, g):
        out = [empty(a.shape, dtype='double') for a in u.x]
        CBPProx(*out, *u.x, u.dx / (2 * u.scale))
        return CBPArray(out, original=True, dx=u.dx, scale=u.scale)

    def doPlot(i, u, fig, plt):
        if pms.get('axes', None) is None:
            pms['axes'] = fig.subplots(2, 1)

        ax = pms['axes'][0]
        measure = u.to_measure()
        arr = [([x, 0], [x, h], [x, 0]) for h, x in measure]
        arr = array([(0, 0)] + [L2 for L1 in arr for L2 in L1] + [(1, 0)], dtype=float)
        title = 'Iter=%d, dofs=%d, disc. gap=%.2e, cont. gap=%.2e' % (i, u.size, pms['gap'], pms['gap0'])
        if pms.get('recon plot', None) is None:
            pms['recon plot'] = ax.plot(arr[:, 0], arr[:, 1])[0]
            ax.set_xlim(0, 1)
        else:
            pms['recon plot'].set_xdata(arr[:, 0])
            pms['recon plot'].set_ydata(arr[:, 1])
        ax.set_title(title)
        ax.set_ylim(1.1 * min(0, pms['recon plot'].get_ydata().min()),
                    1.1 * max(1e-6, pms['recon plot'].get_ydata().max()))

        ax = pms['axes'][1]
        res = oldA.T((D - oldA * measure) / w)
        high = res.discretise(fine_mesh.dof_map)
        low = res.discretise(mesh.dof_map)
        dual = myop.adjoint(myop * u - D)
        title = ('Est. violation=%.3f, exact violation=%.3f, disc. thresh.=%.2f, cont. thresh.=%.2f'
                 % ((1 / pms['delta'] - 1, abs(high).max() - 1, pms['thresh'], pms['thresh0'])))
        if pms.get('dual plot', None) is None:
            pms['dual plot'] = [None] * 5
            pms['dual plot'][0] = ax.plot(array([0, 1]), [0] * 2, 'r:', [0, 1], [0] * 2, 'r:')
            pms['dual plot'][1] = ax.plot(array([0, 1]), [0] * 2, 'r--', [0, 1], [0] * 2, 'r--')
            pms['dual plot'][2] = ax.plot(array([0, 1]), [1] * 2, 'r-', [0, 1], [-1] * 2, 'r-')
            pms['dual plot'][3] = ax.plot(arange(u.dx / 2, 1, u.dx), (abs(dual.x[0]) + (u.dx / 2) * abs(dual.x[2])) / w, color='black')[0]
            pms['dual plot'][4] = ax.plot(mesh.dof_map[:, 0] + .5 * mesh.dof_map[:, 1], low, color='blue')[0]
            ax.set_xlim(0, 1)
        else:
            for j, s in enumerate(('thresh', 'thresh0')):
                if pms[s] > 0:
                    for L in pms['dual plot'][j]:
                        L.set_alpha(1)
                    pms['dual plot'][j][0].set_ydata([pms[s]] * 2)
                    pms['dual plot'][j][1].set_ydata([-pms[s]] * 2)
                else:
                    for L in pms['dual plot'][j]:
                        L.set_alpha(0)
            pms['dual plot'][3].set_ydata((abs(dual.x[0]) + (u.dx / 2) * abs(dual.x[2])) / w)
            pms['dual plot'][4].set_ydata(low)
        ax.set_title(title)
        ax.set_ylim(1.1 * min(-1.1, pms['dual plot'][4].get_ydata().min()),
                    1.1 * max(1.1, pms['dual plot'][3].get_ydata().max(), pms['dual plot'][4].get_ydata().max()))

        if i < 2:
            fig.tight_layout()

    def custom_stop(i, *_, d=None, extras=None, **__): return ((extras[1] < stop) and i > 2)

    stop_crit = stopping_criterion(iters[0], custom_stop, frequency=iters[1], prnt=prnt, record=True,
                              energy=energy, vid=vid, fig=None if plot is False else _makeVid(stage=0, record=vid),
                              callback=(lambda *_: None) if plot is False else doPlot)

    recon = CBPArray(zeros(size, dtype=float), zeros(size, dtype=float), zeros(size, dtype=float), dx=dx, scale=scale)

    if type(algorithm) is str:
        algorithm = algorithm, {}
    if algorithm[0] is 'Greedy':
        default = {'xi':0.95, 'S':1}
        default.update(algorithm[1])
        recon = faster_FISTA(recon, 1 / myop.norm ** 2, gradF, proxG, stop_crit, **default)
    elif algorithm[0] is 'FISTA':
        default = {'a':10, 'restarting':False}
        default.update(algorithm[1])
        recon = FISTA(recon, 1 / myop.norm ** 2, gradF, proxG, stop_crit, **default)
    elif algorithm[0] is 'FB':
        default = {'scale':2}
        default.update(algorithm[1])
        recon = FB(recon, default['scale'] / myop.norm ** 2, gradF, proxG, stop_crit)

    return recon, stop_crit


if __name__ == '__main__':

    import warnings
    warnings.filterwarnings("ignore", message='Adding an axes using the same arguments as a previous axes')
    warnings.filterwarnings("ignore", message='Data has no positive values, and therefore cannot be log-scaled.')
    from numpy import random

    test = 3
    if test == 1:  # check kernel maps
        from sparse_ops import _test_op
        sigma = .1
        x = linspace(0, 1, 11)
        A = kernelMap1D(ker='Gaussian', x=x, sigma=sigma)
        _test_op(A)
        mesh = sparse_mesh(4, 1)
        f = A.T * linspace(1, 10, x.size)
        for k in range(3):
            f.discretise(mesh.dof_map, k)
            f.norm(k)
        f.max_abs(mesh.dof_map)
        print('1D Gaussian operator checked')
        x = linspace(1, 1000, 11)
        A = kernelMap1D(ker='Fourier', x=x)
        _test_op(A)
        mesh = sparse_mesh(4, 1)
        f = A.T * linspace(1, 10, x.size)
        for k in range(3):
            f.discretise(mesh.dof_map, k)
            f.norm(k)
        f.max_abs(mesh.dof_map)
        print('1D Fourier operator checked')
        x = linspace(0, 1, 6)
        sigma = .1
        A = kernelMap2D(ker='Gaussian', x=x, sigma=sigma)
        _test_op(A)
        mesh = sparse_mesh(4, 2)
        f = A.T * linspace(1, 10, x.size ** 2)
        for k in range(3):
            f.discretise(mesh.dof_map, k)
            f.norm(k)
        f.max_abs(mesh.dof_map)
        print('2D Gaussian operator checked')
        x = linspace(0, 1, 6)
        sigma = .1
        A = kernelMapMesh2D(ker='Gaussian', x=x, sigma=sigma)
        _test_op(A)
        mesh = sparse_mesh(4, 2)
        f = A.T * linspace(1, 10, x.size ** 2)
        for k in range(3):
            f.discretise(mesh.dof_map, k)
            f.norm(k)
        f.max_abs(mesh.dof_map)
        print('2D GaussianMesh operator checked')

    elif test == 2:  # 1D optimisation
        random.seed(1)
        iters = (10000, 1.1)
        adaptive = 1000, True
        vid = {'filename':'videos/lasso000', 'fps':5}
#         vid = None

        ker, peaks, sigma = 'Gaussian', 10, 0.12
        gt_measure = array(concatenate([3 * random.randn(peaks, 1), .1 + .8 * random.rand(peaks, 1)], axis=1))
        if ker == 'Fourier':
            points = 100 * 2 * (random.rand(30, 1) - .5)
        else:
            points = linspace(0, 1, 30)

        A = kernelMap1D(ker=ker, x=points, sigma=sigma)
        data = A * gt_measure

        recon, record = Lasso(A, data, .06 if ker[0] == 'G' else .02, adaptive, iters, vid=vid)
#         recon, record = CBP(A, data, .06 if ker[0] == 'G' else .02, adaptive, iters, vid=vid)

        f = plt.figure(1)
        i = record.I + 1
        plt.plot(i, record.extras[:, 0], 'm-', label='continuous gap')
        plt.plot(i, record.extras[:, 1], 'm--', label='discrete gap')
        plt.plot(i, record.extras[:, 5], 'b--', label='eps')
        plt.plot(i, record.extras[:, 6], 'b-', label='pixels')
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(left=1, right=iters[0])
        ax.set_ylim(bottom=1e-4, top=1e3)
        ax.grid(True, 'major', 'both')
        plt.legend(loc='lower left')

        plt.show()
        exit()

    elif test == 3:  # 2D optimisation
        random.seed(1)
        iters = (1000, 1.2)
        adaptive = 10 ** 8, True
        vid = {'filename':'videos/lasso2D_vid_%s%d' % ('adapt' if adaptive[1] else 'fixed', +round(log10(adaptive[0]))),
           'fps':int(log(iters[0]) / log(iters[1]) / 60) + 1}
        vid = None

        ker, peaks, sigma = 'Gaussian', 10, 0.12
        gt_measure = array(concatenate([random.rand(peaks, 1), .1 + .8 * random.rand(peaks, 1), .1 + .8 * random.rand(peaks, 1)], axis=1))
        points = linspace(0, 1, 16)

        A = kernelMap2D(ker=ker, x=points, sigma=sigma)
        data = (A * gt_measure).reshape(points.size, points.size)

        recon, record = Lasso(A, data, .1, adaptive, iters, vid=vid)

        f = plt.figure(figsize=(18, 10))
        i = record.I
        i[0] = 1
        plt.plot(i, record.extras[:, 0], 'm-', label='continuous gap')
        plt.plot(i, record.extras[:, 1], 'm--', label='discrete gap')
        plt.plot(i, record.extras[:, 5], 'b--', label='eps')
        plt.plot(i, record.extras[:, 6], 'b-', label='pixels')
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(left=1, right=iters[0])
#         ax.set_ylim(bottom=1e-4, top=1e3)
        ax.grid(True, 'major', 'both')
        plt.legend(loc='lower left')

        plt.show()
        exit()

    elif test == 4:  # storm data
#         from run_STORM import merge_results
#         merge_results(8, 'large_w')
#         print('finished')
#         exit()
        from skimage.io import imread
        iters = (1000, 1.2)
        adaptive = 10 ** 6, True

        A = kernelMapMesh2D(x=linspace(0, 1, 64 + 1)[:-1] + 1 / 128,
                        sigma=2 / 64 / sqrt(2 * log(2)), Norm=64)
        data = imread('STORM_data/sequence-as-stack-MT4.N2.HD-2D-Exp.tif').astype(FTYPE) / 2000 - 0.07
#         vid = {'filename':'videos/lasso2D_vid_slice', 'fps':5}
        vid = None
#         data = data.mean() - 0.009
#         recon, record = Lasso(A, data, .01, adaptive, iters, vid=vid, plot=True)
        data = data[0]
        recon, record = Lasso(A, data, .15, adaptive, iters, vid=vid, plot=True)
        if vid is None:
            plt.show()

