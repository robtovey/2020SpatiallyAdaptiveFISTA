'''
Created on 7 Sep 2020

@author: Rob Tovey
'''
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from numba import jit, prange
from numpy import log2
from numba.typed import List
__params = {'nopython':True, 'parallel':False, 'fastmath':True, 'cache':True}
__pparams = __params.copy(); __pparams['parallel'] = True
__cparams = __params.copy(); __cparams['cache'] = False
EPS, FTYPE = 1e-25, 'float64'

pass  # Numba wrapping of pymorton

from pymorton import __part1by1, __part1by2, __unpart1by1, __unpart1by2
pm_p1b1, pm_p1b2, pm_u1b1, pm_u1b2 = [jit()(f) for f in (__part1by1, __part1by2, __unpart1by1, __unpart1by2)]


@jit(**__params)
def interleave2(i, j): return pm_p1b1(i) | (pm_p1b1(j) << 1)


@jit(**__params)
def deinterleave2(i, j, out):
    out[0] = pm_u1b1(i)
    out[1] = pm_u1b1(j) >> 1


@jit(**__params)
def interleave3(i, j, k):
    return pm_p1b2(i) | (pm_p1b2(j) << 1) | (pm_p1b2(k) << 2)


@jit(**__params)
def deinterleave3(i, j, k, out):
    out[0] = pm_u1b2(i)
    out[1] = pm_u1b2(j) >> 1
    out[2] = pm_u1b2(k) >> 2


pass  # Adaptive meshing


class adaptive_mesh:  # odl RectPartition

    def __init__(self, init=0, dim=1, coarse_level=5, _parent=None):

        self.dim = dim
        if self.dim == 1:
            self.__splitarr = ((0,), (1,))
        elif self.dim == 2:
            self.__splitarr = ((0, 0), (1, 0), (0, 1), (1, 1))
        else:
            self.__splitarr = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
                               (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))
        self.__splitarr = np.array(self.__splitarr, dtype='int32')
        self.coarse_level = coarse_level

        # TODO: this needs better stability/robustness/utilisation
        # Tracks which mesh is the 'newest'
        self.parent = _parent
        self.age = 0 if _parent is None else _parent.age + 1

        self.__buf = [np.zeros([1] * dim, dtype='int32'), np.zeros([1, dim], dtype='int32')]

        # dof_map[i] = (x,y,...,h) position and size of pixel i
        # h = minimum pixel size
        if type(init) is int:  # produce uniform mesh with <init> refinements
            self.dof_map = np.array([(0,) * dim + (1,)], dtype=FTYPE)
            self.h = 1
            for _ in range(init):
                self.refine()
        elif init.shape[0] == 0:  # empty initialisation, only 1 big pixel
            self.dof_map = np.array([(0,) * dim + (1,)], dtype=FTYPE)
            self.h = 1
        else:  # dof_map is given
            assert init.shape[1] == dim + 1
            self.dof_map = init
            self.h = self.dof_map[:, -1].min()
        self.update()

    def refine(self, indcs=None):
        '''
        self.refine(indcs=None)

        No return value, in-place changing of mesh
        If indcs is None then all pixels are split
        Otherwise, indcs is a list of flags:
            >0 indicates refinement
             0 indicates unchanged
            <0 indicates removal
        '''

        # TODO: what is this for? Single width 0 pixel?
        if self.dof_map.shape[0] == 1 and self.dof_map[0, -1] == 0:
            self.update()
            return

        if indcs is None:  # refine all
            indcs = np.empty(self.size, dtype='int32')
            indcs.fill(1)
        else:
            indcs = np.array(indcs, dtype='int32')
        assert indcs.size == self.dof_map.shape[0]

        # tot_len = memory needed to add all new squares
        tot_len = self.dof_map.shape[0] + (indcs > 0).sum() * 2 ** self.dim
        new = np.empty((tot_len, self.dof_map.shape[1]), dtype=FTYPE)
        if new.size > 0:
            end = self._refine(new, self.dof_map, indcs, self.__splitarr)
            new = new[:end]  # cut off unnecessary elements
        if new.size > 0:
            self.dof_map = new
        else:  # TODO: is this the right trivial element?
            self.dof_map = np.array([(0,) * self.dim + (0,)], dtype=FTYPE)  # 1 box of width 0

        self.h = self.dof_map[:, -1].min()
        self.age += 1
        self.update()

    def fine2coarse(self, dof_map, coarse_level):
        '''
        Pre-compute the indices which correspond to pixels in a coarse discretisation.
        Say coarse_map[i0,i1,...] = j0,j1
        where region coarse pixel (i0,i1) includes dofs [j0,j1-1] inclusive. 
        '''
        scale = 2 ** coarse_level

        coarse_map = np.empty((scale,) * self.dim + (2,), dtype=int)
        coarse_map[...,:] = self.size, -1  # identify areas with no pixels

        self._fine2coarse(dof_map, coarse_map, 1 / scale)
        return coarse_map

    def update(self, coarse_level=None):
        if coarse_level is not None:
            self.coarse_level = coarse_level
        self.coarse_map = self.fine2coarse(self.dof_map, self.coarse_level)
        self._update()

    @property
    def size(self): return self.dof_map.shape[0]

    @property
    def H(self): return self.dof_map[:, -1]

    def _update(self): return

    def _refine(self, new_DM, old_DM, indcs, splitarr):
        '''
        end = self._refine(new_DM, old_DM, indcs, splitarr)

        Function which computes new dof map from a list of instructions to
        refine/coarsen

        Inputs:
        -------
        new_DM : array(shape=(N,dim+1), dtype=FTYPE)
            output array for new dof map
        old_DM : array(shape=(M,dim+1), dtype=FTYPE)
            old dof map
        indcs : array(shape=(M,dim+1), dtype='int32')
            instructions to refine or coarsen
            indcs[i] > 0 ---> refine
            indcs[i] < 0 ---> potentially coarsen
        splitarr : array(shape=(2**dim,dim), dtype='int32')
            order in which to add new squares, e.g. in 2D the square with
            corners [x,y] and [x+2d,y+2d] is split into the ordered set:
            [ [x+i*d, y+j*d] --> [x+(i+1)d, y+(j+1)d] for (i,j) in splitarr]

        Outputs:
        --------
        end : int <= N
            cut-off parameter such that only new_DM[:end] has been written on

        '''
        raise NotImplementedError

    def _fine2coarse(self, dof_map, coarse_map, h):
        '''
        self._fine2coarse(dof_map, coarse_map, h)

        coarse_map is a (coarse) inverse of dof_map which returns a range of
        indices corresponding to a region of space.

        Inputs:
        -------
        dof_map : array(shape=(N,dim+1), dtype=FTYPE)
            dof_map to invert
        h : float between 0 and 1
            resolution at which to compute inverse
        coarse_map : array(shape=(n,)*dim+(2,), dtype='int32')
            The only pixels inside the interval [i*h,(i+1)*h] are j such that
            coarse_map[i,0] <= j < coarse_map[i,1]
        '''
        if self.dim == 1:
            _fine2coarse1(dof_map, coarse_map, h)
        elif self.dim == 2:
            _fine2coarse2(dof_map, coarse_map, h)
        else:
            raise NotImplementedError


class sparse_mesh(adaptive_mesh):

    def _refine(self, new, dof_map, indcs, splitarr):
        # refine mesh means removing the old pixel and replacing it with smaller new ones
        return _refine_mesh(new, dof_map, indcs, splitarr)

    @property
    def size(self): return self.dof_map.shape[0]
    @property
    def shape(self): return self.dof_map.shape[0], 1


class sparse_tree(adaptive_mesh):

    def _refine(self, new, dof_map, indcs, splitarr):
        if not hasattr(self, 'isleaf'):
            self.isleaf = np.empty(dof_map.shape[0], dtype=bool)
            _isleaf(dof_map, self.isleaf)
        oldisleaf = self.isleaf
        newisleaf = np.empty(new.shape[0], dtype=bool)

        # refine tree means keeping old pixel and adding smaller new ones
        end = _refine_tree(new, newisleaf, dof_map, oldisleaf, indcs, splitarr)
        self.isleaf = newisleaf[:end]
        return end

    def _update(self):
        if not hasattr(self, 'isleaf'):
            self.isleaf = np.empty(self.size, dtype=bool)
            _isleaf(self.dof_map, self.isleaf)

    @property
    def size(self): return self.dof_map.shape[0]
    @property
    def shape(self): return self.dof_map.shape[0], 2 ** self.dim - 1


pass  # Adaptive function spaces


class adaptive_FS(adaptive_mesh):  # odl DiscretizedSpace
    # This is slightly lazy, implicitly adaptive_FS inherits from two concepts:
    # adaptive meshes and function spaces

    @property
    def maps(self): return (self.dof_map, self.coarse_map, self.age)

    @maps.setter
    def maps(self, value):
        if hasattr(value, 'dof_map'):
            self.dof_map, self.coarse_map, self.age = value.dof_map, value.coarse_map, value.age
        else:
            self.dof_map, self.coarse_map, self.age = value

    def zero(self, maps=None):
        '''
        f = self.zero(maps=None)

        Return a function representing the 0 function
        '''
        return self.one(0, maps)

    def one(self, v=1, maps=None):
        '''
        f = self.one(v=1)
        Return a function such that f(x) = v for all x
        '''
        raise NotImplementedError

    def integrate(self, x, dof_map=None):
        if hasattr(x, 'dof_map'):
            dof_map = x.dof_map
            x = x.x
        elif dof_map is None:
            dof_map = self.dof_map
        assert x.shape[0] == dof_map.shape[0]
        return self._integrate(x, dof_map)

    def inner(self, x, y, dof_map=None):
        dof_map = self.dof_map if dof_map is None else dof_map
        if hasattr(x, 'dof_map'):
            dof_map = x.dof_map
            x = x.x
        if hasattr(y, 'dof_map'):
            dof_map = y.dof_map
            y = y.x

        assert x.size == y.size
        return self._inner(x, y, dof_map)

    def norm(self, x, p=2, dof_map=None):
        if hasattr(x, 'dof_map'):
            dof_map = x.dof_map
            x = x.x
        elif dof_map is None:
            dof_map = self.dof_map

        return self._norm(x, dof_map, p)

    def old2new(self, arr, dof_map, copy=True):
        if dof_map is self.dof_map:
            new = (arr.copy() if copy else arr)
        else:
            new = np.zeros((arr.shape[0] + self.dof_map.shape[0] - dof_map.shape[0], arr.shape[1]), dtype=FTYPE)
            self._old2new(new, self.dof_map, arr, dof_map)
        return new

    def _integrate(self, x, dof_map):
        '''
        v = self._integrate(x, dof_map)

        Compute integral of a function

        Inputs:
        -------
        x : array(shape=(N,k), dtype=FTYPE)
            array representing function coefficients
        dof_map : array(shape=(N,dim+1), dtype=FTYPE)
            corresponding mesh

        Output:
        -------
        v : float
            integral of the function represented by the x on dof_map

        '''
        raise NotImplementedError

    def _inner(self, x, y, dof_map):
        '''
        v = self._inner(x, y, dof_map)

        Compute inner product of two functions

        Inputs:
        -------
        x,y : array(shape=(N,k), dtype=FTYPE)
            arrays representing function coefficients
        dof_map : array(shape=(N,dim+1), dtype=FTYPE)
            corresponding mesh

        Output:
        -------
        v : float
            integral of the function represented by the x on dof_map

        '''
        raise NotImplementedError

    def _norm(self, x, dof_map, p):
        '''
        v = self._norm(x, dof_map, p)

        Compute norm of a function

        Inputs:
        -------
        x : array(shape=(N,k), dtype=FTYPE)
            array representing function coefficients
        dof_map : array(shape=(N,dim+1), dtype=FTYPE)
            corresponding mesh
        p : float/int >= 0, or 'inf'
            Order of norm, only p=2 is guaranteed to be implemented

        Output:
        -------
        v : float
            p-norm (or pseudo-norm) of the function represented by the x on
            dof_map

        '''
        if p == 2:
            return max(self.inner(x, x, dof_map), 0) ** .5
        else:
            raise NotImplementedError

    def _old2new(self, new_arr, new_DM, old_arr, old_DM):
        '''
        self._old2new(new_arr, new_DM, old_arr, old_DM)

        Cast a function from an old mesh to a new one

        Inputs:
        -------
        new_arr : array(shape=(N,k), dtype=FTYPE)
            empty storage for new coefficients
        new_DM : array(shape=(N,k), dtype=FTYPE)
            dof map for new coefficients
        old_arr : array(shape=(M,k), dtype=FTYPE)
            old coefficients
        old_DM : array(shape=(M,k), dtype=FTYPE)
            dof map for old coefficients

        new_arr is modified inplace to the projection of the old function onto
        the new discretisation.
        '''
# Case 1: x = y, new = old
#     x[j]=y[j], h=h, every j, increment i
# Case 2: x refines y, new[:2**dim] = old
#     y[j]<=x[j]<y[j]+h, every j, increment i
# Case 3: x coarsens y, new=0
#     y[j]+h<=x[j], some j, increment I
        raise NotImplementedError


class sparse_FS(sparse_mesh, adaptive_FS):

    def __init__(self, init=0, dim=1, coarse_level=5, _parent=None):
        sparse_mesh.__init__(self, init, dim, coarse_level, _parent)

    def one(self, v=1, maps=None):
        x = np.empty((self.size if maps is None else maps[0].shape[0], 1), dtype=FTYPE)
        x.fill(v)
        return sparse_function(x, self, maps)

    def _integrate(self, x, dof_map):
        return x.ravel().dot(dof_map[:, -1] ** self.dim)

    def _inner(self, x, y, dof_map):
        return self._integrate(x * y, dof_map)

    def _norm(self, x, dof_map, p):
        if p == 1:
            return self._integrate(abs(x), dof_map)
        elif p == 'inf':
            return abs(x).max()
        elif p == 0:
            return self._integrate((x != 0).astype(x.dtype), dof_map)
        else:
            return self._integrate(abs(x) ** p, dof_map) ** (1 / p)

    def _old2new(self, new_arr, new_DM, old_arr, old_DM):
        if self.dim == 1:
            f = _old2new_mesh1
        elif self.dim == 2:
            f = _old2new_mesh2
        else:
            raise NotImplementedError
#         print(new_arr)
#         print(new_DM)
#         print(old_arr)
#         print(old_DM)
        f(new_arr.ravel(), new_DM, old_arr.ravel(), old_DM)


class Haar_FS(sparse_tree, adaptive_FS):

    def __init__(self, init=0, dim=1, coarse_level=3, _parent=None):
        sparse_tree.__init__(self, init - 1, dim, coarse_level, _parent)

    @property
    def maps(self): return (self.dof_map, self.coarse_map, self.age, self.isleaf)

    @maps.setter
    def maps(self, value):
        if hasattr(value, 'dof_map'):
            self.dof_map, self.coarse_map = value.dof_map, value.coarse_map
            self.age, self.isleaf = value.age, value.isleaf
        else:
            self.dof_map, self.coarse_map, self.age, self.isleaf = value

    def one(self, v=1, maps=None):
        if maps is None:
            sz = (self.size + 1, 2 ** self.dim - 1)
        else:
            sz = (maps[0].shape[0] + 1, 2 ** self.dim - 1)
        x = np.zeros(sz, dtype=FTYPE)
        x[0, 0] = v
        return Haar_function(x, self, maps=maps)

    def _integrate(self, x, dof_map): return x[0, 0]

    def _inner(self, x, y, dof_map):
        x, y = x.ravel(), y.ravel()
        n = 2 ** self.dim - 1  # first few entries are not wavelets
        return x[0] * y[0] + x[n:].dot(y[n:])  # orthogonal basis

    def _old2new(self, new_arr, new_DM, old_arr, old_DM):
        if self.dim == 1:
            f = _old2new_tree1
        elif self.dim == 2:
            f = _old2new_tree2
        else:
            raise NotImplementedError
        new_arr[0] = old_arr[0]
        f(new_arr[1:], new_DM, old_arr[1:], old_DM)


pass  # Adaptive functions


class adaptive_func:  # odl DiscretizedSpaceElement

    def __init__(self, arr, FS, maps=None):
        self.maps = FS if maps is None else maps
        self.x = arr
        self.FS = FS
        assert self.x.shape[0] == self.dof_map.shape[0] + 1

    def asarray(self): return self.x

    def ravel(self): return self.x.ravel()

    @property
    def size(self): return self.x.size

    @property
    def shape(self): return self.x.shape

    @property
    def maps(self): return (self.dof_map, self.coarse_map, self.age)

    @maps.setter
    def maps(self, value):
        if hasattr(value, 'dof_map'):
            self.dof_map, self.coarse_map, self.age = value.dof_map, value.coarse_map, value.age
        else:
            self.dof_map, self.coarse_map, self.age = value

    def update(self, x=None, FS=None):
        '''
        self.update(x=None, FS=None)

        Inplace update of function values and/or the discretisation

        Inputs:
        -------
        x : array(shape=self.shape) or None
            Optional, default None. New function values to take.
            If None then re-discretise the original function
        FS : adaptive_FS instance or None
            Optional, default None. New mesh to project onto
            If None then just update function value
        '''
        if FS is None:
            if x is not None:
                self.x = x.reshape(self.x.shape)
            return
        elif x is None:
            x = FS.old2new(self.x, self.dof_map, copy=False)
        else:
            assert x.shape[0] - FS.size == self.x.shape[0] - self.dof_map.shape[0]
        self.x, self.FS = x, FS
        self.maps = FS

    def inner(self, y):
        '''
        v = self.inner(y)

        Inner product with function/array

        Input:
        ------
        y : adaptive_func instance or array(shape=self.shape)

        Output:
        -------
        v : float
            inner product between self and y.
        '''
        if hasattr(y, 'dof_map'):
            if self.dof_map is y.dof_map:
                return self.FS._inner(self.x, y.x, self.dof_map)
            else:
                # TODO: this is a weird default behaviour
                x = self.FS.old2new(self.x, self.dof_map, copy=False)
                y = self.FS.old2new(y.x, y.dof_map, copy=False)
                return self.FS._inner(x, y, self.FS.dof_map)
        else:
            assert y.size == self.size
            return self.FS._inner(self.x, y, self.dof_map)

    def __add__(self, other):
        if self.dof_map is other.dof_map:
            return self.copy(self.x + other.x)
        elif not (self.FS is other.FS):
            raise NotImplementedError
        else:
            FS = self.FS
            return self._copy(FS.old2new(self.x, self.dof_map, copy=False) +
                              FS.old2new(other.x, other.dof_map, copy=False),
                              FS, None)

    def __sub__(self, other):
        if self.dof_map is other.dof_map:
            return self.copy(self.x - other.x)
        elif not (self.FS is other.FS):
            raise NotImplementedError
        else:
            FS = self.FS
            return self._copy(FS.old2new(self.x, self.dof_map, copy=False) -
                              FS.old2new(other.x, other.dof_map, copy=False),
                              FS, None)

    def __mul__(self, other):
        assert np.isscalar(other)
        return self.copy(self.x * other)

    def __truediv__(self, other): return self * (1 / other)

    def __rtruediv__(self, other): raise NotImplementedError

    def __iadd__(self, other):
        if self.dof_map is other.dof_map:
            self.x += other.x
        elif not (self.FS is other.FS):
            raise NotImplementedError
        else:
            FS = self.FS
            self.update(FS.old2new(self.x, self.dof_map, copy=False)
                       +FS.old2new(other.x, other.dof_map, copy=False),
                        FS)

    def __isub__(self, other):
        if self.dof_map is other.dof_map:
            self.x -= other.x
        elif not (self.FS is other.FS):
            raise NotImplementedError
        else:
            FS = self.FS
            self.update(FS.old2new(self.x, self.dof_map, copy=False)
                       -FS.old2new(other.x, other.dof_map, copy=False),
                        FS)

    def __imul__(self, other):
        assert np.isscalar(other)
        self.x *= other

    def __itruediv__(self, other): self *= (1 / other)

    def __neg__(self): return self.copy(-self.x)

# TODO: representation of the 1 function
#     __radd__ = __add__
#     __rsub__ = __sub__
    __rmul__ = __mul__

    def discretise(self, level):
        '''
        arr = self.discretise(level)

        Computes uniformly discretised array at given resolution

        Input:
        ------
        level : int
            Request for resolution 2^{-level}

        Output:
        -------
        arr : array(shape=(2^level,...,2^level), ndim=dim, dtype=FTYPE)
            Pixel values of arr are exact average values of self on that pixel.
            In 1D:
                arr[i] = 2^level \int_{i*2^{-level}}^{(i+1)*2^{-level}} self(x) dx
        '''
        raise NotImplementedError

    def from_discrete(self, arr):
        '''
        self.from_discrete(arr)

        Computes projection of the given array onto this function space

        Input:
        ------
        arr : array(shape=(2^L,...,2^L), ndim=dim)
            Interpreted as a uniform discretisation of piece-wise constant
            intensity

        After function is called, self represents the projection of arr onto
        the current mesh.
        '''
        raise NotImplementedError

    def plot(self, level, ax=None, update=None, **kwargs):
        '''
        out = self.plot(level, ax=None, update=None, ...)

        Plotting functionality

        Inputs:
        -------
        level : int
            Plot at resolution 2^{-level}
        ax : matplotlib axis instance or None
            axis on which to plot (default matplotlib.gca() )
        update : None or 'out' from previous call
            Updating a plot is much faster than recreating the whole plot.
            Usage is:
                out = self.plot(*args, update=None, **kwargs)
                self.plot(*args, update=out, **kwargs)

        Output:
        -------
        out : matplotlib objects
            Intended for use with the update functionality, see above.
        '''
        raise NotImplementedError

    def _copy(self, arr, FS, maps):
        '''
        new = self._copy(arr, FS, maps)

        Utility function to make sure that new instance is of the same function
        class as self. Arguments are the same as adaptive_func.__init__.
        '''
        return adaptive_func(arr, FS, maps)

    def copy(self, like=None):
        '''
        new = self.copy(like=None)

        Create a deep copy of self, possibly representing new function values
        given by like.

        Input:
        ------
        like : None or array(shape=self.x.shape)
                or array(shape=(self.FS.size, self.x.shape[1]))
            Optional argument, if None then returns a deep copy of self.
            If provided then returns a new function with same function space
            as self.

        Output:
        -------
        new : adaptive_func instance
            Object with same class and function space as self. If possible
            then also same dof_map.
        '''
        # TODO: this looks a bit weird now, is it the function or space that copies?
        if like is None:
            arr = self.x.copy()
        elif np.isscalar(like):
            return self.FS.one(like, self.maps)
        elif like.size == self.x.size:
            arr = like.reshape(self.x.shape)
        else:
            return self._copy(like.reshape(self.FS.size, -1), self.FS, None)
        return self._copy(arr, self.FS, self.maps)


class sparse_func(adaptive_func):

    def __init__(self, arr, FS, maps=None):
        if np.isscalar(arr):
            tmp = np.empty((FS.size, 1), dtype=FTYPE)
            tmp.fill(arr)
            arr = tmp
        adaptive_func.__init__(self, arr, FS, maps)


class sparse_func1D(sparse_func):

    def __init__(self, arr, FS, maps=None):
        assert FS.dim == 1
        sparse_func.__init__(self, arr, FS, maps)

    def discretise(self, level, pad=False):
        '''
        arr = self.discretise(level, pad=False)
        See adaptive_func.discretise for the main functionality.
        Extra input:
        ------------
        pad : bool
            If True then an extra value is appended to arr such that
            arr.size = 2^level+1 and arr[-1]==arr[-2].
        '''
        # The mesh contains pixels of sizes 1, .5, .25,...,2^{-level},...
        # To discretise at resolution 2^{-level}, we first compute
        # discretisations at each intermediate level. We start with all pixels
        # smaller or equal to 2^{-level} then incrementally 'filter' out the
        # larger and larger pixels. Once we have uniform discretisations of
        # size 1, 2, 4, ..., 2^{level}, they are 'upscaled' and summed to make
        # a single uniform discretisation

        X, DM = self.x.reshape(-1), self.dof_map
        # preallocate storage for each level of discretisation
        img = [np.zeros(2 ** L, dtype=FTYPE) for L in range(level)]
        img.append(np.zeros(2 ** level + (1 if pad else 0), dtype=FTYPE))

        for L in range(level, -1, -1):  # start with fine pixels and proceed to coarse
            # initialise buffers to store all pixels size > 2^{-L}
            x, dm = np.empty(X.shape, dtype=FTYPE), np.empty(DM.shape, dtype=FTYPE)
            # pre-filtering: X contains all pixels to be discretised
            # post-filtering: x[:sz] contains all pixels which have not yet
            #     been discretised (size >2^{-L})
            #     img[L] is the discretisation of pixels size 2^{-L}
            sz = _sparse2discrete1D(X, DM, x, dm, img[L], 2.** -L)
            # cut off excess buffer for next iteration
            X, DM = x[:sz], dm[:sz]

        for L in range(level):
            # img[L] is the discretisation of pixels >= 2^{-L}
            # img[L+1] is the discretisation of pixels = 2^{-L-1}
            _upscale_sparse1D(img[L], img[L + 1])
            # img[L+1] is the discretisation of pixels >= 2^{-L-1}

        if pad:
            img[-1][-1] = img[-1][-2]
        return img[-1]

    def from_discrete(self, img):
        level = round(np.log2(img.size))
        assert img.size == 2 ** level  # power 2 array size

        x = np.empty(self.x.size, dtype=FTYPE)
        _from_sparse1D(x, self.dof_map, 2.** -level, img)

        return self.copy(x)

    def plot(self, level, ax=None, update=None, max_ticks=500, mass=False, background=0, **kwargs):
        '''
        out = self.plot(level, ax=None, update=None,
                max_ticks=500, mass=False, background=0, **kwargs)

        See adaptive_func.plot for the main functionality.
        Extra inputs:
        -------------
        max_ticks : int
            Maximum number of cells to mark on the x-axis
        mass : bool
            If True then scales such that sum of plotted values is equal to
            integral of function, default is to plot function values.
            This is useful when trying to plot a Dirac delta.
        background : float
            Plots self + background, default is background=0

        kwargs are passed as key-word arguments to matplotlib.pyplot.step
        '''
        y = self.discretise(level, pad=True) + background
        if mass:
            y *= 2.** -level
        x = np.linspace(0, 1, 2 ** level + 1)
        if update is not None:
            update.set_ydata(y)
            if np.isscalar(max_ticks) and max_ticks >= 1:
                update.axes.xaxis.set_minor_locator(FixedLocator(self.dof_map[::max(1, self.size // max_ticks), 0]))
                update.axes.xaxis.set_tick_params(which='minor', length=8, direction='inout')
            return update
        ax = ax if ax is not None else plt.gca()
        if np.isscalar(max_ticks) and max_ticks >= 1:
            ax.xaxis.set_minor_locator(FixedLocator(self.dof_map[::max(1, self.size // max_ticks), 0]))
            ax.xaxis.set_tick_params(which='minor', length=8, direction='inout')
        ax.set_xlim(0, 1)
        tmp = ax.step(x, y, where='post', **kwargs)
        return tmp[0]

    def _copy(self, arr, FS, maps): return sparse_func1D(arr, FS, maps)


class sparse_func2D(sparse_func):

    def __init__(self, arr, FS, maps=None):
        assert FS.dim == 2
        sparse_func.__init__(self, arr, FS, maps)

    def discretise(self, level):
        # The mesh contains pixels of sizes 1, .5, .25,...,2^{-level},...
        # To discretise at resolution 2^{-level}, we first compute
        # discretisations at each intermediate level. We start with all pixels
        # smaller or equal to 2^{-level} then incrementally 'filter' out the
        # larger and larger pixels. Once we have uniform discretisations of
        # size 1, 2, 4, ..., 2^{level}, they are 'upscaled' and summed to make
        # a single uniform discretisation

        X, DM = self.x.reshape(-1), self.dof_map
        # preallocate storage for each level of discretisation
        img = [np.zeros((2 ** L,) * 2, dtype=FTYPE) for L in range(level + 1)]

        for L in range(level, -1, -1):  # start with fine pixels and proceed to coarse
            # initialise buffers to store all pixels size > 2^{-L}
            x, dm = np.empty(X.shape, dtype=FTYPE), np.empty(DM.shape, dtype=FTYPE)
            # pre-filtering: X contains all pixels to be discretised
            # post-filtering: x[:sz] contains all pixels which have not yet
            #     been discretised (size >2^{-L})
            #     img[L] is the discretisation of pixels size 2^{-L}
            sz = _filter_sparse2D(X, DM, x, dm, img[L], 2.** -L)
            # cut off excess buffer for next iteration
            X, DM = x[:sz], dm[:sz]

        for L in range(level):
            # img[L] is the discretisation of pixels >= 2^{-L}
            # img[L+1] is the discretisation of pixels = 2^{-L-1}
            _upscale_sparse2D(img[L], img[L + 1])
            # img[L+1] is the discretisation of pixels >= 2^{-L-1}

        return img[-1]

    def from_discrete(self, img):
        assert img.shape[0] == img.shape[1]  # square array
        level = int(np.log2(img.shape[0]).round())
        assert img.shape[0] == 2 ** level  # power 2 array size

        x = np.empty(self.x.size, dtype=FTYPE)
        _from_sparse2D(x, self.dof_map, 2. ** -level, img)

        return self.copy(x)

    def plot(self, level, ax=None, update=None, mass=False,
             scale=None, background=0, extent=[0, 1, 0, 1], **kwargs):
        '''
        out = self.plot(level, ax=None, update=None,
                mass=False, scale=None, background=0, **kwargs)

        See adaptive_func.plot for the main functionality.
        Extra inputs:
        -------------
        mass : bool
            If True then scales such that sum of plotted values is equal to
            integral of function, default is to plot function values.
            This is useful when trying to plot a Dirac delta.
        scale : None or 'log'
            If 'log' then plots intensities in an (un-signed) log scale.
        background : float
            Plots self + background, default is background=0

        kwargs are passed as key-word arguments to matplotlib.pyplot.imshow
        '''
        y = self.discretise(level) + background
        if mass:
            y *= 2. ** -(2 * level)
        if scale == 'log':
            y = np.log10(abs(y) + 1e-10 * max(1e-6, y.max()))
        if update is not None:
            return update.set_data(y)
        ax = ax if ax is not None else plt.gca()
        return ax.imshow(y, origin='lower', extent=extent, **kwargs)

    def _copy(self, arr, FS, maps): return sparse_func2D(arr, FS, maps)


def sparse_function(arr, FS, *args, **kwargs):
    '''
    u = sparse_function()
    u.dof_map[i] = (x,y,...,h) is positions and size of pixel[i]
    u.x[i] is amplitude of function on pixel[i], not mass
    '''
    if FS.dim == 1:
        return sparse_func1D(arr, FS, *args, **kwargs)
    elif FS.dim == 2:
        return sparse_func2D(arr, FS, *args, **kwargs)
    else:
        raise NotImplementedError


class wave_func(adaptive_func):

    def __init__(self, arr, FS, maps=None):
        if np.isscalar(arr):
            tmp = np.zeros((FS.size + 1, 2 ** FS.dim - 1), dtype=FTYPE)
            tmp[0, 0] = arr
            arr = tmp
        adaptive_func.__init__(self, arr, FS, maps)

    @property
    def maps(self): return (self.dof_map, self.coarse_map, self.age, self.isleaf)

    @maps.setter
    def maps(self, value):
        if hasattr(value, 'dof_map'):
            self.dof_map, self.coarse_map = value.dof_map, value.coarse_map
            self.age, self.isleaf = value.age, value.isleaf
        else:
            self.dof_map, self.coarse_map, self.age, self.isleaf = value


class Haar_func1D(wave_func):

    def __init__(self, arr, FS, maps=None):
        assert FS.dim == 1
        wave_func.__init__(self, arr, FS, maps)

    def discretise(self, level, pad=False):
        '''
        arr = self.discretise(level, pad=False)
        See adaptive_func.discretise for the main functionality.
        Extra input:
        ------------
        pad : bool
            If True then an extra value is appended to arr such that
            arr.size = 2^level+1 and arr[-1]==arr[-2].
        '''
        out = np.empty(2 ** level + (1 if pad else 0), dtype='float64')
        x = self.x.reshape(-1)

        _discretise_Haar1D(x[0], x[1:], self.dof_map, 1, out[:2 ** level])
        if pad:
            out[-1] = out[-2]
        return out

    def from_discrete(self, img):
        level = round(np.log2(img.size))
        assert img.size == 2 ** level

        x = np.empty(self.x.shape, dtype='float64')
        x[0, 0] = _from_discrete_Haar1D(img, self.dof_map, 1, x[1:, 0])

        return self.copy(x)

    def plot(self, level, ax=None, update=None, **kwargs):
        '''
        out = self.plot(level, ax=None, update=None,
                max_ticks=500, mass=False, background=0, **kwargs)

        See adaptive_func.plot for the main functionality.
        Extra inputs:
        -------------
        max_ticks : int
            Maximum number of cells to mark on the x-axis
        mass : bool
            If True then scales such that sum of plotted values is equal to
            integral of function, default is to plot function values.
            This is useful when trying to plot a Dirac delta.
        background : float
            Plots self + background, default is background=0

        kwargs are passed as key-word arguments to matplotlib.pyplot.step
        '''
        y = self.discretise(level, pad=True)
        x = np.linspace(0, 1, 2 ** level + 1)
        if update:
            return update.set_data(x, y)
        ax = ax if ax is not None else plt
        return ax.step(x, y, where='post', **kwargs)[0]

    def plot_tree(self, ax=None, update=None, **kwargs):
        ax = ax if ax is not None else plt.gca()
        y = self.x[1:].ravel()
        eps = abs(y).max() * 1e-8

        if update is not None:  # not yet implemented
            ax = update
            ax.clear()

        # place in middle of support (at jump)
        x = self.dof_map[:, 0] + .5 * self.dof_map[:, 1]
        x[0] = 0

        # do plotting
        I = [i for i in range(y.size) if abs(y[i]) > eps]
        lines = ax.plot(x[I], y[I], '*', **kwargs)
        I = [i for i in range(y.size) if abs(y[i]) <= eps]
        lines += ax.plot(x[I], y[I], '.', **kwargs)

        ax.set_xlim(-0.01, 1)
        ax.set_xlabel('Jump location')
        ax.set_ylabel('Coefficient value')
        return ax

    def _copy(self, arr, FS, maps): return Haar_func1D(arr, FS, maps)


class Haar_func2D(wave_func):

    def __init__(self, arr, FS, maps=None):
        assert FS.dim == 2
        wave_func.__init__(self, arr, FS, maps)

        # TODO: half of this is bollocks
        '''
        dof_map[d] = (level, index0, index1, index2)
        arr[d] = (v0, v1, v2)

        Define the three base wavelets:

                -1/2     x\in[0,1], y\in[0,2]   |             -1/2     x\in[0,2], y\in[0,1]
    W_0(x, y) = +1/2     x\in[1,2], y\in[0,2],  |  W_1(x,y) = +1/2     x\in[0,2], y\in[1,2]
                 0              else            |              0              else
    ---------------------------------------------------------------------------------------
                -1/2     x\in[0,1], y\in[1,2]    or    x\in[1,2], y\in[0,1]
    W_2(x, y) = +1/2     x\in[0,1], y\in[0,1]    or    x\in[1,2], y\in[1,2]
                 0              else

    f(x) = arr[0] + sum_{d,i} arr[d,i] * 2^d.level * W_i(2^d.level * x-d.index[i])


                                    |---------------------------|
                                    |             |             |
                                    |  -v0+v1-v2  |  +v0+v1+v2  |
    2*(v0*W_0 + v1*W_1 = v2*W_2) =  |             |             |
                                    |---------------------------|
                                    |             |             |
                                    |  -v0-v1+v2  |  +v0-v1-v2  |
                                    |             |             |
                                    |---------------------------|
        '''

    def discretise(self, level):
        X, DM = self.x[1:], self.dof_map
        # preallocate storage for each level of discretisation
        img = [np.zeros((2 ** L,) * 2, dtype=FTYPE) for L in range(level + 1)]
        for L in range(level, 0, -1):  # start with fine pixels and proceed to coarse
            # initialise buffers to store all pixels size > 2^{-L}
            x, dm = np.empty(X.shape, dtype=FTYPE), np.empty(DM.shape, dtype=FTYPE)
            # pre-filtering: X contains all pixels to be discretised
            # post-filtering: x[:sz] contains all pixels which have not yet
            #     been discretised (size >2^{-L})
            #     img[L] is the discretisation of pixels size 2^{-L}
            sz = _filter_Haar2D(X, DM, x, dm, img[L], 2.** (1 - L))
            # cut off excess buffer for next iteration
            X, DM = x[:sz], dm[:sz]
        img[0][0, 0] = self.x[0, 0]

        for L in range(level):
            # img[L] is the discretisation of pixels >= 2^{-L}
            # img[L+1] is the discretisation of pixels = 2^{-L-1}
            _upscale_sparse2D(img[L], img[L + 1])
            # img[L+1] is the discretisation of pixels >= 2^{-L-1}

#         out = np.empty((2 ** level,) * 2, dtype='float64')
#         _discretise_Haar2D(self.x[0, 0], self.x[1:], self.dof_map, 1, out)
        return img[-1]

    def from_discrete(self, arr):
        assert arr.shape[0] == arr.shape[1]  # square input
        level = int(np.log2(arr.shape[0]).round())
        assert arr.shape[0] == 2 ** level  # power of 2 size

        img = [np.empty((2 ** L,) * 2, dtype=FTYPE) for L in range(level)] + [np.require(arr, requirements='C')]
        for L in range(level, 0, -1):  # start with fine pixels and proceed to coarse
            _downsample_2D(img[L], img[L - 1])

        x = np.empty(self.x.shape, dtype='float64')
        x[0, 0] = img[0][0, 0]
        x[0, 1:] = 0
        _from_discrete_Haar2D(List(img[1:]), self.dof_map, -log2(self.dof_map[:, 2]).astype('int32'), x[1:])

        return self.copy(x)

    def plot(self, level, ax=None, update=None, extent=[0, 1, 0, 1], **kwargs):
        y = self.discretise(level)
        if update:
            return update.set_data(y)
        ax = ax if ax is not None else plt.gca()
        return ax.imshow(y, origin='lower', extent=extent, **kwargs)

    def plot_tree(self, ax=None, update=None, level=5, **kwargs):
        ax = ax if ax is not None else plt.gca()
        y = self.x[1:]
        eps = abs(y).max() * 1e-10

        # place in middle of support (at jump)
        x = self.dof_map[:, 1::-1] + .5 * self.dof_map[:, -1:]

        # do plotting
        lines = []
        tmp, eps = np.log(abs(y).max(1) + eps), np.log(eps)
        if tmp.ptp() > 1e-15:
            tmp, eps = (tmp - tmp.min()) / tmp.ptp(), (eps - tmp.min()) / tmp.ptp()
        else:
            tmp *= 0; eps = 0 * eps + 1
        I = [i for i in range(tmp.size) if tmp[i] <= eps]
        L = ax.hist2d(x[:, 1], x[:, 0], range=[[0, 1], [0, 1]],
                               bins=min(100, 1 + int(x.shape[0] ** .5 / 10)), density=True)
        ax.clear()
        lines.append(ax.imshow(L[0], origin='lower', interpolation='bicubic',
                               cmap='Blues', vmin=0, alpha=.5,
                               extent=[0, 1, 0, 1]))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = {'red':((0, .1, .1), (1, 1, 1)),
                'blue':((0, .1, .1), (1, 0, 0)),
                'green':((0, .1, .1), (1, 0, 0))}
        cmap = LinearSegmentedColormap('k2r', cmap)
        if y.shape[0] < 50 ** 2:
            lines.append(ax.scatter(x[I, 1], x[I, 0], 2, tmp[I], '.', cmap=cmap, **kwargs))
        I = (tmp > eps) * (self.dof_map[:, 2] > 2 ** -level)
#         [i for i in range(tmp.size) if tmp[i] > eps]
        lines.append(ax.scatter(x[I, 0], x[I, 1], 40, tmp[I], '*', cmap=cmap, **kwargs))

        ax.set_aspect('equal', 'box')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Jump location x')
        ax.set_ylabel('Jump location y')
        ax.set_title('Log. intensity in red, density in blue')

        return lines

    def _copy(self, arr, FS, maps): return Haar_func2D(arr, FS, maps)


def Haar_function(arr, FS, *args, **kwargs):
    if FS.dim == 1:
        return Haar_func1D(arr, FS, *args, **kwargs)
    elif FS.dim == 2:
        return Haar_func2D(arr, FS, *args, **kwargs)
    else:
        raise NotImplementedError


pass  # Compiled functions for meshes


@jit(**__params)
def _refine_mesh(new, old, indcs, c):
    '''
    Refining a mesh, if a pixel is refined then the old one is discarded
    and if a pixel is coarsened then it is also discarded.
    '''
    j = 0
    for i in range(indcs.size):
        I = indcs[i]
        if I == 0:  # only keep these pixels
            for k in range(old.shape[1]):
                new[j, k] = old[i, k]
            j += 1
        elif I > 0:
            h = .5 * old[i, c.shape[1]]
            for k0 in range(c.shape[0]):  # loop over new pixels
                for k1 in range(c.shape[1]):  # loop over dimensions
                    new[j, k1] = old[i, k1] + h * c[k0, k1]
                new[j, c.shape[1]] = h
                j += 1
        i += 1
    return j


@jit(**__params)
def _refine_tree(new, n_isleaf, old, o_isleaf, indcs, c):
    '''
    Refining a tree, if a pixel is refined then the old one is kept.
    Coarsening is not yet supported.
    '''
    # TODO: implement tree coarsening
    j = 0
    for i in range(indcs.size):
        I = indcs[i]
        if I <= 0:  # ignore request to coarsen
            new[j,:] = old[i,:]
            n_isleaf[j] = o_isleaf[i]
            j += 1
        elif I > 0:
            # keep old pixel
            new[j,:] = old[i,:]
            n_isleaf[j] = False
            j += 1
            if o_isleaf[i]:  # perform refinement
                h = .5 * old[i, c.shape[1]]
                for k0 in range(c.shape[0]):  # loop over new pixels
                    for k1 in range(c.shape[1]):  # loop over dimensions
                        new[j, k1] = old[i, k1] + h * c[k0, k1]
                    new[j, c.shape[1]] = h
                    n_isleaf[j] = True
                    j += 1
    return j


@jit(**__params)
def _fine2coarse1(dof_map, CM, dx):
    '''
    Quite naive computation of inverse of dof_map
    '''
    for i in range(dof_map.shape[0]):  # for each pixel in the domain
        x0, h = dof_map[i, 0], dof_map[i, 1]  # compute bounding box
        x1 = x0 + h
        jx = int(x0 / dx)  # compute first index in CM
        x = jx * dx
        while True:  # For each jx such that CM[jx] intersects [x0,x1]
            CM[jx, 0] = min(CM[jx, 0], i)
            CM[jx, 1] = max(CM[jx, 1], i + 1)
            x += dx; jx += 1
            if x >= x1:
                break


@jit(**__params)
def _fine2coarse2(dof_map, CM, dx):
    for i in range(dof_map.shape[0]):  # for each pixel in the domain
        x0, y0, h = dof_map[i, 0], dof_map[i, 1], dof_map[i, 2]
        x1, y1 = x0 + h, y0 + h  # compute the bounding box
        jx, jy = int(x0 / dx), int(y0 / dx)
        x, y = jx * dx, jy * dx
        while True:  # For each j such that CM[j] intersects [x0,x1]x[y0,y1]
            CM[jx, jy, 0] = min(CM[jx, jy, 0], i)
            CM[jx, jy, 1] = max(CM[jx, jy, 1], i + 1)
            x += dx;  jx += 1
            if x >= x1:
                y += dx; jy += 1
                if y >= y1:
                    break
                jx = int(x0 / dx)
                x = jx * dx


@jit(**__pparams)
def _isleaf(dof_map, arr):
    dim = dof_map.shape[1] - 1
    for i in prange(dof_map.shape[0] - 1):
        b = dof_map[i, 0] == dof_map[i + 1, 0]
        for j in range(1, dim):
            b = b and (dof_map[i, j] == dof_map[i + 1, j])
        arr[i] = not b
    arr[dof_map.shape[0] - 1] = True


pass  # Compiled functions for re-meshing


@jit(**__params)
def _old2new_mesh1(new, new_DM, old, old_DM):
    i, I = 0, 0
    while i < new_DM.shape[0] and I < old_DM.shape[0]:
        x, y = new_DM[i], old_DM[I]
        if x[0] >= y[0] + y[1]:  # new box too far ahead
            I += 1
        elif x[0] + x[1] <= y[0]:  # old box too far ahead
#             new[i] = 0
            i += 1
        elif x[1] <= y[2]:  # new contained in old
            new[i] = old[I]  # transfer average values
            i += 1
        else:  # new contains old
            new[i] += old[I] * (y[1] / x[1])  # transfer mass
            I += 1


@jit(**__params)
def _old2new_mesh2(new, new_DM, old, old_DM):
    i, I = 0, 0
    while i < new.size and I < old.size:
        x, y = new_DM[i], old_DM[I]
        s = 1 / max(x[2], y[2])
        j = interleave2(int(x[0] * s), int(x[1] * s))
        J = interleave2(int(y[0] * s), int(y[1] * s))
        if j > J:  # new box too far ahead
            I += 1
        elif j < J:  # old box too far ahead
#             new[i] = 0
            i += 1
        elif x[2] <= y[2]:  # new contained in old
            new[i] = old[I]
            i += 1
        else:  # new contains old
            new[i] += old[I] * (y[2] * s) ** 2
            I += 1


@jit(**__params)
def _old2new_tree1(new, new_DM, old, old_DM):
    i, I = 0, 0
    while i < new_DM.shape[0] and I < old_DM.shape[0]:
        x, y = new_DM[i], old_DM[I]
        if x[0] >= y[0] + y[1]:  # new box too far ahead
            I += 1
        elif x[0] + x[1] <= y[0]:  # old box too far ahead
#             new[i] = 0
            i += 1
        else:  # new equals old
            new[i] = old[I]
            i += 1; I += 1


@jit(**__params)
def _old2new_tree2(new, new_DM, old, old_DM):
    i, I = 0, 0
    while i < new_DM.shape[0] and I < old_DM.shape[0]:
        x, y = new_DM[i], old_DM[I]
        s = 1 / max(x[2], y[2])
        j = interleave2(int(x[0] * s), int(x[1] * s))
        J = interleave2(int(y[0] * s), int(y[1] * s))
        if j > J:  # new box too far ahead
            I += 1
        elif j < J:  # old box too far ahead
#             new[i] = 0
            i += 1
        else:  # new equals old
            new[i] = old[I]
            i += 1; I += 1


pass  # Compiled functions for parsing arrays


@jit(**__params)
def _square_intersect1D(x0, x1, px0, px1):
    '''
    Length of intersection of [x0,x1] and [px0,px1]
    '''
    return max(0, min(x1, px1) - max(x0, px0))


@jit(**__params)
def _square_intersect2D(x0, y0, x1, y1, px0, py0, px1, py1):
    '''
    Area of intersection of [x0,x1]x[y0,y1] and [px0,px1]x[py0,py1]
    '''
    return _square_intersect1D(x0, x1, px0, px1) * _square_intersect1D(y0, y1, py0, py1)


@jit(**__params)
def _sparse2discrete1D(old, old_DM, new, new_DM, coarse, s):
    i, t = 0, 1 / s
    for j in range(old.size):  # for each mesh point
        if old_DM[j, 1] <= s:
            # if mesh size smaller than s then discretise by average intensity
            i0 = int(old_DM[j, 0] * t)
            coarse[i0] += old[j] * (old_DM[j, 1] * t)
        else:
            # if mesh size larger than s then copy to new list of cells
            # to add to discretisation later
            new[i] = old[j]
            new_DM[i, 0], new_DM[i, 1] = old_DM[j, 0], old_DM[j, 1]
            i += 1
    return i


@jit(**__pparams)
def _upscale_sparse1D(coarse, fine):
    # coarse.shape = n
    # fine.shape = 2n
    # coarse and fine both store average values per pixel so they sum directly
    for I in prange(coarse.shape[0]):
        fine[2 * I] += coarse[I]
        fine[2 * I + 1] += coarse[I]


@jit(**__pparams)
def _from_sparse1D(arr, dof_map, dx, img):
    for j in prange(arr.size):  # for each mesh point
        # current cell
        x0 = dof_map[j, 0]
        x1 = x0 + dof_map[j, 1]
        scale = 1 / dof_map[j, 1]

        # cell covers pixels [im0,iM0]
        im0, iM0 = int(x0 / dx), int(x1 / dx)
        if iM0 * dx != x1:
            iM0 += 1  # current mesh-point ends half-way through a pixel
        iM0 = min(iM0, img.shape[0])

        arr[j] = 0
        for i0 in range(im0, iM0):  # for each pixel
            px = i0 * dx  # current pixel
            # compute overlap of cell and pixel
            v = _square_intersect1D(px, px + dx, x0, x1)
            # absorb the mass of pixel into this cell
            arr[j] += img[i0] * v * scale  # mean intensity


@jit(**__params)
def _filter_sparse2D(old, old_DM, new, new_DM, coarse, s):
    i, t = 0, 1 / s
    for j in range(old.size):  # for each mesh point
        if old_DM[j, 2] <= s:
            # if cell size smaller than s then discretise by average intensity
            i0, i1 = int(old_DM[j, 0] * t), int(old_DM[j, 1] * t)
            coarse[i0, i1] += old[j] * (old_DM[j, 2] * t) ** 2
        else:
            # if cell size larger than s then copy to new list of cells
            new[i] = old[j]
            new_DM[i] = old_DM[j]
            i += 1
    return i


@jit(**__pparams)
def _upscale_sparse2D(coarse, fine):
    # coarse.shape = (n,n)
    # fine.shape = (2n,2n)
    # coarse and fine both store average values per pixel so they sum directly
    for I in prange(coarse.shape[0]):
        for i in range(2 * I, 2 * I + 2):
            for j in range(coarse.shape[1]):
                # fine[2I:2I+2,2j:2j+2] = coarse[I,j]
                fine[i, 2 * j] += coarse[I, j]
                fine[i, 2 * j + 1] += coarse[I, j]


@jit(**__pparams)
def _from_sparse2D(arr, dof_map, dx, img):
    for j in prange(arr.size):  # for each mesh point
        # current cell
        x0, y0 = dof_map[j, 0], dof_map[j, 1]
        x1, y1 = x0 + dof_map[j, 2], y0 + dof_map[j, 2]
        scale = 1 / dof_map[j, 2] ** 2

        # cell covers pixels [im0,iM0]x[im1,iM1]

        im0, iM0 = int(x0 / dx), int(x1 / dx)
        if iM0 * dx != x1:
            iM0 += 1  # current mesh-point ends half-way through a pixel
        iM0 = min(iM0, img.shape[0])

        im1, iM1 = int(y0 / dx), int(y1 / dx)
        if iM1 * dx != y1:
            iM1 += 1  # current mesh-point ends half-way through a pixel
        iM1 = min(iM1, img.shape[1])

        arr[j] = 0
        for i0 in range(im0, iM0):  # for each x-pixel
            px, px1 = i0 * dx, (i0 + 1) * dx  # current x-pixel
            for i1 in range(im1, iM1):  # for each y-pixel
                py, py1 = i1 * dx, (i1 + 1) * dx  # current y-pixel
                # compute overlap of cell and pixel
                v = _square_intersect2D(px, py, px1, py1, x0, y0, x1, y1)
                # absorb the mass of pixel into this cell
                arr[j] += img[i0, i1] * v * scale  # mean intensity


@jit(**__cparams)
def _discretise_Haar1D(value, arr, dof_map, h, out):
    if arr.shape[0] == 0:
        out[:] = value
        return

    mid = dof_map[0, 0] + .5 * h
    i = 2  # skip root and first child
    while i < dof_map.shape[0]:
        if dof_map[i, 0] >= mid:
            break
        i += 1
    scale, N = h ** (-.5), out.shape[0] // 2
    _discretise_Haar1D(value - scale * arr[0], arr[1:i], dof_map[1:i], .5 * h, out[:N])
    _discretise_Haar1D(value + scale * arr[0], arr[i:], dof_map[i:], .5 * h, out[N:])


@jit(**__cparams)
def _from_discrete_Haar1D(arr, dof_map, h, out):
    if out.shape[0] == 0:  # pixels are smaller than wavelets
        value = 0
        for i in range(arr.size):
            value += arr[i]
        return value / arr.size * h  # integral on this cell
    elif arr.shape[0] == 1:
        out[:] = 0  # all smaller wavelets integrate to 0
        return arr[0] * h  # integral on this cell

    mid = dof_map[0, 0] + .5 * h
    i = 2  # skip root and first child
    while i < dof_map.shape[0]:
        if dof_map[i, 0] >= mid:
            break
        i += 1

    scale, N = h ** (-.5), arr.shape[0] // 2
    value0 = _from_discrete_Haar1D(arr[:N], dof_map[1:i], .5 * h, out[1:i])
    value1 = _from_discrete_Haar1D(arr[N:], dof_map[i:], .5 * h, out[i:])

    out[0] = scale * (-value0 + value1)
    return value0 + value1


@jit(**__params)
def _filter_Haar2D(old, old_DM, new, new_DM, coarse, s):
    i = 0; t = 1 / s; scale = 1 / s
    for j in range(old.shape[0]):
        h = old_DM[j, 2]
        if h > s:  # save for future filtering
            new[i] = old[j]
            new_DM[i] = old_DM[j]
            i += 1
        elif h == s:
            i0, i1 = int(old_DM[j, 0] * t), int(old_DM[j, 1] * t)
            coarse[2 * i0, 2 * i1] = scale * (-old[j, 0] - old[j, 1] + old[j, 2])
            coarse[2 * i0, 2 * i1 + 1] = scale * (-old[j, 0] + old[j, 1] - old[j, 2])
            coarse[2 * i0 + 1, 2 * i1] = scale * (+old[j, 0] - old[j, 1] - old[j, 2])
            coarse[2 * i0 + 1, 2 * i1 + 1] = scale * (+old[j, 0] + old[j, 1] + old[j, 2])
    return i


@jit(**__params)
def _downsample_2D(fine, coarse):
    for i in range(coarse.shape[0]):
        for j in range(coarse.shape[1]):
            coarse[i, j] = fine[2 * i, 2 * j] + fine[2 * i, 2 * j + 1]
        for j in range(coarse.shape[1]):
            coarse[i, j] += fine[2 * i + 1, 2 * j] + fine[2 * i + 1, 2 * j + 1]
            coarse[i, j] *= .25  # average value


@jit(**__cparams)
def _from_discrete_Haar2D(arr, dof_map, level, out):
    for i in range(dof_map.shape[0]):
        if level[i] >= len(arr):  # small wavelets are all 0
            out[i] = 0
        else:
            A = arr[level[i]]
            h = dof_map[i, 2]
            scale = .25 * h
            i0, i1 = int(dof_map[i, 0] / h), int(dof_map[i, 1] / h)
            v0 = A[2 * i0, 2 * i1]; v1 = A[2 * i0, 2 * i1 + 1]
            v2 = A[2 * i0 + 1, 2 * i1]; v3 = A[2 * i0 + 1, 2 * i1 + 1]

            out[i, 0] = (-v0 - v1 + v2 + v3) * scale
            out[i, 1] = (-v0 + v1 - v2 + v3) * scale
            out[i, 2] = (+v0 - v1 - v2 + v3) * scale


if __name__ == '__main__':
    test = 2.5
    if test == 1:  # sparse_mesh initiation and refining
        mesh = sparse_mesh(2, 1, 1)
        assert mesh.age == 2
        assert mesh.h == 1 / 4
        assert abs(4 * mesh.dof_map - (4 * mesh.dof_map).round()).max() < EPS
        assert all(tuple(mesh.coarse_map[i]) == j for i, j in enumerate(((0, 2), (2, 4))))
        mesh.update(2)
        assert all(tuple(mesh.coarse_map[i]) == (i, i + 1) for i in range(4))
        mesh.update(3)
        assert all(tuple(mesh.coarse_map[i]) == (i // 2, i // 2 + 1) for i in range(8))
        print('Mesh construction 1D checked')

        mesh = sparse_mesh(2, 2, 1)
        assert mesh.age == 2
        assert mesh.h == 1 / 4
        tmp = ([0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [3, 0], [2, 1], [3, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 2], [3, 2], [2, 3], [3, 3])
        assert all(abs(4 * mesh.dof_map[i,:2] - np.array(j)).max() < EPS for i, j in enumerate(tmp))
        assert all(tuple(mesh.coarse_map[i0, i1]) == (4 * j, 4 * j + 4) for j, (i0, i1) in enumerate(tmp[:4]))
        mesh.update(2)
        assert all(tuple(mesh.coarse_map[i0, i1]) == (j, j + 1) for j, (i0, i1) in enumerate(tmp))
        mesh.update(3)
        assert all(tuple(mesh.coarse_map[I0, I1]) == (j, j + 1) for j, (i0, i1) in enumerate(tmp) for I0 in (2 * i0, 2 * i0 + 1) for I1 in (2 * i1, 2 * i1 + 1))
        print('Mesh construction 2D checked')

        mesh = sparse_mesh(2, 1, 2)
        mesh.refine([1, 0, -1, 0])
        assert mesh.age == 3
        assert mesh.h == 1 / 8
        assert all(8 * mesh.dof_map[i, 0] == j for i, j in enumerate((0, 1, 2, 6)))
        assert all(tuple(mesh.coarse_map[i]) == j for i, j in enumerate(((0, 2), (2, 3), (4, -1), (3, 4))))
        print('Re-meshing 1D checked')

        mesh = sparse_mesh(1, 2, 2)
        mesh.refine([0, 1, 0, -1])
        assert mesh.age == 2
        assert mesh.h == 1 / 4
        assert all(abs(4 * mesh.dof_map[i,:2] - np.array(j)).max() < EPS for i, j in enumerate(
            ([0, 0], [2, 0], [3, 0], [2, 1], [3, 1], [0, 2])))
        tmp = ((slice(2), slice(2), 0, 1), (2, 0, 1, 2), (3, 0, 2, 3), (2, 1, 3, 4), (3, 1, 4, 5), (slice(2), slice(2, 4), 5, 6), (slice(2, 4), slice(2, 4), 6, -1))
        assert all(abs(mesh.coarse_map[x[:2]][..., 0] - x[2]).max() + abs(mesh.coarse_map[x[:2]][..., 1] - x[3]).max() < EPS for x in tmp)
        print('Re-meshing 2D checked')

    elif test == 1.5:  # sparse_tree initiation and refining
        mesh = sparse_tree(2, 1, 1)
        assert mesh.age == 2
        assert mesh.h == 1 / 4
        assert abs(4 * mesh.dof_map - (4 * mesh.dof_map).round()).max() < EPS
        assert all(tuple(mesh.coarse_map[i]) == (0, j) for i, j in enumerate((4, 7)))
        mesh.update(2)
        assert all(tuple(mesh.coarse_map[i]) == (0, j) for i, j in enumerate((3, 4, 6, 7)))
        mesh.update(3)
        assert abs(mesh.coarse_map[..., 0]).max() == 0
        assert all(abs(mesh.coarse_map[2 * i:2 * i + 2, 1] - j).max() == 0 for i, j in enumerate((3, 4, 6, 7)))
        print('Mesh construction 1D checked')

        mesh = sparse_tree(2, 2, 1)
        assert mesh.age == 2
        assert mesh.h == 1 / 4
        tmp = ([0, 0], [0, 0], [0, 0], [1, 0], [0, 1], [1, 1], [2, 0], [2, 0], [3, 0], [2, 1], [3, 1],
               [0, 2], [0, 2], [1, 2], [0, 3], [1, 3], [2, 2], [2, 2], [3, 2], [2, 3], [3, 3])
        assert all(abs(4 * mesh.dof_map[i,:2] - np.array(j)).max() < EPS for i, j in enumerate(tmp))
        assert all(tuple(mesh.coarse_map.reshape(-1, 2)[i]) == (0, j) for i, j in enumerate((6, 16, 11, 21)))
        mesh.update(2)
        tmp = np.array((3, 5, 13, 15, 4, 6, 14, 16, 8, 10, 18, 20, 9, 11, 19, 21)).reshape(4, 4)
        assert abs(mesh.coarse_map[..., 0]).max() == 0
        assert all(mesh.coarse_map[..., 1].ravel() == tmp.ravel())
        mesh.update(3)
        assert all(abs(mesh.coarse_map[2 * i:2 * i + 2, 2 * j:2 * j + 2, 1] - tmp[i, j]).max() == 0 for i in range(4) for j in range(4))
        print('Mesh construction 2D checked')

        mesh = sparse_tree(2, 1, 2)
        mesh.refine([1, 1, 1, 0, -1, -1, 0])
        assert mesh.age == 3
        assert mesh.h == 1 / 8
        assert all(8 * mesh.dof_map[i, 0] == j for i, j in enumerate((0, 0, 0, 0, 1, 2, 4, 4, 6)))
        assert all(tuple(mesh.coarse_map[i]) == (0, j) for i, j in enumerate((5, 6, 8, 9)))
        print('Re-meshing 1D checked')

        mesh = sparse_tree(1, 2, 2)
        mesh.refine([1, 1, 0, 1, -1])
        assert mesh.age == 2
        assert mesh.h == 1 / 4
        assert abs(4 * mesh.dof_map - np.array([int(i) for i in '004002001101011111202022021121031131222']).reshape(-1, 3)).max() == 0
        assert abs(mesh.coarse_map[..., 0]).max() == 0
        assert abs(mesh.coarse_map[..., 1] - np.array([3, 5, 9, 11, 4, 6, 10, 12, 7, 7, 13, 13, 7, 7, 13, 13]).reshape(4, 4)).max() == 0
        print('Re-meshing 2D checked')

    elif test == 2:
        level = 10
        my_func = np.array([0] * (2 ** level // 3) + [1] * (2 ** level - 2 ** level // 3), dtype='float64')
        plt.subplot(121)
        plt.plot([0, 1 / 3, 1 / 3, 1], [0, 0, 1, 1], 'k', linewidth=3)
        for init in (1, 3, 5, 7, level):
            my_grid = sparse_function(0, sparse_FS(init, 1, 3))
            my_grid = my_grid.from_discrete(my_func)
            my_grid.plot(level=4, label='level = %d' % init)
            plt.legend(loc='lower right')
            plt.pause(.3)
        assert abs(my_grid.discretise(level) - my_func).max() == 0
        print('Discretising 1D checked')

        plt.subplot(122)
        x = np.linspace(0, 1, 2 ** level)
        my_func = ((x.reshape(-1, 1) - .1) ** 2 + (x.reshape(1, -1) - .7) ** 2 < .5 ** 2).astype(FTYPE)
        ax = None
        for level in range(1, 8):
            my_grid = sparse_function(0, sparse_FS(level, 2))
            my_grid = my_grid.from_discrete(my_func)

            ax = my_grid.plot(level=level, update=ax, vmin=0, vmax=1)
            plt.title('level = ' + str(level))
            plt.pause(.3)
        print('Discretising 2D checked')
        plt.show()

    elif test == 2.5:
        level = 10

        my_func = np.array([0] * (2 ** level // 3) + [1] * (2 ** level - 2 ** level // 3), dtype='float64')
        plt.subplot(121)
        plt.plot([0, 1 / 3, 1 / 3, 1], [0, 0, 1, 1], 'k', linewidth=3)
        for init in (1, 3, 5, 7, level):
            my_grid = Haar_function(0, Haar_FS(init, 1, 3))
            my_grid = my_grid.from_discrete(my_func)
            my_grid.plot(level, label='level = %d' % init)
            plt.legend(loc='lower right')
            plt.pause(.3)
        assert abs(my_grid.discretise(level) - my_func).max() < 1e-10
        print('Discretising 1D wavelet checked')

        plt.subplot(122)
        x = np.linspace(0, 1, 2 ** level)
        my_func = ((x.reshape(-1, 1) - .1) ** 2 + (x.reshape(1, -1) - .7) ** 2 < .5 ** 2).astype(FTYPE)
        ax = None
        for init in (1, 3, 5, 7, level):
            my_grid = Haar_function(0, Haar_FS(init, 2, 3))
            my_grid = my_grid.from_discrete(my_func)

            ax = my_grid.plot(init, update=ax, vmin=0, vmax=1)
            plt.title('level = ' + str(init))
            plt.pause(.3)
        assert abs(my_grid.discretise(level) - my_func).max() < 1e-10
        print('Discretising 2D wavelet checked')
        plt.show()
