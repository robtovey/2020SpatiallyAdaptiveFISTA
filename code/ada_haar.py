'''
Created on 7 Sep 2020

@author: Rob Tovey
'''

from haar_bin import (Radon_to_sparse_matrix, Radon_update_sparse_matrix, _smallgrad, 
                      _discretise1D, _from_discrete1D, _discretise2D, _from_discrete2D,
                      len_intersect, line_intersect_square, jit, prange, __pparams,
                      _FP_leaf, _FP_wave2leaf, _BP_leaf, _BP_leaf2wave)
from matplotlib import pyplot as plt
from numpy import (array, empty, sqrt, zeros, isscalar, array_equal, concatenate,
                   cos, sin, pi, log, nanmin, nanmax, log2)
from anytree import Node, RenderTree, PreOrderIter, PostOrderIter
from algorithms import shrink, stopping_criterion, faster_FISTA, \
    _makeVid, FB, Norm, FISTA
from pymorton import interleave
from lasso import op
from scipy.sparse import csr_matrix

@jit(['void(F[:],F[:],F[:],I[:,:],b1[:],F[:])'.replace('F', 'f8').replace('I', 'i4')], **__pparams)
def _unknown_grad_denoising(r, grid, L, dof_map, is_leaf, out):
    '''
    df/du_j = sum_I <w_j,k_I>r_I
    Orthogonality of k_I gives:
    |df/du_j|^2 <= sum_I |w_j|^2 |k_I 1_{supp(w_j)}|^2 |r_I|^2
                 = sum_I |k_I 1_{supp(w_j)}|^2 |r_I|^2
    '''

    for i in prange(dof_map.shape[0]):
        if is_leaf[i]:
            # only leaves have errors
            j = dof_map[i]
            supp0, supp1 = j[1] * 2 ** float(1 - j[0]), (j[1] + 1) * 2 ** float(1 - j[0])

            # P^Tr|_[g[k],g[k+1]] = r[k]/sqrt(L[k])
            total = 0
            for k in range(L.size):
                total += r[k] * r[k] * len_intersect(supp0, supp1, grid[k], grid[k + 1]) / L[k]

            out[i] = sqrt(total)
        else:
            out[i] = -1


@jit(['void(F[:,:],F[:],F[:,:],F[:],I[:,:],b1[:],F[:])'.replace('F', 'f8').replace('I', 'i4')], **__pparams)
def _unknown_grad_tomo(res, g, slope, centre, dof_map, is_leaf, out):
    '''
    df/du_j = sum_Psum_I <w_j,k_{P,I}>r_{P,I}
    Orthogonality of k_{P,I} and |w_j|=1 gives:
    |df/du_j| <= |w_j| sum_P |sum_I k_{P,I} 1_{supp(w_j)}|
    |sum_I k_{P,I} 1_{supp(w_j)}|^2 = sum_I |k_{P,I}1_{supp(w_j)}|^2
    '''
    for i in prange(dof_map.shape[0]):
        if is_leaf[i]:
            c = 2.**(1 - dof_map[i, 0])
            b0x, b0y = dof_map[i, 1] * c - centre[0], dof_map[i, 2] * c - centre[1]
            b1x, b1y = b0x + c, b0y + c
            midx, midy = b0x + .5 * c, b0y + .5 * c
            r = 2.**(-.5) * c

            big_value = 0
            for j in range(slope.shape[0]):
                t = midx * slope[j, 1] - midy * slope[j, 0]
                value = 0
                for k in range(g.size - 1):
                    if g[k] > t + r:
                        break
                    elif g[k + 1] > t - r:
                        value += res[j, k] ** 2 * abs(
                            line_intersect_square(g[k + 1] * slope[j, 1], -g[k + 1] * slope[j, 0], slope[j], b0x, b0y, b1x, b1y)
                            -line_intersect_square(g[k] * slope[j, 1], -g[k] * slope[j, 0], slope[j], b0x, b0y, b1x, b1y))

                big_value += sqrt(value)
            out[i] = big_value
        else:
            out[i] = -1


def wavelet_node(*ind, parent=None):
    ind = ind if len(ind) > 1 else ind[0]
    # TODO: this line is slow because it checks for loops in the
    return Node(str(ind), index=tuple(ind), parent=parent)


def wavelet_children(tree, *ind, parent=None):
    ind = ind if len(ind) > 1 else ind[0]
    if parent is None:
        parent = tree.get(ind, None)

    if len(ind) == 2:
        for i in range(2 * ind[1], 2 * ind[1] + 2):
            tree[(ind[0] + 1, i)] = wavelet_node(ind[0] + 1, i, parent=parent)
    elif len(ind) == 3:
        for i0 in range(2 * ind[1], 2 * ind[1] + 2):
            for i1 in range(2 * ind[2], 2 * ind[2] + 2):
                tree[(ind[0] + 1, i0, i1)] = wavelet_node(ind[0] + 1, i0, i1, parent=parent)
    else:
        # Can probably generalise this with itertools
        raise NotImplementedError


def copy_tree(tree):
    new = {i:wavelet_node(*i) for i in tree}

    for i in tree:
        root = (0,) * len(i)
        break

    for node in PreOrderIter(tree[root]):
        new_node = new[node.index]
        new_node.value = 0 + node.value
        if not node.is_root:
            new_node.parent = new[node.parent.index]
    return new


def config_tree_axes(ax, dof_map, x0=0, max_depth=5, max_ticks=100, labelsize=10):
    depth = min(dof_map[:, 0].max(), max_depth) + 1
    dim = dof_map.shape[1] - 1

#     ax.tick_params(axis='x', which='major', labelsize=labelsize)
    if dim == 1:
        label_spacing = int(dof_map.shape[0] / (3 * len(ax.get_xticklabels()))) + 1
    else:
        label_spacing = int(dof_map.shape[0] / (4 * len(ax.get_xticklabels()))) + 1
    tick_spacing = int(dof_map.shape[0] / max_ticks) + 1

    if dim == 1:
        grid = [[0, x0, ' ' * (depth - 1) + '-', '  0']]
        x = dof_map[:, 1] * 2.**(1 - dof_map[:, 0])
        for i in range(1, dof_map.shape[0]):
            if dof_map[i, 0] < depth:
                grid.append([dof_map[i, 0], x0 + i,
                             ' ' * (depth - dof_map[i, 0] - 1) + '-' * (dof_map[i, 0] + 1),
                             '  ' + str(float(x[i])) if x[i] > x[i - 1] else ''])
            elif i % tick_spacing == 0:
                grid.append([depth + 1, x0 + i, '-' * depth, ''])

    elif dim == 2:
        grid = [[0, x0, ' ' * (depth - 1) + '-', '  (0, 0)']]
        x = dof_map[:, 1:] * (2.**(1 - dof_map[:, 0])).reshape(-1, 1)
        for i in range(1, dof_map.shape[0]):
            if dof_map[i, 0] < depth:
                grid.append([dof_map[i, 0], x0 + i,
                             ' ' * (depth - dof_map[i, 0] - 1) + '-' * (dof_map[i, 0] + 1),
                             '  ' + str(tuple(x[i])) if any(x[i] > x[i - 1]) else ''])
            elif i % tick_spacing == 0:
                grid.append([depth + 1, x0 + i, '-' * depth, ''])

    grid.sort(key=lambda n: n[0])  # sort by level
    for i, g in enumerate(grid):
        if g[3] and all(abs(g[1] - y[1]) > label_spacing for y in grid[:i]):
            ax.axvline(g[1] - .5, linestyle='-', color='gray', linewidth=.5)
        else:
            g[3] = ''
    plt.xticks([g[1] for g in grid], [g[2] + g[3] for g in grid], rotation=-90)
    plt.tick_params(which='both', length=0)

class Array:

    def __init__(self, x, isFEM=None, FS=None):
        if isFEM is None:
            isFEM = not isinstance(x, ndarray)
        else:
            self.x = x
        self.FS = FS

    @property
    def size(self): return self.asarray().size

    def asarray(self):
        if self.isFEM == 1:
            return self.x.vector().get_local()
        elif self.isFEM == 2:
            raise NotImplementedError('Trying to get array from Expression')
        else:
            return self.x

    def update(self, x, FS=None):
        self.x = x
        if FS is not None:
            self.FS = FS
        elif self.isFEM == 1:
            self.FS = x.function_space()

    def __add__(self, other):
        other = other.x if isinstance(other, Array) else other
        out = None
        if self.isFEM == 1:
            if hasattr(other, 'vector'):
                tmp = self.asarray(), other.vector().get_local()
                if tmp[0].size == tmp[1].size:
                    out = func_copy(self.FS, tmp[0] + tmp[1])
            elif hasattr(other, 'shape') or isscalar(other):
                out = func_copy(self.FS, self.asarray() + other)
        if self.isFEM and out is None:
            out = df.project(self.x + other, self.FS)

        if not self.isFEM:
            out = self.x + other

        return Array(out, self.isFEM, self.FS)

    def __sub__(self, other):
        other = other.x if isinstance(other, Array) else other
        out = None
        if self.isFEM == 1:
            if hasattr(other, 'vector'):
                tmp = self.asarray(), other.vector().get_local()
                if tmp[0].size == tmp[1].size:
                    out = func_copy(self.FS, tmp[0] - tmp[1])
            elif hasattr(other, 'shape') or isscalar(other):
                out = func_copy(self.FS, self.asarray() - other)
        if self.isFEM and out is None:
            out = df.project(self.x - other, self.FS)

        if not self.isFEM:
            out = self.x - other

        return Array(out, self.isFEM, self.FS)

    def __mul__(self, other):
        other = other.x if isinstance(other, Array) else other
        out = None
        if self.isFEM == 1:
            if hasattr(other, 'vector'):
                tmp = self.asarray(), other.vector().get_local()
                if tmp[0].size == tmp[1].size:
                    out = func_copy(self.FS, tmp[0] * tmp[1])
            elif hasattr(other, 'shape') or isscalar(other):
                out = func_copy(self.FS, self.asarray() * other)
        if self.isFEM and out is None:
            out = df.project(self.x * other, self.FS)

        if not self.isFEM:
            out = self.x * other

        return Array(out, self.isFEM, self.FS)

    def __truediv__(self, other):
        other = other.x if isinstance(other, Array) else other
        out = None
        if self.isFEM == 1:
            if hasattr(other, 'vector'):
                tmp = self.asarray(), other.vector().get_local()
                if tmp[0].size == tmp[1].size:
                    out = func_copy(self.FS, tmp[0] / tmp[1])
            elif hasattr(other, 'shape') or isscalar(other):
                out = func_copy(self.FS, self.asarray() / other)
        if self.isFEM and out is None:
            out = df.project(self.x / other, self.FS)

        if not self.isFEM:
            out = self.x / other

        return Array(out, self.isFEM, self.FS)

    def __rtruediv__(self, other):
        out = None
        if self.isFEM == 1:
            if hasattr(other, 'shape') or isscalar(other):
                out = func_copy(self.FS, other / self.asarray())
        if self.isFEM and out is None:
            out = df.project(other / self.x, self.FS)

        if not self.isFEM:
            out = other / self.x

        return Array(out, self.isFEM, self.FS)

    def __iadd__(self, other):
        new = self + other
        self.x = new.x
        self.isFEM = new.isFEM
        self.FS = new.FS

    def __isub__(self, other):
        new = self - other
        self.x = new.x
        self.isFEM = new.isFEM
        self.FS = new.FS

    def __imul__(self, other):
        new = self * other
        self.x = new.x
        self.isFEM = new.isFEM
        self.FS = new.FS

    def __itruediv__(self, other):
        new = self / other
        self.x = new.x
        self.isFEM = new.isFEM
        self.FS = new.FS

    def __neg__(self):
        if self.isFEM == 1:
            out = func_copy(self.FS, -self.asarray())
        else:
            out = -self.x
        return Array(out, self.isFEM, self.FS)

    __radd__ = __add__
#     __rsub__ = __sub__
    __rmul__ = __mul__

class Haar_space:

    def __init__(self, init=1, dim=1, _safe=False):
        if type(init) is dict:
            for k in init:
                dim = len(k) - 1
                break
            tree = init if _safe else copy_tree(init)
        else:
            assert type(init) is int
            assert init >= 1  # Must always have root and first wavelet
            tree = {(0,) * (1 + dim): wavelet_node((0,) * (1 + dim))}
            tree[(1,) + (0,) * dim] = wavelet_node((1,) + (0,) * dim, parent=tree[(0,) * (1 + dim)])

            if dim == 1:
                for k in range(1, init):
                    for j in range(2 ** (k - 1)):
                        wavelet_children(tree, k, j)
            elif dim == 2:
                for k in range(1, init):
                    for j0 in range(2 ** (k - 1)):
                        for j1 in range(2 ** (k - 1)):
                            wavelet_children(tree, k, j0, j1)
            else:
                # Can probably generalise this with itertools
                raise NotImplementedError

        self.tree, self.dim = tree, dim
        self.update()
        # data[i] = (level, index0, ...)
        # level=0 is constant
        # level=k is Haar wavelet support size 2^{k-1}
        # index0=0,1,...,2^k-1 index of wavelet

        # In 1D
        #            -1/sqrt(2)     x \in [0,1]
        # Let W(x) = +1/sqrt(2)     x \in [1,2]
        #            0              else
        # f(x) = arr[0] + sum_d arr[d] * sqrt(2)^d.level * W(2^d.level * x-d.index)

    def _dof_map(self, level=None):
        dof_map = empty((len(self.tree), self.dim + 1), dtype='int32')
        is_leaf = empty(len(self.tree), dtype=bool)

        i = 0
        for node in PreOrderIter(self.tree[(0,) * (self.dim + 1)],
                                 maxlevel=None if level is None else level + 1):
            dof_map[i] = node.index
            is_leaf[i] = len(node.children) == 0
#             node.pre_index = i
            i += 1
        dof_map = dof_map[:i]

#         n = i - 1
#         post2pre = empty(dof_map.shape[0], dtype='int32')
#         leaf_map = empty(dof_map.shape, dtype='int32')
#         i = 0
#         for node in PostOrderIter(self.tree[(0,) * (self.dim + 1)],
#                                  maxlevel=None if level is None else level + 1):
#             leaf_map[-i] = node.index
#             node.post_index = n - i
#             post2pre[node.pre_index] = n - i
#             i += 1

        return dof_map, is_leaf  # leaf_map, post2pre

    def update(self):
        # update dof_map
        self.dof_map, self.is_leaf = self._dof_map()
        self.depth = self.dof_map[:, 0].max()

    def split(self, indcs):
        indcs = array(indcs, dtype='int32')
        if indcs.ndim == 1:
            # assume raw indices
            indcs = [self.dof_map[i] for i in indcs]
        else:
            indcs = list(indcs)

        if self.dim == 1:
            indcs = [(tuple(i), i[1]) for i in indcs]
        elif self.dim > 1 and self.dim < 4:
            indcs = [(tuple(i), interleave(*(int(j) for j in i[1:]))) for i in indcs]
        else:
            raise NotImplementedError
        indcs.sort(key=lambda n: n[1])  # sort by interleaved index
        indcs.sort(key=lambda n: n[0][0], reverse=True)  # inverse sort by level, no double-insertions

        for i in indcs:
            N = self.tree.get(i[0], None)
            if N is not None and len(N.children) == 0:
                wavelet_children(self.tree, i[0], parent=N)

        self.update()

    def trim(self, indcs):
        indcs = array(indcs, dtype='int32')
        if indcs.ndim == 1:
            # assume raw indices
            indcs = [self.dof_map[i] for i in indcs]
        else:
            indcs = list(indcs)

        if self.dim == 1:
            indcs = [(tuple(i), i[1]) for i in indcs]
        elif self.dim > 1 and self.dim < 4:
            indcs = [(tuple(i), interleave(*(int(j) for j in i[1:]))) for i in indcs]
        else:
            raise NotImplementedError
        indcs.sort(key=lambda n: n[1])  # sort by interleaved index
        indcs.sort(key=lambda n: n[0][0])  # sort by level, no double-deletes
        while indcs and indcs[0][0][0] <= 1:
            indcs = indcs[1:]

        # Filter into clusters of children
        if self.dim == 1:

            def fltr(ii):
                if ii[0][0][1] % 2 != 0:
                    return False
                return (ii[0][0][0] == ii[1][0][0] and ii[0][1] == ii[1][1] - 1)

        else:

            def fltr(ii):
                if any(i % 2 != 0 for i in ii[0][0][1:]):
#                     print('Not start of chain', ii[0])
                    return False
                elif any(i[0][0] != ii[0][0][0] for i in ii[1:]):
                    return False
                else:
                    return all(j[1] == ii[0][1] + i for i, j in enumerate(ii))

        indcs = [indcs[i] for i in range(len(indcs) - 2 ** self.dim + 1)
                    if fltr(indcs[i:i + 2 ** self.dim])]

        # Only remove leaf pairs
        for i in indcs:
            N = self.tree.get(i[0], None)
            if N is not None:
                N = N.parent
                if all(len(c.children) == 0 for c in N.children):
                    for c in N.children:
                        del self.tree[c.index]
                    N.children = []

        self.update()

    def __str__(self): return RenderTree(self.tree[(0,) * (self.dim + 1)], maxlevel=10).by_attr() + '\n'

    def old2new(self, arr, dof_map, copy=True):
        if array_equal(dof_map, self.dof_map):
            return (arr.copy() if copy else arr)

        old = {tuple(dof_map[i]):arr[i] for i in range(arr.shape[0])}
        new = empty(self.shape, dtype='float64')

        for i in range(new.shape[0]):
            new[i] = old.get(tuple(self.dof_map[i]), 0)

        return new

    def norm(self, x): return (x.asarray() ** 2).sum() ** .5

    @property
    def size(self): return len(self.tree)

    @property
    def shape(self): return (self.size,) + (() if self.dim == 1 else (2 ** self.dim - 1,))


class Haar(Array):

    def __init__(self, arr, FS, maps=None):
        if isscalar(arr):
            tmp = zeros(FS.shape, dtype='float64')
            if tmp.ndim == 1:
                tmp[0] = arr
            else:
                tmp[0, 0] = arr
            arr = tmp
        Array.__init__(self, arr, isFEM=False, FS=FS)
        if maps is None:
            self.dof_map, self.is_leaf = self.FS.dof_map, self.FS.is_leaf
        else:
            self.dof_map, self.is_leaf = maps
        assert self.x.shape[0] == self.dof_map.shape[0]
        # data[i] = (level, index, type, coeff)
        # level=0 is constant
        # level=k is Haar wavelet support size 2^{k-1}
        # index=0,1,...,2^k-1 index of wavelet
        # type=0 has no use in 1D wavelets
        # coeff \in R

        #            -1/sqrt(2)     x \in [0,1]
        # Let W(x) = +1/sqrt(2)     x \in [1,2]
        #            0              else
        # f(x) = data[0].coeff + sum_{d} d.coeff * sqrt(2)^d.level * W(2^d.level * x-d.index))

    def asarray(self): return self.x

    @property
    def size(self): return self.x.size if self.FS.dim == 1 else (self.x.size + 1 - self.x.shape[1])

    @property
    def shape(self): return (self.dof_map.shape[0],) + (() if self.FS.dim == 1 else (2 ** self.FS.dim - 1,))

    def loc(self, centre=False):
        if centre:
            X = [(self.dof_map[:, i] + .5) * 2.**(1 - self.dof_map[:, 0])
                    for i in range(1, self.dof_map.shape[1])]
        else:
            X = [self.dof_map[:, i] * 2.**(1 - self.dof_map[:, 0])
                    for i in range(1, self.dof_map.shape[1])]
        if len(X) == 1:
            return X[0]
        else:
            return concatenate([xx.reshape(-1, 1) for xx in X], axis=1)

    def update(self, x, FS=None):
        if FS is None:
            assert self.x.shape == x.shape
            self.x = x
        else:
            self.x = x
            self.FS = FS
            self.dof_map = self.FS.dof_map
            assert self.x.shape[0] == self.dof_map.shape[0]

    def __add__(self, other):
        if array_equal(self.dof_map, other.dof_map):
            return self.copy(self.x + other.x)
        elif not (self.FS is other.FS):
            raise
        else:
            FS = self.FS
            return self._copy(FS.old2new(self.x, self.dof_map, copy=False) +
                              FS.old2new(other.x, other.dof_map, copy=False),
                              FS, None)

    def __sub__(self, other): return self + (-other)

    def __mul__(self, other):
        assert isscalar(other)
        return self.copy(self.x * other)

    def __truediv__(self, other): return self * (1 / other)

    def __rtruediv__(self, other):
        raise NotImplementedError
        assert isscalar(other)
        return self.copy(other / self.x)

    def __iadd__(self, other):
        if array_equal(self.dof_map, other.dof_map):
            self.x += other.x
        elif not (self.FS is other.FS):
            raise
        else:
            FS = self.FS
            self.x = (FS.old2new(self.x, self.dof_map, copy=False)
                      +FS.old2new(other.x, other.dof_map, copy=False))
            self.dof_map = FS.dof_map

    def __isub__(self, other): self += (-other)

    def __imul__(self, other):
        assert isscalar(other)
        self.x *= other

    def __itruediv__(self, other): self *= (1 / other)

    def __neg__(self): return self.copy(-self.x)

    __radd__ = __add__
#     __rsub__ = __sub__
    __rmul__ = __mul__

    def discretise(self, level): raise NotImplementedError

    def from_discrete(self, arr): raise NotImplementedError

    def plot(self, level, ax=None, **kwargs): raise NotImplementedError

    def _copy(self, arr, FS, maps): raise NotImplementedError

    def copy(self, like=None):
        if like is None:
            arr = self.x.copy()
        elif isscalar(like):
            arr = zeros(self.shape, dtype=self.x.dtype)
            arr[0] = like
        else:
            assert like.size == self.x.size
            arr = like.reshape(self.x.shape)
        return self._copy(arr, self.FS, (self.dof_map, self.is_leaf))


class Haar_1D(Haar):

    def __init__(self, arr, FS, maps=None):
        assert FS.dim == 1
        Haar.__init__(self, arr, FS, maps)

    def discretise(self, level):
        out = empty(2 ** level, dtype='float64')

        _discretise1D(self.x[0], self.x[1:], self.dof_map[1:], 1, out)
        return out

    def from_discrete(self, arr):
        level = round(log2(arr.size))
        assert arr.size == 2 ** level

        x = empty(self.x.shape, dtype='float64')
        x[0] = _from_discrete1D(arr, self.dof_map[1:], 1, x[1:])

        return self.copy(x)

    def plot(self, level, ax=None, update=False, **kwargs):
        ax = ax if ax is not None else plt
        y = self.discretise(level)
        x = linspace(0, 1, 2 ** level)
        if update:
            ax.set_data(x, y)
            return ax
        return ax.step(x, y, where='mid', **kwargs)[0]

    def plot_tree(self, level=5, max_ticks=200, ax=None, spatial=True, update=False, **kwargs):
        ax = ax if ax is not None else plt.gca()
        eps = abs(self.x).max() * 1e-8

        if update:
            ax.clear()

        if spatial:
            # place in middle of support (at jump)
            x = (self.dof_map[:, 1] + .5) * 2.**(1 - self.dof_map[:, 0])
            x[0] = 0

            # do plotting
            I = [i for i in range(self.size) if abs(self.x[i]) > eps]
            lines = ax.plot(x[I], self.x[I], '*', **kwargs)
            I = [i for i in range(self.size) if abs(self.x[i]) <= eps]
            lines += ax.plot(x[I], self.x[I], '.', **kwargs)

            ax.set_xlim(-0.01, 1)
            ax.set_xlabel('Jump location')

        else:
            # do plotting
            I = [i for i in range(self.size) if abs(self.x[i]) > eps]
            lines = ax.plot(I, self.x[I], '*', **kwargs)
            I = [i for i in range(self.size) if abs(self.x[i]) <= eps]
            lines += ax.plot(I, self.x[I], '.', **kwargs)

            # prepare x ticks
            label_spacing = int(self.dof_map.shape[0] / (3 * len(ax.get_xticklabels()))) + 1
            tick_spacing = int(self.dof_map.shape[0] / max_ticks) + 1
            x = self.dof_map[:, 1] * 2.**(1 - self.dof_map[:, 0])

            grid = [[0, 0, ' ' * (level - 1) + '-', '  0']]
            for i in range(1, self.dof_map.shape[0]):
                L = self.dof_map[i, 0]
                if L < level:
                    grid.append([L, i, ' ' * (level - L - 1) + '-' * (L + 1),
                                 '  ' + str(float(x[i])) if x[i] > x[i - 1] else ''])
                elif i % tick_spacing == 0:
                    grid.append([level + 1, i, '-' * level, ''])

            # refine x ticks
            grid.sort(key=lambda n: n[0])  # sort by level
            for i, g in enumerate(grid):
                if g[3] and all(abs(g[1] - y[1]) > label_spacing for y in grid[:i]):
                    ax.axvline(g[1] - .5, linestyle='-', color='gray', linewidth=.5)
                else:
                    g[3] = ''

            # update x ticks
            ax.set_xticks([g[1] for g in grid], [g[2] + g[3] for g in grid], rotation=-90)
            ax.tick_params(which='both', length=0)
            ax.set_xlabel('Wavelet index')

        ax.set_ylabel('Coefficient value')
        return ax

    def _copy(self, arr, FS, maps): return Haar_1D(arr, FS, maps)


def Haar_denoising(n_points, alpha, iters=None, prnt=True, plot=True, vid=None, algorithm='Greedy'):
    '''
    min 1/2|Pu-d|_2^2 + alpha*|Wu|_1

    Define I_j = [p_j,p_{j+1}]
    Define (Pu)_j = \int_{I_j} u(x) dx/ sqrt(|I_j|)

    P^T d|_{I_j} = d_j / sqrt(|I_j|)
    PP^T d = d_j => |P| = 1

    f(u) = 1/2|Pu-d|_2^2
    \nabla f(u)|_{I_j} = \int_{I_j} u/sqrt(|I_j|) - d_j

    g(u) = k*|Wu|_1
    prox_(tg)(U) = \argmin 1/2|u-U|_2^2 + t*k|Wu|_1
        -> <u,w_i> = <U,w_i> or 0
    '''
    from numpy import random
    random.seed(101)

    grid = list(random.rand(n_points - 1))
    grid.sort()
    grid = array([0] + list(grid) + [1], dtype='float64')
    L = array([grid[i + 1] - grid[i] for i in range(n_points)])

    data = random.rand(n_points)

    pms = {'i':0, 'eps':1e-18}
    recon = Haar_1D(0, Haar_space(1, dim=1))

    def residual(u):
        return array([u.x[0] * sqrt(L[j])
                      +interval_integral(grid[j:j + 2], u.x[1:], u.dof_map[1:], 1) / sqrt(L[j]) - data[j]
             for j in range(n_points)], dtype='float64')

    def energy(u):
        refine(u, True)
        return [pms[t] for t in ('F', 'discrete_err', 'cont_err')]

    @jit(nopython=True, parallel=False, fastmath=True)
    def interval_integral(x, arr, dof_map, level):
        if arr.shape[0] == 0:
            return 0
        n = dof_map[0, 1]
        supp0, supp1 = n * 2 ** float(1 - level), (n + 1) * 2 ** float(1 - level)
        # non-intersecting support:
        if x[0] >= supp1 or x[1] <= supp0:
            return 0
        # totally overlapping support:
        if x[0] <= supp0 and x[1] >= supp1:
            return 0

        midpoint, scale = .5 * (supp0 + supp1), 2 ** ((level - 1) / 2)
        value = scale * arr[0] * (len_intersect(midpoint, supp1, x[0], x[1])
                                  -len_intersect(supp0, midpoint, x[0], x[1]))

        i = 2  # skip root and first child
        while i < dof_map.shape[0]:
            if dof_map[i, 0] == level + 1:
                break
            i += 1

        return (value
                +interval_integral(x, arr[1:i], dof_map[1:i], level + 1)
                +interval_integral(x, arr[i:], dof_map[i:], level + 1))

    def gradF(u, refine_flag=True):
        if refine_flag:
            pms['i'] += 1
            if pms['i'] % 10 == 0:
                u = refine(u)

        R = residual(u)

        DF = u.copy(0)
        DF.x[0] = sum(R[k] * L[k] / sqrt(L[k]) for k in range(n_points))
        _gradF(u.dof_map, grid, L, R, DF.x)
        return DF

    def proxG(u, t):
        U = u.copy(0)
        shrink(u.x.reshape(-1, 1), t * alpha, U.x.reshape(-1, 1))
        return U

    def doPlot(i, u, fig, plt):
        if pms.get('axes', None) is None:
            pms['axes'] = fig.subplots(ncols=3)

        data2func = lambda d: [d[0] / sqrt(L[0])] + [d[i] / sqrt(L[i]) for i in range(n_points)]

        ax = pms['axes'][0]
        if pms.get('recon plot', None) is None:
            pms['recon plot'] = (u.plot(ax=ax, level=10, linewidth=2, color='red', linestyle=':', label='Reconstruction'),
                                 ax.step(grid, data2func(residual(u) + data), linewidth=3, color='red', linestyle='-', where='pre', label='Projection of reconstruction')[0],
                                 ax.step(grid, data2func(data), linewidth=3, color='blue', where='pre', label='Data')[0])
            ax.set_xlim(0, 1)
            ax.legend(loc='upper right')
        else:
            u.plot(level=10, max_ticks=500, mass=True, ax=pms['recon plot'][0], update=True)
            pms['recon plot'][1].set_ydata(data2func(residual(u) + data))
            pms['recon plot'][2].set_ydata(data2func(data))
        ax.set_ylim(-.1, min(.8 * n_points, 1.1 * max(data2func(data))))

        ax = pms['axes'][1]
        if pms.get('recon stars', None) is None:
            pms['recon stars'] = u.plot_tree(ax=ax, max_ticks=100)
            ax.set_title('Coefficients')
        else:
            u.plot_tree(ax=pms['recon stars'], max_ticks=100, update=True)

        ax = pms['axes'][2]
        if pms.get('error plots', None) is None:
            pms['error plots'] = (ax.plot(1 + stop_crit.I, stop_crit.extras[:, 1], label='Discrete error')[0],
                                  ax.plot(1 + stop_crit.I, stop_crit.extras[:, 2], label='Continuous error')[0])

            ax.set_yscale('log')
            ax.legend(loc='upper right')
            ax.set_xlabel('Iteration number')
            ax.set_ylabel('L2 norm of gradient')
        else:
            pms['error plots'][0].set_data(1 + stop_crit.I, stop_crit.extras[:, 1])
            pms['error plots'][1].set_data(1 + stop_crit.I, stop_crit.extras[:, 2])
            ax.set_xlim(1, max(i, 2))
            ax.set_ylim(.9 * nanmin(stop_crit.extras[:, 1]), 1.1 * nanmax(stop_crit.extras[:, 2]))

        if i < 2:
            fig.tight_layout()

    # Gradient check:
#     recon, dx = random.randn(recon.size), random.randn(recon.size)
#     f = lambda u: .5 * (residual(u) ** 2).sum()
#     f0, df = f(recon), (gradF(recon) * dx).sum()
#     for e in [10 ** (-i) for i in range(7)]:
#         f1 = f(recon + e * dx)
#         print('e = %.1e, f-Pf = %+.1e, f-P\'f = %+.1e, Df = %+.1e ~ %+.1e' % (e, f1 - f0, f1 - f0 - e * df, df, (f1 - f0) / e))
#
#     exit()

    # normalise data such that alpha=0 => recon=0

    def refine(u, dummy=False):
        '''
        Compute primal-dual gap:

        F(u) = 1/2|Au-d|_2^2 + alpha*|u|_1

        \nabla F(u) = A^*(Au-d) + alpha*sign(u)
        |\nabla F(u)|^2 = |known|^2 + |unknown|^2

        A = PW^T
        '''

        r = residual(u)  # data vector
        df = gradF(u, refine_flag=False).asarray()  # known gradient of f

        dF = df.copy()
        _smallgrad(dF, u.asarray(), alpha, pms['eps'])  # known gradient

        # unknown gradient of f = P^Tr on unknown indices
        # |unknown gradient|^2 = |P^Tr|^2-|A^*r|^2 = |r|^2 - |df|^2
        leaf_error = empty(df.shape, dtype=df.dtype)
        _unknown_grad_denoising(r, grid, L, u.dof_map, u.is_leaf, leaf_error)

        eF = Norm(leaf_error[leaf_error > alpha] - alpha)
#         print(eF, leaf_error.round(1))

        pms['F'] = (.5 * (r ** 2).sum() + alpha * abs(u.asarray()).sum())
        pms['discrete_err'] = Norm(dF)
        pms['cont_err'] = (Norm(dF) ** 2 + eF ** 2) ** .5

        if dummy:
            return u

        FS = u.FS
        # Trim leaves with 0 intensity with 0 error even after coarsening
        FS.trim(u.dof_map[(leaf_error >= -.5) * (leaf_error < alpha / 2) * (u.asarray() == 0)])
        # Split leaves with larger continuous than discrete gradient
        FS.split(u.dof_map[(leaf_error - alpha > abs(dF))])

        return Haar_1D(FS.old2new(u.x, u.dof_map), FS)

#     def custom_stop(i, *_, d=None, extras=None, **__): return ((extras[0 if adaptive[1] else 1] < stop) and i > 2)
    def custom_stop(*_, d=None, extras=None, **__): return (d + extras[2] < 1e-12)

    stop_crit = stopping_criterion(iters[0], custom_stop, frequency=iters[1], prnt=prnt, record=True,
                              energy=energy, vid=vid, fig=None if plot is False else _makeVid(stage=0, record=vid),
                              callback=doPlot if plot else lambda *args:None)

    if type(algorithm) is str:
        algorithm = algorithm, {}
    if algorithm[0] is 'Greedy':
        default = {'xi':0.95, 'S':1}
        default.update(algorithm[1])
        recon = faster_FISTA(recon, 1, gradF, proxG, stop_crit, **default)
    elif algorithm[0] is 'FISTA':
        default = {'a':10, 'restarting':False}
        default.update(algorithm[1])
        recon = FISTA(recon, 1, gradF, proxG, stop_crit, **default)
    elif algorithm[0] is 'FB':
        default = {'scale':2}
        default.update(algorithm[1])
        recon = FB(recon, default['scale'], gradF, proxG, stop_crit)

    return recon, stop_crit


class Haar_2D(Haar):

    def __init__(self, arr, FS, maps=None):
        assert FS.dim == 2
        Haar.__init__(self, arr, FS, maps)

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
        out = empty((2 ** level,) * 2, dtype='float64')
        _discretise2D(self.x[0, 0], self.x[1:], self.dof_map[1:], 1, out)
        return out

    def from_discrete(self, arr):
        level = log2(arr.shape).round().astype(int)
        assert level[0] == level[1]
        level = level[0]
        assert arr.shape[0] == 2 ** level and arr.shape[1] == 2 ** level

        x = empty(self.x.shape, dtype='float64')
        x[0, 0] = _from_discrete2D(arr, self.dof_map[1:], 1, x[1:])
        x[0, 1:] = 0

        return self.copy(x)

    def plot(self, level, ax=None, update=False, **kwargs):
        y = self.discretise(level).T
        if update:
            assert ax is not None
            return ax.set_data(y)
        ax = ax if ax is not None else plt
        return ax.imshow(y, origin='lower', extent=[0, 1, 0, 1], **kwargs)

    def plot_tree(self, level=5, max_ticks=200, ax=None, spatial=True, update=False, **kwargs):
        ax = ax if ax is not None else plt.gca()
        eps = abs(self.x).max() * 1e-10

        if spatial:
            # place in middle of support (at jump)
            x = (self.dof_map[:, 1:] + .5) * 2.**(1 - self.dof_map[:, :1])
            x[0, :] = 0.01

            # do plotting
            lines = []
            tmp, eps = log(abs(self.x).max(1) + eps), log(eps)
            tmp, eps = (tmp - tmp.min()) / tmp.ptp(), (eps - tmp.min()) / tmp.ptp()
            I = [i for i in range(tmp.size) if tmp[i] <= eps]
            L = ax.hist2d(x[:, 0], x[:, 1], range=[[0, 1], [0, 1]],
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
            if self.x.shape[0] < 50 ** 2:
                lines.append(ax.scatter(x[I, 1], x[I, 0], 2, tmp[I], '.', cmap=cmap, **kwargs))
            I = [i for i in range(tmp.size) if tmp[i] > eps]
            lines.append(ax.scatter(x[I, 1], x[I, 0], 40, tmp[I], '*', cmap=cmap, **kwargs))

            ax.set_aspect('equal', 'box')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Jump location x')
            ax.set_ylabel('Jump location y')
            ax.set_title('Logarithmic intensity in red, density in blue')

        else:

            # place at start of support
            x = self.dof_map[:, 1:] * 2.**(1 - self.dof_map[:, :1]).reshape(-1, 1)

            # do plotting
            I = [i for i in range(self.x.shape[0]) if abs(self.x[i]).max() <= eps]
            lines = ax.plot(I, 0 * self.x[I, 0], '.', color='tab:orange', **kwargs)
            for j in range(3):
                I = [i for i in range(self.x.shape[0]) if abs(self.x[i, j]) > eps]
                lines += ax.plot(I, self.x[I, j], '*', color='tab:blue', **kwargs)

            # prepare x ticks
            level = min(level, 5)  # it's just too expensive otherwise
            label_spacing = int(self.dof_map.shape[0] / (4 * len(ax.get_xticklabels()))) + 1
            tick_spacing = int(self.dof_map.shape[0] / max_ticks) + 1

            grid = [[0, 0, ' ' * (level - 1) + '-', '  (0, 0)']]
            for i in range(1, self.dof_map.shape[0]):
                L = self.dof_map[i, 0]
                if L < level:
                    grid.append([L, i, ' ' * (level - L - 1) + '-' * (L + 1),
                                 '  ' + str(tuple(x[i])) if any(x[i] > x[i - 1]) else ''])
                elif i % tick_spacing == 0:
                    grid.append([level + 1, i, '-' * level, ''])

            # refine x ticks
            grid.sort(key=lambda n: n[0])  # sort by level
            for i, g in enumerate(grid):
                if g[3] and all(abs(g[1] - y[1]) > label_spacing for y in grid[:i]):
                    ax.axvline(g[1] - .5, linestyle='-', color='gray', linewidth=.5)
                elif any(abs(g[1] - y[1]) < tick_spacing for y in grid[:i]):
                    g[3] = None
                else:
                    g[3] = ''
            grid = [g for g in grid if g[3] is not None]

            # update x ticks
            ax.set_xticks([g[1] for g in grid], [g[2] + g[3] for g in grid], rotation=-90)
            ax.tick_params(which='both', length=0)
            ax.set_xlabel('Wavelet index')
            ax.set_ylabel('Coefficient value')

        return lines

    def _copy(self, arr, FS, maps): return Haar_2D(arr, FS, maps)


class Haar2Sino(op):

    def __init__(self, angles, grid, FS, centre=[0, 0]):
        A = array(angles) * pi / 180
        directions = array([(cos(a), sin(a)) for a in A], dtype='float64')
        centre = array(centre, dtype='float64').reshape(2)

        op.__init__(self, self.fwrd, self.bwrd, out_sz=(len(angles), len(grid) - 1), norm=1)

        self.angles, self.d = angles, directions
        self.grid, self.centre = grid, centre
        self.FS, self._buf = FS, {'DM':None}

        self.scale = self.fwrd(Haar_2D(1, Haar_space(dim=2)))  # A good preconditioner to make each angle have norm 1
        self._norm = Norm(self.scale.max(1) ** .5)  # max/norm over orthogonal/non-orthogonal kernels

    def update(self, FS, refine=None):
        DM = FS.dof_map
        FS = FS.FS if hasattr(FS, 'FS') else FS
        self.FS = FS
        if DM is self._buf['DM']:
            return
        return
        self._buf['DM'] = DM
        total_sz = int(2 * self.d.size * self.grid.size * (2.** (-.5 * DM[:, 0])).sum())

        while True:
            ptr, indcs = empty(DM.shape[0] * 3 + 1, dtype='int32'), empty(total_sz, dtype='int32')
            data = empty(total_sz, dtype='float64')
            try:
                if self._buf.get('data', None) is None or refine is None:
                    Radon_to_sparse_matrix(data, indcs, ptr, DM, self.grid, self.d, self.centre)
                else:
                    Radon_update_sparse_matrix(self._buf['data'], self._buf['indcs'], self._buf['ptr'], refine,
                                               data, indcs, ptr, DM, self.grid, self.d, self.centre)
                break
            except MemoryError:
                total_sz *= 2

        tot = ptr.max()
        self._buf['bwrd_mat'] = csr_matrix((data[:tot], indcs[:tot], ptr), shape=(DM.shape[0] * 3, self.d.shape[0] * (self.grid.size - 1)))
        self._buf['fwrd_mat'] = self._buf['bwrd_mat'].T.tocsr()
        self._buf['fwrd_mat'].sort_indices()
        self._buf['data'], self._buf['indcs'], self._buf['ptr'] = data, indcs, ptr

    def fwrd(self, w):

        if w.dof_map is self._buf['DM']:
            return self._buf['fwrd_mat'].dot(w.x.reshape(-1))
        else:
            s = zeros(self._shape[1], dtype='float64')
            buf = zeros((w.dof_map.shape[0], 4), dtype='float64')
            _FP_wave2leaf(w.x[0, 0], w.x[1:], w.dof_map[1:], 1, buf[1:])
            _FP_leaf(buf, w.dof_map, w.is_leaf, self.grid, self.d, self.centre, s)
#             if abs(s.mean() - 0.00473710197206454) < 1e-5:
#                 print(s.round(4))
#                 exit()
            return s

    def bwrd(self, s, like=None):
        dof_map = self.FS.dof_map if like is None else like.dof_map
        is_leaf = self.FS.is_leaf if like is None else like.is_leaf

        if dof_map is self._buf['DM']:
            out = self._buf['bwrd_mat'].dot(s.reshape(-1)).reshape(dof_map.shape[0], 3)
        else:
            s = s.reshape(self._shape[1])
            out = zeros((dof_map.shape[0], 3), dtype='float64')
            buf = empty(dof_map.shape[0], dtype='float64')
            _BP_leaf(s, dof_map, is_leaf, self.grid, self.d, self.centre, out, buf)
            out[0, 0] = _BP_leaf2wave(buf[1:], dof_map[1:], 1, out[1:])

        if like is None:
            return Haar_2D(out, self.FS)
        else:
            return like.copy(out)

    def plot(self, s, ax=None, update=False, **kwargs):
        if update:
            assert ax is not None
            return ax.set_data(s.T)
        ax = ax if ax is not None else plt
        return ax.imshow(s.T, origin='lower', aspect='auto',
                         extent=[self.angles.min(), self.angles.max(), self.grid.min(), self.grid.max()], **kwargs)


def Haar_tomo(A, data, alpha, iters=None, prnt=True, plot=True, vid=None, algorithm='Greedy'):
    '''
    min 1/2|Pu-d|_2^2 + alpha*|Wu|_1

    Define I_j = [p_j,p_{j+1}]
    Define (Pu)_j = \int_{I_j} u(x) dx/ sqrt(|I_j|)

    P^T d|_{I_j} = d_j / sqrt(|I_j|)
    PP^T d = d_j => |P| = 1

    f(u) = 1/2|Pu-d|_2^2
    \nabla f(u)|_{I_j} = \int_{I_j} u/sqrt(|I_j|) - d_j

    g(u) = k*|Wu|_1
    prox_(tg)(U) = \argmin 1/2|u-U|_2^2 + t*k|Wu|_1
        -> <u,w_i> = <U,w_i> or 0
    '''
    from numpy import random
    random.seed(101)

    pms = {'i':0, 'eps':1e-18, 'refineI':2}
    recon = Haar_2D(0, Haar_space(1, dim=2))
    A.update(recon.FS)

    def residual(u): return A(u) - data

    def energy(u):
        refine(u, True)
        return [pms[t] for t in ('F', 'discrete_err', 'cont_err', 'dof', 'depth', 'efficiency')]

    def gradF(u):
        pms['i'] += 1
        if pms['i'] >= pms['refineI']:
            u = refine(u)
            pms['refineI'] = max(pms['refineI'] + 1, pms['refineI'] * 1.15)  # 20 refinements for each factor of 10 iterations
#             pms['refineI'] = max(pms['refineI'] + 1, pms['refineI'] * 1.30)  #  9 refinements for each factor of 10 iterations
        return A.bwrd(A(u) - data, like=u)

    def proxG(u, t):
        U = u.copy(0)
        shrink(u.x.reshape(-1, 1), t * alpha, U.x.reshape(-1, 1))
        return U

    def doPlot(i, u, fig, plt):
        s = data.max()

        if pms.get('axes', None) is None:
            pms['axes'] = [fig.add_subplot(*x) for x in
                           ((2, 3, 1), (2, 3, 2), (2, 3, 5), (2, 3, 4), (1, 3, 3))]

        ax = pms['axes'][0]
        if pms.get('sino plot', None) is None:
            pms['sino plot'] = A.plot(A(u), ax=ax, vmin=-.1 * s, vmax=1.1 * s)
            ax.set_title('Reconstructed sinogram')
        else:
            A.plot(A(u), ax=pms['sino plot'], update=True)

        ax = pms['axes'][1]
        if pms.get('recon plot', None) is None:
            pms['recon plot'] = u.plot(ax=ax, level=8, vmin=-.1, vmax=1.1)
            ax.set_title('Reconstruction')
        else:
            u.plot(level=8, ax=pms['recon plot'], update=True)

        u.plot_tree(ax=pms['axes'][2], level=10, update=True)
        ax.set_title('Coefficients')

        ax = pms['axes'][3]
        if pms.get('data plot', None) is None:
            pms['data plot'] = A.plot(data, ax=ax, vmin=-.1 * s, vmax=1.1 * s)
            ax.set_title('Data sinogram')

        ax = pms['axes'][4]
        if pms.get('error plots', None) is None:
            pms['error plots'] = (ax.plot(1 + stop_crit.I, stop_crit.extras[:, 1], label='Discrete error')[0],
                                  ax.plot(1 + stop_crit.I, stop_crit.extras[:, 2], label='Continuous error')[0])
            ax.set_yscale('log')
            ax.legend(loc='upper right')
            ax.set_xlabel('Iteration number')
            ax.set_ylabel('L2 norm of gradient')
        else:
            pms['error plots'][0].set_data(1 + stop_crit.I, stop_crit.extras[:, 1])
            pms['error plots'][1].set_data(1 + stop_crit.I, stop_crit.extras[:, 2])
            ax.set_xlim(1, max(i, 2))
            ax.set_ylim(.9 * nanmin(stop_crit.extras[:, 1]), 1.1 * nanmax(stop_crit.extras[:, 2]))

        if i < 2:
            fig.tight_layout()

    # Gradient check:
#     recon, dx = random.randn(recon.size), random.randn(recon.size)
#     f = lambda u: .5 * (residual(u) ** 2).sum()
#     f0, df = f(recon), (gradF(recon) * dx).sum()
#     for e in [10 ** (-i) for i in range(7)]:
#         f1 = f(recon + e * dx)
#         print('e = %.1e, f-Pf = %+.1e, f-P\'f = %+.1e, Df = %+.1e ~ %+.1e' % (e, f1 - f0, f1 - f0 - e * df, df, (f1 - f0) / e))
#
#     exit()

    # normalise data such that alpha=0 => recon=0

    def refine(u, dummy=False):
        '''
        Compute primal-dual gap:

        F(u) = 1/2|Au-d|_2^2 + alpha*|u|_1

        \nabla F(u) = A^*(Au-d) + alpha*sign(u)
        |\nabla F(u)|^2 = |known|^2 + |unknown|^2

        A = PW^T
        '''
        if u.dof_map is not u.FS.dof_map:
            u = Haar_2D(u.FS.old2new(u.x, u.dof_map), u.FS)

        r = A(u) - data
        df = A.bwrd(r, like=u).asarray()  # known gradient of f

        dF = df.copy()
        _smallgrad(dF.reshape(-1), u.asarray().reshape(-1), alpha, pms['eps'])  # known gradient

        # unknown gradient of f = P^Tr on unknown indices
        # |unknown gradient|^2 = |P^Tr|^2-|A^*r|^2 = |r|^2 - |df|^2
        leaf_error = empty(df.shape[0], dtype=df.dtype)
        _unknown_grad_tomo(r, A.grid, A.d, A.centre, u.dof_map, u.is_leaf, leaf_error)

        eF = Norm(leaf_error[leaf_error > alpha] - alpha)

        pms['F'] = (.5 * (r ** 2).sum() + alpha * abs(u.asarray()).sum())
        pms['discrete_err'] = Norm(dF)
        pms['cont_err'] = (Norm(dF) ** 2 + eF ** 2) ** .5
        pms['dof'] = u.size
        pms['depth'] = u.FS.depth
        pms['efficiency'] = u.size / 4 ** u.FS.depth

        if dummy:
            return u

        # The coarsening works but is more delicate and doesn't hugely improve the number of degrees of freedom
        FS = u.FS
#         # Trim leaves with 0 intensity with 0 error even after coarsening
#         FS.trim(u.dof_map[(leaf_error >= -.5) * (leaf_error < alpha / 4) * (abs(u.asarray()).max(1) == 0)])
#         # Split leaves with larger continuous than discrete gradient
        FS.split(u.dof_map[(leaf_error - alpha > 2 * abs(dF).max(1))])

#         A.update(FS, refine)
        u = Haar_2D(FS.old2new(u.x, u.dof_map), FS)
        return u

#     def custom_stop(i, *_, d=None, extras=None, **__): return ((extras[0 if adaptive[1] else 1] < stop) and i > 2)
    def custom_stop(*_, d=None, extras=None, **__): return (d + extras[2] < 1e-12)

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


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", message='Adding an axes using the same arguments as a previous axes')
    warnings.filterwarnings("ignore", message='Data has no positive values, and therefore cannot be log-scaled.')
    from numpy import linspace
#     plt.figure(figsize=(18, 10))

    test = 5
    # 1  : 1D wavlet functionality test
    # 1.5: 1D wavlet plotting test
    # 2  : 1D denoising example
    # 3  : 2D wavelet functionality test
    # 3.5: 2D wavelet plotting test
    # 4  : Tomography operator test
    # 5  : Tomography recon example

    if test == 1:  # Check 1D wavelet space methods are working
        my_space = Haar_space()
        for i in range(14):
            my_space.split(list(my_space.tree))
        for i in range(11):
            my_space.trim(list(my_space.tree))
        print(my_space)
        my_space.trim([
            (2, 0), (2, 1),  # ignore
            (4, 0), (4, 1),  # complete
            (3, 3), (3, 4),  # ignore
            (4, 5), (4, 6),  # ignore
            ])
        print(my_space)
        my_space.split([
        (2, 0),  # ignore
        (3, 0),  # complete
        (4, 0),  # ignore
        (4, 5),  # complete
        (4, 2),  # complete
        ])
        print(my_space)

    elif test == 1.5:  # Check 1D plotting methods
        level = 10
        my_func = array([0] * (2 ** level // 3) + [1] * (2 ** level - 2 ** level // 3), dtype='float64')
        plt.plot([0, 1 / 3, 1 / 3, 1], [0, 0, 1, 1], 'k', linewidth=3)
        for init in (1, 3, 5, 7, level):
            my_wave = Haar_1D(0, Haar_space(init=init))
            new_wave = my_wave.from_discrete(my_func)
            new_wave.plot(level=level)
            plt.pause(1)
        print('This should be zero:', abs(new_wave.discretise(level) - my_func).max())
        plt.show()

    elif test == 2:  # 1D optimisation problem
        peaks = 10  # 10 or 100
        iters = ((1000, 10) if peaks > 10 else (150, 2))
        vid = {'filename':'haar_vid_1D_denoising_' + ('large' if peaks > 10 else 'small'), 'fps':int(iters[0] / iters[1] / 30) + 1}
        vid = None
        recon, record = Haar_denoising(peaks, .01, vid=vid, iters=iters)
        if vid is None:
            plt.show()

    elif test == 3:
        my_space = Haar_space(dim=2)
        for i in range(5):
            my_space.split(list(my_space.tree))
        for i in range(2):
            my_space.trim(list(my_space.tree))
        print(my_space)
        my_space.trim([
            (3, 2, 0), (3, 2, 1), (3, 3, 0), (3, 3, 1),  # ignore
            (4, 4, 0), (4, 4, 1), (4, 5, 0), (4, 5, 1),  # complete
            ])
        my_space.split([
        (2, 1, 1),  # ignore
        (3, 2, 2),  # ignore
        (4, 4, 4),  # complete
        ])
        print(my_space)

    elif test == 3.5:
        level = 10
        x = linspace(0, 1, 2 ** level)
        my_func = ((x.reshape(-1, 1) - .7) ** 2 + (x.reshape(1, -1) - .5) ** 2 < .4 ** 2).astype('float64')

        plt.subplot(131);plt.imshow(my_func, origin='lower', extent=[0, 1, 0, 1])
        ax1 = plt.subplot(132)
        ax2 = plt.subplot(133)
        for init in (1, 3, 5, 7, level):
            my_wave = Haar_2D(0, Haar_space(dim=2, init=init))
            new_wave = my_wave.from_discrete(my_func)
            ax1.cla()
            new_wave.plot(ax=ax1, level=level, vmin=0, vmax=1)
            plt.title('h = ' + str(2.** -init))
            ax2.cla()
            new_wave.plot_tree(ax=ax2, level=level)
            plt.tight_layout()
            plt.pause(2)
        print(abs(new_wave.discretise(level) - my_func).max())
        plt.show()

    elif test == 4:
        level = 5
        x = linspace(0, 1, 2 ** level)
        my_func = ((x.reshape(-1, 1) - .5) ** 2 + (x.reshape(1, -1) - .5) ** 2 < .3 ** 2).astype('float64')

        my_wave = Haar_2D(1, Haar_space(dim=2, init=level))
        my_wave = my_wave.from_discrete(my_func)

        A = Haar2Sino(linspace(0, 180, 50)[1:-1], linspace(-.5, .5, 50), my_wave.FS, centre=[.5] * 2)
        d = A(my_wave)
        new_wave = A.bwrd(d)

        plt.subplot(131); my_wave.plot(level=level)
        plt.subplot(132); A.plot(d)
        plt.subplot(133); new_wave.plot(level=level)
        plt.show()

    elif test == 5:
        from numpy import random
        random.seed(101)
        level, noise = 10, .05

        x = linspace(0, 1, 2 ** level)
        gt_arr = ((x.reshape(-1, 1) - .5) ** 2 + (x.reshape(1, -1) - .5) ** 2 < .3 ** 2).astype(float)
        gt = Haar_2D(1, Haar_space(dim=2, init=level)).from_discrete(gt_arr)

        A = Haar2Sino(linspace(0, 180, 52)[1:-1], linspace(-.5, .5, 51), Haar_space(2, dim=2), centre=[.5] * 2)
        data = A(gt)
        data += noise * Norm(data) * random.randn(*data.shape) / data.size ** .5

        iters = (10, 25)
        vid = {'filename':'haar_vid_2D_disc', 'fps':int(iters[0] / iters[1] / 30) + 1}
        vid = None
        recon, record = Haar_tomo(A, data, .0001, vid=vid, iters=iters, prnt=False)
        if vid is None:
            plt.show()

    elif test == 6:
        from numpy import random
        random.seed(101)
        level, noise = 10, .0
        from haar_bin import phantom

        gt_arr = phantom(2 ** level)
        gt = Haar_2D(1, Haar_space(dim=2, init=level)).from_discrete(gt_arr)

        A = Haar2Sino(linspace(0, 180, 52)[1:-1], linspace(-.5, .5, 51), Haar_space(2, dim=2), centre=[.5] * 2)
        data = A(gt)
        data += noise * Norm(data) * random.randn(*data.shape) / data.size ** .5

        iters = (1000, 10)
        vid = {'filename':'haar_vid_2D_shepp', 'fps':int(iters[0] / iters[1] / 30) + 1}
        vid = None

        recon, record = Haar_tomo(A, data, .00002, vid=vid, iters=iters)
        print('2', recon.size)
        if vid is None:
            plt.show()
