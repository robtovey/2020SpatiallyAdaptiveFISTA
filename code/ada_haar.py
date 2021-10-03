'''
Created on 7 Sep 2020

@author: Rob Tovey
'''

from haar_bin import (Radon_to_sparse_matrix, Radon_update_sparse_matrix, _smallgrad,
                      len_intersect, line_intersect_square, jit, prange, __pparams,
                      _FP_tomo, _BP_tomo, _gradF)
from matplotlib import pyplot as plt
from numpy import (array, empty, sqrt, zeros, cos, sin, pi, nanmin, nanmax, ones, random, log2)
from anytree import Node, PreOrderIter
from algorithms import shrink, stopping_criterion, faster_FISTA, \
    _makeVid, FB, Norm, FISTA
from lasso import op, sparse_matvec
from adaptive_spaces import Haar_FS as Haar_space, Haar_function as Haar
from scipy.sparse import csr_matrix


@jit(**__pparams)
def _unknown_grad_denoising(r, grid, L, dof_map, isleaf, out):
    '''
    df/du_j = sum_I <w_j,k_I>r_I
    Orthogonality of k_I gives:
    |df/du_j|^2 <= sum_I |w_j|^2 |k_I 1_{supp(w_j)}|^2 |r_I|^2
                 = sum_I |k_I 1_{supp(w_j)}|^2 |r_I|^2
    '''
    for i in prange(dof_map.shape[0]):
        if isleaf[i]:
            # only leaves have errors
            supp0, supp1 = dof_map[i, 0], dof_map[i, 0] + dof_map[i, 1]

            # P^Tr|_[g[k],g[k+1]] = r[k]/sqrt(L[k])
            total = 0
            for k in range(L.size):
                total += r[k] ** 2 * len_intersect(supp0, supp1, grid[k], grid[k + 1]) / L[k]

            out[i] = sqrt(total)
        else:
            out[i] = -1


@jit(**__pparams)
def _unknown_grad_tomo(res, g, slope, centre, dof_map, isleaf, out):
    '''
    df/du_j = sum_Psum_I <w_j,k_{P,I}>r_{P,I}
    Orthogonality of k_{P,I} and |w_j|=1 gives:
    |df/du_j| <= |w_j| sum_P |sum_I k_{P,I} 1_{supp(w_j)}|
    |sum_I k_{P,I} 1_{supp(w_j)}|^2 = sum_I |k_{P,I}1_{supp(w_j)}|^2
    '''
    for i in prange(dof_map.shape[0]):
        if isleaf[i]:
            h = dof_map[i, 2]
            b0x, b0y = dof_map[i, 0] - centre[0], dof_map[i, 1] - centre[1]
            b1x, b1y = b0x + h, b0y + h
            midx, midy = b0x + .5 * h, b0y + .5 * h
            r = 2.**(-.5) * h

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
    random.seed(101)

    grid = list(random.rand(n_points - 1))
    grid.sort()
    grid = array([0] + list(grid) + [1], dtype='float64')
    L = array([grid[i + 1] - grid[i] for i in range(n_points)])

    data = random.rand(n_points)

    pms = {'i':0, 'eps':1e-18}
    recon = Haar(0, Haar_space(2, dim=1))

    def residual(u):
        return array([u.x[0, 0] * sqrt(L[j])
                      +interval_integral(grid[j:j + 2], u.x[1:], u.dof_map) / sqrt(L[j]) - data[j]
             for j in range(n_points)], dtype='float64')

    def energy(u):
        refine(u, True)
        return [pms[t] for t in ('F', 'discrete_err', 'cont_err')]

    @jit(nopython=True, parallel=False, fastmath=True)
    def interval_integral(x, arr, dof_map):
        '''
        integrate wavelet arr over [x[0],x[1]]
        '''
        value = 0

        for i in range(arr.shape[0]):
            supp0 = dof_map[i, 0]
            h = dof_map[i, 1]
            supp1 = supp0 + h
            if supp0 < x[1] and supp1 > x[0]:
                midpoint, scale = .5 * (supp0 + supp1), (.5 / h) ** .5
                value += scale * arr[i, 0] * (len_intersect(midpoint, supp1, x[0], x[1])
                                  -len_intersect(supp0, midpoint, x[0], x[1]))
        return value

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
            u.plot(level=10, update=pms['recon plot'][0])
            pms['recon plot'][1].set_ydata(data2func(residual(u) + data))
            pms['recon plot'][2].set_ydata(data2func(data))
        ax.set_ylim(-.1, min(.8 * n_points, 1.1 * max(data2func(data))))

        ax = pms['axes'][1]
        if pms.get('recon stars', None) is None:
            pms['recon stars'] = u.plot_tree(ax=ax)  # , max_ticks=100)
            ax.set_title('Coefficients')
        else:
#             u.plot_tree(ax=pms['recon stars'], max_ticks=100, update=True)
            u.plot_tree(update=pms['recon stars'])

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
#             ax.set_ylim(.9 * nanmin(stop_crit.extras[:, 1]), 1.1 * nanmax(stop_crit.extras[:, 2]))

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
        _unknown_grad_denoising(r, grid, L, u.dof_map, u.isleaf, leaf_error)

        eF = Norm(leaf_error[leaf_error > alpha] - alpha)
#         print(eF, leaf_error.round(1))

        pms['F'] = (.5 * (r ** 2).sum() + alpha * abs(u.asarray()).sum())
        pms['discrete_err'] = Norm(dF)
        pms['cont_err'] = (Norm(dF) ** 2 + eF ** 2) ** .5

        if dummy:
            return u

        FS = u.FS
        # Trim leaves with 0 intensity with 0 error even after coarsening
        # Refine leaves with larger continuous than discrete gradient
#         FS.refine(leaf_error[1:] - alpha > abs(dF[1:]))
#         FS.refine(-(leaf_error >= -.5) * (leaf_error < alpha / 2) * (u.asarray() == 0))

        return Haar(FS.old2new(u.x, u.dof_map), FS)

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

# class Haar_2D(Haar):
#
#     def __init__(self, arr, FS, maps=None):
#         assert FS.dim == 2
#         Haar.__init__(self, arr, FS, maps)
#
#         '''
#         dof_map[d] = (level, index0, index1, index2)
#         arr[d] = (v0, v1, v2)
#
#         Define the three base wavelets:
#
#                 -1/2     x\in[0,1], y\in[0,2]   |             -1/2     x\in[0,2], y\in[0,1]
#     W_0(x, y) = +1/2     x\in[1,2], y\in[0,2],  |  W_1(x,y) = +1/2     x\in[0,2], y\in[1,2]
#                  0              else            |              0              else
#     ---------------------------------------------------------------------------------------
#                 -1/2     x\in[0,1], y\in[1,2]    or    x\in[1,2], y\in[0,1]
#     W_2(x, y) = +1/2     x\in[0,1], y\in[0,1]    or    x\in[1,2], y\in[1,2]
#                  0              else
#
#     f(x) = arr[0] + sum_{d,i} arr[d,i] * 2^d.level * W_i(2^d.level * x-d.index[i])
#
#
#                                     |---------------------------|
#                                     |             |             |
#                                     |  -v0+v1-v2  |  +v0+v1+v2  |
#     2*(v0*W_0 + v1*W_1 = v2*W_2) =  |             |             |
#                                     |---------------------------|
#                                     |             |             |
#                                     |  -v0-v1+v2  |  +v0-v1-v2  |
#                                     |             |             |
#                                     |---------------------------|
#         '''
#
#     def discretise(self, level):
#         out = empty((2 ** level,) * 2, dtype='float64')
#         _discretise2D(self.x[0, 0], self.x[1:], self.dof_map[1:], 1, out)
#         return out
#
#     def from_discrete(self, arr):
#         level = log2(arr.shape).round().astype(int)
#         assert level[0] == level[1]
#         level = level[0]
#         assert arr.shape[0] == 2 ** level and arr.shape[1] == 2 ** level
#
#         x = empty(self.x.shape, dtype='float64')
#         x[0, 0] = _from_discrete2D(arr, self.dof_map[1:], 1, x[1:])
#         x[0, 1:] = 0
#
#         return self.copy(x)
#
#     def plot(self, level, ax=None, update=False, origin='lower', extent=[0, 1, 0, 1], **kwargs):
#         y = self.discretise(level)
#         if update:
#             assert ax is not None
#             return ax.set_data(y)
#         ax = ax if ax is not None else plt
#         return ax.imshow(y, origin=origin, extent=extent, **kwargs)
#
#     def plot_tree(self, level=5, max_ticks=200, ax=None, spatial=True, update=False, **kwargs):
#         ax = ax if ax is not None else plt.gca()
#         eps = abs(self.x).max() * 1e-10
#
#         if spatial:
#             # place in middle of support (at jump)
#             x = (self.dof_map[:, 1:] + .5) * 2.**(1 - self.dof_map[:,:1])
#             x[0,:] = 0.01
#
#             # do plotting
#             lines = []
#             tmp, eps = log(abs(self.x).max(1) + eps), log(eps)
#             tmp, eps = (tmp - tmp.min()) / tmp.ptp(), (eps - tmp.min()) / tmp.ptp()
#             I = [i for i in range(tmp.size) if tmp[i] <= eps]
#             L = ax.hist2d(x[:, 0], x[:, 1], range=[[0, 1], [0, 1]],
#                                    bins=min(100, 1 + int(x.shape[0] ** .5 / 10)), density=True)
#             ax.clear()
#             lines.append(ax.imshow(L[0], origin='lower', interpolation='bicubic',
#                                    cmap='Blues', vmin=0, alpha=.5,
#                                    extent=[0, 1, 0, 1]))
#             from matplotlib.colors import LinearSegmentedColormap
#             cmap = {'red':((0, .1, .1), (1, 1, 1)),
#                     'blue':((0, .1, .1), (1, 0, 0)),
#                     'green':((0, .1, .1), (1, 0, 0))}
#             cmap = LinearSegmentedColormap('k2r', cmap)
#             if self.x.shape[0] < 50 ** 2:
#                 lines.append(ax.scatter(x[I, 1], x[I, 0], 2, tmp[I], '.', cmap=cmap, **kwargs))
#             I = [i for i in range(tmp.size) if tmp[i] > eps]
#             lines.append(ax.scatter(x[I, 1], x[I, 0], 40, tmp[I], '*', cmap=cmap, **kwargs))
#
#             ax.set_aspect('equal', 'box')
#             ax.set_xlim(0, 1)
#             ax.set_ylim(0, 1)
#             ax.set_xlabel('Jump location x')
#             ax.set_ylabel('Jump location y')
#             ax.set_title('Logarithmic intensity in red, density in blue')
#
#         else:
#
#             # place at start of support
#             x = self.dof_map[:, 1:] * 2.**(1 - self.dof_map[:,:1]).reshape(-1, 1)
#
#             # do plotting
#             I = [i for i in range(self.x.shape[0]) if abs(self.x[i]).max() <= eps]
#             lines = ax.plot(I, 0 * self.x[I, 0], '.', color='tab:orange', **kwargs)
#             for j in range(3):
#                 I = [i for i in range(self.x.shape[0]) if abs(self.x[i, j]) > eps]
#                 lines += ax.plot(I, self.x[I, j], '*', color='tab:blue', **kwargs)
#
#             # prepare x ticks
#             level = min(level, 5)  # it's just too expensive otherwise
#             label_spacing = int(self.dof_map.shape[0] / (4 * len(ax.get_xticklabels()))) + 1
#             tick_spacing = int(self.dof_map.shape[0] / max_ticks) + 1
#
#             grid = [[0, 0, ' ' * (level - 1) + '-', '  (0, 0)']]
#             for i in range(1, self.dof_map.shape[0]):
#                 L = self.dof_map[i, 0]
#                 if L < level:
#                     grid.append([L, i, ' ' * (level - L - 1) + '-' * (L + 1),
#                                  '  ' + str(tuple(x[i])) if any(x[i] > x[i - 1]) else ''])
#                 elif i % tick_spacing == 0:
#                     grid.append([level + 1, i, '-' * level, ''])
#
#             # refine x ticks
#             grid.sort(key=lambda n: n[0])  # sort by level
#             for i, g in enumerate(grid):
#                 if g[3] and all(abs(g[1] - y[1]) > label_spacing for y in grid[:i]):
#                     ax.axvline(g[1] - .5, linestyle='-', color='gray', linewidth=.5)
#                 elif any(abs(g[1] - y[1]) < tick_spacing for y in grid[:i]):
#                     g[3] = None
#                 else:
#                     g[3] = ''
#             grid = [g for g in grid if g[3] is not None]
#
#             # update x ticks
#             ax.set_xticks([g[1] for g in grid], [g[2] + g[3] for g in grid], rotation=-90)
#             ax.tick_params(which='both', length=0)
#             ax.set_xlabel('Wavelet index')
#             ax.set_ylabel('Coefficient value')
#
#         return lines
#
#     def _copy(self, arr, FS, maps): return Haar_2D(arr, FS, maps)


class Haar2Sino(op):

    def __init__(self, angles, grid, FS, centre=[0, 0], parallel=True):
        A = array(angles) * pi / 180
        directions = array([(cos(a), sin(a)) for a in A], dtype='float64')
        centre = array(centre, dtype='float64').reshape(2)

        op.__init__(self, self.fwrd, self.bwrd, out_sz=(len(angles), len(grid) - 1), norm=1)

        self.angles, self.d = angles, directions
        self.grid, self.centre = grid, centre
        self.FS, self._buf = FS, {'DM':None}
        self._parallel = parallel

        self.scale = self.fwrd(Haar(1, Haar_space(dim=2)))  # A good preconditioner to make each angle have norm 1
        self._norm = Norm(self.scale.max(1) ** .5)  # max/norm over orthogonal/non-orthogonal kernels

    def update(self, FS, refine=None):
        DM = FS.dof_map
        FS = FS.FS if hasattr(FS, 'FS') else FS
        self.FS = FS
        if DM is self._buf['DM']:
            return
        self._buf['DM'] = DM
        total_sz = int(self.d.size * self.grid.size * (1 + 4 * DM[:, 2].sum()))

        while True:
            ptr, indcs = empty(DM.size + 4, dtype='int32'), empty(total_sz, dtype='int32')
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
        self._buf['bwrd_mat'] = csr_matrix((data[:tot], indcs[:tot], ptr), shape=(3 + DM.shape[0] * 3, self.d.shape[0] * (self.grid.size - 1)))
        self._buf['fwrd_mat'] = self._buf['bwrd_mat'].T.tocsr()
        self._buf['fwrd_mat'].sort_indices()
        self._buf['data'], self._buf['indcs'], self._buf['ptr'] = data, indcs, ptr

    def fwrd(self, w):

        if w.dof_map is self._buf['DM']:
            if self._parallel:
                A = self._buf['fwrd_mat']
                vec = empty(A.shape[0], dtype=A.dtype)
                sparse_matvec(A.data, A.indices, A.indptr, w.ravel(), vec)
                return vec
            else:
                return self._buf['fwrd_mat'].dot(w.ravel())
        else:
            # s = empty(sinogram)
            s = empty(self._shape[1], dtype='float64')
            _FP_tomo(w.x, w.dof_map, self.grid, self.d, self.centre, s)
            return s

    def bwrd(self, s, like=None):
        dof_map = self.FS.dof_map if like is None else like.dof_map
        shape = 1 + dof_map.shape[0], 3

        if dof_map is self._buf['DM']:
            if self._parallel:
                A = self._buf['bwrd_mat']
                out = empty(shape, dtype=A.dtype)
                sparse_matvec(A.data, A.indices, A.indptr, s.ravel(), out.ravel())
            else:
                out = self._buf['bwrd_mat'].dot(s.ravel()).reshape(shape)
        else:
            s = s.reshape(self._shape[1])
            out = empty(shape, dtype='float64')
            _BP_tomo(s, dof_map, self.grid, self.d, self.centre, out)

        if like is None:
            return Haar(out, self.FS)
        else:
            return like.copy(out)

    def plot(self, s, ax=None, update=False, **kwargs):
        if update:
            assert ax is not None
            return ax.set_data(s.T)
        ax = ax if ax is not None else plt
        return ax.imshow(s.T, origin='lower', aspect='auto',
                         extent=[self.angles.min(), self.angles.max(), self.grid.min(), self.grid.max()], **kwargs)


def Haar_tomo(A, data, alpha, huber, iters=None, prnt=True, plot=True, vid=None, algorithm='Greedy', gt=None):
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
    random.seed(101)

    pms = {'i':0, 'eps':1e-18, 'refineI':2}
    recon = Haar(0, Haar_space(2, dim=2))
    A.update(recon.FS)

    from numba import vectorize
    @vectorize(identity=0, nopython=True)
    def Huber(x):
        x = abs(x)
        if x < huber:
            return .5 * x * x
        else:
            return huber * (x - .5 * huber)
    @vectorize(identity=0, nopython=True)
    def Huber_grad(x):
        if abs(x) < huber:
            return x
        elif x > 0:
            return huber
        else:
            return -huber
    alpha = alpha * huber
#     @vectorize(identity=0, nopython=True)
#     def Huber(x): return .5 * x * x
#     @vectorize(identity=0, nopython=True)
#     def Huber_grad(x): return x

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
        return A.bwrd(Huber_grad(A(u) - data), like=u)

    def proxG(u, t):
        U = u.copy(0)
        shrink(u.x.reshape(-1, 1), t * alpha, U.x.reshape(-1, 1))
        return U

    def doPlot(i, u, fig, plt):
#         print('energy = %.2f, reg = %.2f' % (Huber(A(u) - data).sum() / Huber(A(gt) - data).sum(),
#                                            abs(u.x[1:]).sum() / abs(gt.x[1:]).sum()),
#         u.x.size, -log2(u.FS.h))

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
            u.plot(level=8, update=pms['recon plot'])

        u.plot_tree(ax=pms['axes'][2], update=True)
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
            ax.set_ylabel('Sup-norm of gradient')
        else:
            pms['error plots'][0].set_data(1 + stop_crit.I, stop_crit.extras[:, 1])
            pms['error plots'][1].set_data(1 + stop_crit.I, stop_crit.extras[:, 2])
            ax.set_xlim(1, max(i, 2))
            ax.set_ylim(.9 * nanmin(stop_crit.extras[:, 1]), 1.1 * nanmax(stop_crit.extras[:, 2]))

        if i < 2:
            fig.tight_layout()

    # Gradient check:
#     recon.x[:] = random.randn(*recon.x.shape)
#     dx = recon.copy(); dx.x[:] = random.randn(*recon.x.shape)
#     f = lambda u: Huber(residual(u)).sum()
#     f0, df = f(recon), gradF(recon).inner(dx)
#     for e in [10 ** (-i) for i in range(7)]:
#         f1 = f(recon + e * dx)
#         print('e = %.1e, f-Pf = %+.1e, f-P\'f = %+.1e, Df = %+.1e ~ %+.1e' % (e, f1 - f0, f1 - f0 - e * df, df, (f1 - f0) / e))
#
#     exit()

    def refine(u, dummy=False):
        '''
        Compute primal-dual gap:

        F(u) = 1/2|Au-d|_2^2 + alpha*|u|_1

        \nabla F(u) = A^*(Au-d) + alpha*sign(u)
        |\nabla F(u)|^2 = |known|^2 + |unknown|^2

        A = PW^T
        '''
        if u.dof_map is not u.FS.dof_map:
            u = Haar(u.FS.old2new(u.x, u.dof_map), u.FS)

        r = A(u) - data
        Hr = Huber_grad(r)
        df = A.bwrd(Hr, like=u).asarray()  # known gradient of f

        dF = df.copy()
        _smallgrad(dF.ravel(), u.ravel(), alpha, pms['eps'])  # known gradient

        # unknown gradient of f = P^Tr on unknown indices
        # |unknown gradient|^2 = |P^Tr|^2-|A^*r|^2 = |r|^2 - |df|^2
        leaf_error = empty(df.shape[0], dtype=df.dtype)
        _unknown_grad_tomo(Hr, A.grid, A.d, A.centre, u.dof_map, u.isleaf, leaf_error[1:])
        leaf_error[0] = -1

        pms['F'] = (Huber(r).sum() + alpha * abs(u.asarray()).sum())
        pms['discrete_err'] = dF.max()
        pms['cont_err'] = max(dF.max(), (leaf_error - alpha).max())
        pms['dof'] = (u.size - 2)
        pms['depth'] = 1 - log2(u.FS.h)
        pms['efficiency'] = (u.size - 2) * .25 * u.FS.h ** 2

        if dummy:
            return u

        FS = u.FS
        ref = (leaf_error - alpha > 10 * abs(dF).max(1)).astype('float64')
        ref[0] = 0; ref[1:] *= FS.isleaf * (FS.H > 2 ** -10)
# #         # Trim leaves with 0 intensity with 0 error even after coarsening
# #         FS.refine(-(leaf_error >= -.5) * (leaf_error < alpha / 4) * (abs(u.asarray()).max(1) == 0))
# #         # refine leaves with larger continuous than discrete gradient
        FS.refine(ref[1:])
#         print(-log2(u.FS.h), u.size)
#         print(((leaf_error >= -.5) * (leaf_error < alpha / 4) * (abs(u.asarray()).max(1) == 0)).sum() / u.size)
#
        A.update(FS, ref)
        u = Haar(FS.old2new(u.x, u.dof_map), FS)
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
        recon = faster_FISTA(recon, huber / A.norm ** 2, gradF, proxG, stop_crit, **default)
    elif algorithm[0] is 'FISTA':
        default = {'a':10, 'restarting':False}
        default.update(algorithm[1])
        recon = FISTA(recon, huber / A.norm ** 2, gradF, proxG, stop_crit, **default)
    elif algorithm[0] is 'FB':
        default = {'scale':2}
        default.update(algorithm[1])
        recon = FB(recon, default['scale'] * huber / A.norm ** 2, gradF, proxG, stop_crit)

    return recon, stop_crit


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore", message='Adding an axes using the same arguments as a previous axes')
    warnings.filterwarnings("ignore", message='Data has no positive values, and therefore cannot be log-scaled.')
    from numpy import linspace
#     plt.figure(figsize=(18, 10))

    test = 4
    # 1  : 1D wavlet functionality test
    # 1.5: 1D wavlet plotting test
    # 2  : 1D denoising example
    # 3  : 2D wavelet functionality test
    # 3.5: 2D wavelet plotting test
    # 4  : Tomography operator test
    # 5  : Tomography recon example

    if test == 1:  # Check 1D wavelet space methods are working
        my_space = Haar_space()
        for i in range(2):
            my_space.refine(ones(my_space.size))
#         for i in range(11):
#             my_space.refine(-ones(my_space.size))
        print(my_space.dof_map)
        my_space.refine([1, 1, 0, -1, 0, -1, 1])
        print(my_space.dof_map)
        assert my_space.dof_map.shape == (9, 2)
        assert (my_space.dof_map[2] == array((0, .25))).all()
        assert (my_space.dof_map[-1] == array((.875, .125))).all()

    elif test == 1.5:  # Check 1D plotting methods
        level = 10
        my_func = array([0] * (2 ** level // 3) + [1] * (2 ** level - 2 ** level // 3), dtype='float64')
        plt.plot([0, 1 / 3, 1 / 3, 1], [0, 0, 1, 1], 'k', linewidth=3)
        for init in (1, 3, 5, 7, level):
            my_wave = Haar(0, Haar_space(dim=1, init=init))
            new_wave = my_wave.from_discrete(my_func)
            new_wave.plot(level=level)
            plt.pause(1)
        print('This should be zero: %e' % abs(new_wave.discretise(level) - my_func).max())
        plt.show()

    elif test == 2:  # 1D optimisation problem
#         raise NotImplementedError
        peaks = 10  # 10 or 100
        iters = ((1000, 10) if peaks > 10 else (150, 2))
        vid = {'filename':'haar_vid_1D_denoising_' + ('large' if peaks > 10 else 'small'), 'fps':int(iters[0] / iters[1] / 30) + 1}
        vid = None
        recon, record = Haar_denoising(peaks, .01, vid=vid, iters=iters)
        if vid is None:
            plt.show()

    elif test == 3:
        my_space = Haar_space(dim=2)
        for i in range(1):
            my_space.refine(ones(my_space.size))
#         for i in range(3):
#             my_space.refine(-ones(my_space.size))
        print(my_space.dof_map)
        my_space.refine([1, 0, 1, -1, -1])
        print(my_space.dof_map)
        assert (my_space.dof_map[:, -1] == array([1, .5, .5, .25, .25, .25, .25, .5, .5])).all()

    elif test == 3.5:
        level = 10
        x = linspace(0, 1, 2 ** level)
        my_func = ((x.reshape(-1, 1) - .7) ** 2 + (x.reshape(1, -1) - .5) ** 2 < .4 ** 2).astype('float64')

        plt.subplot(131); plt.imshow(my_func, origin='lower', extent=[0, 1, 0, 1])
        ax1 = plt.subplot(132)
        ax2 = plt.subplot(133)
        for init in (1, 3, 5, 7, level):
            my_wave = Haar(0, Haar_space(dim=2, init=init))
            new_wave = my_wave.from_discrete(my_func)
            ax1.cla()
            new_wave.plot(ax=ax1, level=level, vmin=0, vmax=1)
            plt.title('h = ' + str(2.** -init))
            ax2.cla()
            new_wave.plot_tree(ax=ax2)
            plt.tight_layout()
            plt.pause(2)
        print(abs(new_wave.discretise(level) - my_func).max())
        plt.show()

    elif test == 4:
        level = 6
        x = linspace(0, 1, 2 ** level)
        my_func = ((x.reshape(-1, 1) - .5) ** 2 + (x.reshape(1, -1) - .5) ** 2 < .3 ** 2).astype('float64')

        my_wave = Haar(1, Haar_space(dim=2, init=level))
        my_wave = my_wave.from_discrete(my_func)
#         my_wave.x[:] = 0
#         my_wave.x[2, 1] = 1
#         my_func = my_wave.discretise(level=6)
#         my_wave = my_wave.from_discrete(my_func)

#         print(abs(my_wave.discretise(level=6) - my_func).max())
#         plt.subplot(121).imshow(my_func, origin='lower')
#         plt.subplot(122); my_wave.plot(level=6)
#         plt.show()
#         exit()

        A = Haar2Sino(linspace(0, 180, 50)[1:-1], linspace(-.5, .5, 2 ** 5), my_wave.FS, centre=[.5] * 2)

        d = A(my_wave)
        new_wave = A.bwrd(d)
        A.update(Haar(1, Haar_space(dim=2, init=1)))
        A.update(my_wave.FS)
        print('matrix error: %.3e, %.3e' % (abs(A.bwrd(d).x - new_wave.x).max(), abs(A(my_wave) - d).max()))

        plt.subplot(131); my_wave.plot(level=5)
        plt.subplot(132); A.plot(d)
        plt.subplot(133); new_wave.plot(level=level)
        plt.show()

    elif test == 5:
        random.seed(101)
        level, noise = 10, .05

        x = linspace(0, 1, 2 ** level)
        gt_arr = ((x.reshape(-1, 1) - .5) ** 2 + (x.reshape(1, -1) - .5) ** 2 < .3 ** 2).astype(float)
        gt = Haar(1, Haar_space(dim=2, init=level)).from_discrete(gt_arr)

        A = Haar2Sino(linspace(0, 180, 52)[1:-1], linspace(-.5, .5, 51), Haar_space(2, dim=2), centre=[.5] * 2)
        data = A(gt)
#         data += noise * Norm(data) * random.randn(*data.shape) / data.size ** .5 # Gaussian noise
        data += Norm(data) * random.laplace(0, noise, data.shape) * (.5 / data.size) ** .5  # Laplacian noise

        iters = (1000, 1.1)
        vid = {'filename':'haar_vid_2D_disc', 'fps':int(iters[0] / iters[1] / 30) + 1}
        vid = None
        recon, record = Haar_tomo(A, data, .1, .002, vid=vid, iters=iters, prnt=False)
        if vid is None:
            plt.show()

    elif test == 6:
        random.seed(101)
        level, noise = 10, .0
        from haar_bin import phantom

        gt_arr = phantom(2 ** level)
        gt = Haar(1, Haar_space(dim=2, init=level)).from_discrete(gt_arr)

        A = Haar2Sino(linspace(0, 180, 52)[1:-1], linspace(-.5, .5, 51), Haar_space(2, dim=2), centre=[.5] * 2)
        data = A(gt)
#         data += noise * Norm(data) * random.randn(*data.shape) / data.size ** .5 # Gaussian noise
        data += Norm(data) * random.laplace(0, noise, data.shape) * (.5 / data.size) ** .5  # Laplacian noise

        iters = (1000, 1.1)
        vid = {'filename':'haar_vid_2D_shepp', 'fps':int(iters[0] / iters[1] / 30) + 1}
        vid = None

        recon, record = Haar_tomo(A, data, .1, .001, vid=vid, iters=iters)
        if vid is None:
            plt.show()
