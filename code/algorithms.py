'''
Created on 10 Oct 2019

@author: Rob
'''
from numpy import ndarray, inner, minimum, prod, array, empty, inf, zeros, nan, \
    isscalar, isnan, logical_not, log
from numpy.linalg import norm
from scipy.sparse.linalg.interface import LinearOperator
from time import time

def Inner(x, y):
    x = x.asarray() if hasattr(x, 'asarray') else x
    y = y.asarray() if hasattr(y, 'asarray') else y
    if isinstance(x, ndarray):
        IP = inner(x.reshape(-1), y.reshape(-1))
    elif hasattr(x, 'InnerProduct'):
        IP = x.InnerProduct(y)
    elif hasattr(x, 'FS'):
        return x.FS.inner(x, y)
    elif hasattr(x, 'asarray'):
        IP = inner(x.asarray().ravel(), y.asarray().ravel())
    else:
        raise ValueError(
            "Don't know the inner product between objects of type %s" % repr(type(x)))
    return IP


def Norm(x):
    if isinstance(x, ndarray):
        return norm(x.reshape(-1)) if x.size > 0 else 0
    elif hasattr(x, 'Norm'):
        return x.Norm()
    elif hasattr(x, 'FS'):
        return x.FS.norm(x)
    elif hasattr(x, 'asarray'):
        return norm(x.asarray().ravel())
    elif hasattr(x, '__iter__'):
        return sum(Norm(xx) ** 2 for xx in x) ** .5
    else:
        raise ValueError(
            "Don't know the inner product between objects of type %s" % repr(type(x)))


def FISTA(x, gamma, gradF, proxG, stop, a=10, restarting=False):
    n, X, mom = 0, x, 0 * x
    while True:
        alpha = n / (n + a + 1)
        y = (1 + alpha) * X - alpha * x
        X, x = proxG(y - gamma * gradF(y), gamma), X
        mom = X - x
        n += 1

        if stop(n, X, x, mom):
            return X

        if restarting and acute(y - X, mom):
            # Can probably do something smarter here
            # Look at Jingwei's 'Improving FISTA' paper
            n, x, mom = 0, X, 0 * X


def FB(x, gamma, gradF, proxG, stop, line_search=None):
    n, X, mom = 0, x, 0 * x
    while True:
        X, x = proxG(X - gamma * gradF(X), gamma), X
        if line_search is not None:
            X = line_search(X, X - x)
        mom = X - x
        n += 1

        if stop(n, X, x, mom):
            return X


def faster_FISTA(x, gamma, gradF, proxG, stop, xi=0.95, S=1, line_search=None):
    ''' This is Algorithm 5, greedy FISTA from Jingwei's
    'faster, smarter, greedier' paper'''
    n, X, mom = 0, x, 0 * x
    gamma0, gamma, T = gamma, 1.3 * gamma, None
    while True:
        y = X + mom
        X, x = proxG(y - gamma * gradF(y), gamma), X
        if line_search is not None:
            X = line_search(X, X - x)
        mom = X - x
        n += 1

        if stop(n, X, x, mom):
            return X

        if acute(y - X, mom):
            X, mom = x, 0 * x
#             T = None
        elif T is None:
            T = Norm(X - x)
        elif (gamma > gamma0) and (Norm(X - x) > S * T):
            gamma = max(gamma * xi, gamma0)


def DR(x, gamma, proxF, proxG, stop, a=1):
    n, X, mom = 0, x, 0 * x
    S = 0 * x
    # a = 2 generates classic DR
    # If the last coordinate of p=1 then it is PRS

    while True:
        X, x = proxG(S, gamma), X
        S = a * (proxF(2 * X - S, gamma) - X) + S
        mom = X - x
        n += 1

        if stop(n, X, x, mom):
            return X


def AP(x, y, gamma, proxF, proxG, stop):
    n, X, mom = 0, x, 0 * x
    Y = y

    while True:
        X, x = proxF(X, gamma, Y), x
        Y, y = proxG(Y, gamma, X), y
        mom = X - x
        n += 1

        if stop(n, X, x, mom):
            return X


def GD(x, gamma, grad, line_search, stop):
    n, X = 0, x
    while True:
        v = grad(X)
        v = (-gamma / (Norm(v) * (n + 1))) * v
        X, x = line_search(X + v, v), X
        n += 1

        if stop(n, X, x, X - x):
            return X


def Frank_Wolfe(x, gamma, search_dir, stop, line_search=None):
    n, X = 0, x
    if line_search is None:
        line_search = lambda u, v:u + (gamma / (n + 1)) * v
    while True:
        v = search_dir(X)
        # This is proper Frank-Wolfe:
        X, x = line_search(X, v - X), X
        n += 1

        if stop(n, X, x, X - x):
            return X


def acute(x, y):
    IP = Inner(x, y)
    return (IP.real + IP.imag) > 0


pass
##########
# Numba back-end for finite differences and other operators
##########
try:
    import numba

    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def shrink(y, scale, out):
        for i in numba.prange(y.shape[0]):
            n = 0
            for j in range(y.shape[1]):
                n += y[i, j] * y[i, j]
            if n < scale * scale:
                for j in range(y.shape[1]):
                    out[i, j] = 0
            else:
                n = 1 - scale * (n ** -.5)
                for j in range(y.shape[1]):
                    out[i, j] = n * y[i, j]

    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def shrink_vec(y, scale, out):
        for i in numba.prange(y.shape[0]):
            n, s = 0, scale[i]
            for j in range(y.shape[1]):
                n += y[i, j] * y[i, j]
            if n < s * s:
                for j in range(y.shape[1]):
                    out[i, j] = 0
            else:
                n = 1 - s * (n ** -.5)
                for j in range(y.shape[1]):
                    out[i, j] = n * y[i, j]

    @numba.jit(nopython=True, parallel=True, fastmath=True)
    def proj(y, scale, out):
        for i in numba.prange(y.shape[0]):
            n = 0
            for j in range(y.shape[1]):
                n += y[i, j] * y[i, j]
            if n <= scale * scale:
                for j in range(y.shape[1]):
                    out[i, j] = y[i, j]
            else:
                n = scale * (n ** -.5)
                for j in range(y.shape[1]):
                    out[i, j] = n * y[i, j]

except Exception:
    from numpy import maximum

    if 'shrink' not in globals():

        def shrink(y, scale, out):
            n = norm(y, 2, -1, True) + 1e-8
            n = maximum(0, 1 - scale / n)
            out[...] = y * n

    if 'proj' not in globals():

        def proj(y, scale, out):
            n = maximum(1, norm(y, 2, -1, True) / scale)
            out[...] = y / n[..., None]

    pass


class stopping_criterion:

    def __init__(self, maxiter=1000, stepsize=0, frequency=1, energy=None, prnt=True,
                 record=False, callback=lambda i, X, f, plt:None, vid=None, scaling=None, fig=None):
        self.maxiter = maxiter
        self.stepsize = stepsize
        self.frequency = frequency
        self.energy = energy
        self.prnt = prnt
        self.record = record or prnt
        if scaling is None:
            scaling = 'linear' if type(frequency) is int else 'exponential'
        self.iter, self.iiter, self.nextiter = 0, 0, 0
        self.scaling = scaling
        if self.record:
            self.I = []
            i, m = [0] * 2, 0
            while True:
                if i[0] >= i[1]:
                    self.I.append(i[0])
                    m += 1
                    i[1] = (i[1] + frequency) if scaling == 'linear' else max(i[1] + 1, i[1] * frequency)
                    i[1] = min(i[1], maxiter - 1)
                if i[0] >= maxiter:
                    break
                else:
                    i[0] += 1
            if self.energy:
                self.E = empty(m) * nan
            self.d = empty(m) * nan
            self.I = array(self.I, dtype=int)
            self.T = empty(m) * nan
        self.callback = callback
        if fig is None:
            self.fig, self._vid, self.__vid = None, vid, (None, None)
        else:
            from matplotlib import pyplot
            self.fig = fig
            self._vid = vid
            if not vid:
                self.__vid = _makeVid(self.fig, stage=1, record=False), pyplot
            else:
                self.__vid = _makeVid(self.fig, stage=1, record=True, **vid), pyplot

    def __call__(self, i, X, x, m):
        I = self.iiter
        if self.iter >= self.nextiter:
            self.iiter += 1
            if self.scaling == 'linear':
                self.nextiter += self.frequency
            else:
                self.nextiter = max(self.nextiter + 1, self.nextiter * self.frequency)
            self.nextiter = min(self.nextiter, self.maxiter - 1)

            if self.record:
                if self.energy:
                    energy = self.energy(X)
                    if isscalar(energy):
                        energy = (energy,)
                    self.E[I] = energy[0]
                    if len(energy) > 1:
                        if not hasattr(self, 'extras'):
                            self.extras = empty((self.E.size, len(energy)))
                            self.extras.fill(nan)
                        self.extras[I, :] = energy
                self.d[I] = Norm(m)
                self.I[I] = self.iter
                self.T[I] = time()
            if self.prnt:
                if i < self.frequency:
                    print(' Iter   |x^n-x^{n-1}|       E(x^n)')
                print('%4d\t% 1.5e\t% 1.5e' % (self.iter,
                                               self.d[I],
                                               self.E[I]))  # TODO: might not have E attribute

            self.callback(self.iter, X, self.fig, self.__vid[1])
            if self.fig is not None:
                _makeVid(self.__vid, stage=2, record=self._vid)

        b = False
        if self.iter >= self.maxiter:
            b = True
        elif not isscalar(self.stepsize):
            b = self.stepsize(i, X, x, m,
                              d=(self.d[I] if self.record else None),
                              E=(self.E[I] if hasattr(self, 'E') else None),
                              extras=(self.extras[I] if hasattr(self, 'extras') else None))
        elif self.stepsize > 0:
            n = Norm(m) / max(1e-10, Norm(X))
            b = (n < self.stepsize) and (i > 1)
        self.iter += 1

        if b:
            if self.record:
                if isnan(self.d[I]):
                    I -= 1
                for arr in (self.d, self.T, self.E):
                    arr[I + 1:] = arr[I]
                self.T -= self.T.min()
                if hasattr(self, 'extras'):
                    self.extras[I + 1:, :] = self.extras[I, :]
            if self.fig is not None:
                _makeVid(self.__vid, stage=3, record=self._vid)
        return b


def _makeVid(*args, filename=None, stage=0, fps=5, record=False):
    '''
    stage 0: args=[], returns figure
        sets graphics library
    stage 1: args=[figure, title], returns writer
        initiates file/writer
    stage 2: args=[writer, pyplot]
        records new frame
    stage 3: args=[writer, pyplot]
        saves video
    '''
    if stage == 0:
        import matplotlib
        if record:
            matplotlib.use('Agg')
        if matplotlib.pyplot.fignum_exists('makevid'):
            F = matplotlib.pyplot.figure('makevid')
        else:
            F = matplotlib.pyplot.figure('makevid', figsize=(18, 10))
        F.clear()
        if not record:
            matplotlib.pyplot.pause(0.001)
        return F

    if record:
        if stage == 1:
            from matplotlib import animation as mv
            if len(args) == 2:
                filename = args[1]
            writer = mv.writers['ffmpeg'](fps=fps, metadata={'title': filename})
            writer.setup(args[0], filename + '.mp4', dpi=100)
            return writer
        elif stage == 2:
            if len(args) == 1:
                args = args[0]
            args[1].draw()
            args[0].grab_frame()
        elif stage == 3:
            if hasattr(args[0], '__len__'):
                args = args[0]
            args[0].finish()
    elif stage == 1:
#         args[0].draw()
        args[0].show()
        return args[0]
    elif stage == 2:
        if len(args) == 1:
            args = args[0]
        args[0].canvas.draw()
#         args[0].canvas.update()
        args[0].canvas.flush_events()
#         args[1].pause(.01)


if __name__ == '__main__':
    from skimage.data import moon, binary_blobs
    from matplotlib.pyplot import *
    from numpy import random, maximum
    alg = DR
    random.seed(1)
    gt = binary_blobs(length=128, n_dim=2,
                      volume_fraction=0.1, seed=1).astype('f4')

    data = gt + random.randn(*gt.shape)
    subplot(221); imshow(gt, vmin=-.3, vmax=1.3)
    subplot(222); imshow(data, vmin=-.3, vmax=1.3)

    if alg is FISTA:
        # dual L1 denoising: f(x) = 1/2|u-data|^2, g(u) = constraint(|u|\leq lambda)
        L = .5

        def energy(u): return Norm(u - data.reshape(-1)) ** 2 / 2

        def gradF(u): return (u - data.reshape(-1))

        def proxG(u, g):
            proj(u.reshape(-1, 1), L, u.reshape(-1, 1))
            return u

        stop = stopping_criterion(
            10000, 1e-16, frequency=100, prnt=True, record=True, energy=energy)
        recon = FISTA(0 * data.reshape(-1), 1,
                      gradF, proxG,
                      stop, a=2, restarting=True)
        recon = data - recon.reshape(gt.shape)
        print(Norm(recon - data), Norm(gt - data))
    elif alg is DR:
        # Projection: f(x) = 1/2|u-data|^2, g(u) = constraint(|u|\leq lambda)
        L = .5

        def energy(u): return Norm(u - data.reshape(-1)) ** 2 / 2

        def proxF(u, g): return (u + g * data.reshape(-1)) / (1 + g)

        def proxG(u, g):
            proj(u.reshape(-1, 1), L, u.reshape(-1, 1))
            return u

    #     def proxG(u, g): u[u > g] = g; u[u < -g] = -g; return u

        stop = stopping_criterion(
            100, 1e-16, prnt=True, record=True, energy=energy)
        recon = DR(0 * data.reshape(-1), 1,
                   proxF, proxG, stop)
        recon = recon.reshape(gt.shape)

    subplot(223)
    imshow(recon, vmin=-.3, vmax=1.3)
    subplot(224)
    semilogy(stop.d)
    semilogy(stop.E - stop.E[-10:].min())
    pause(.001)
    show()
