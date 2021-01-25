'''
Created on 16 Jun 2020

@author: Rob Tovey
'''
from skimage.io import imread
from lasso import FTYPE, kernelMapMesh2D, sqrt, log, Lasso, sparse_mesh, sparse_func
from sparse_bin import _merge2D
from numpy import load, savez, isscalar, linspace, array, empty
from numba import jit, prange
from numba.typed import List as numba_List
from affinity import getaffinity, setaffinity
import re, os
HOME_DIR = os.path.dirname(__file__)


def List(iterable):
    x = numba_List()
    for i in iterable:
        x.append(i)
    return x


def res2func(x, DM, i, arr):
    FS = sparse_mesh(init=DM, dim=2)
    return sparse_func(x, FS), i, arr


def set_result(results, recon, record, adaptive, iters, index):
    if hasattr(record, 'extras'):
        arr = record.extras.T.copy()
        arr[7:9] = 1 - arr[7:9]
        i = record.I + 1
    else:
        arr = record[0].T.copy()
        i = record[1] + 1
    if hasattr(recon, 'x'):
        x, DM = recon.x, recon.dof_map
    else:
        x, DM = recon
    results[tuple(iters) + tuple(adaptive) + (index,)] = x, DM, i, arr
    return res2func(x, DM, i, arr)


def get_result(results, adaptive=None, iters=None, index=None):
    check = []
    if iters is not None:
        if isscalar(iters):
            check.append((0, iters))
        else:
            check.append((0, iters[0]))
            check.append((1, iters[1]))
    if adaptive is not None:
        if isscalar(adaptive):
            check.append((2, adaptive))
        else:
            check.append((2, adaptive[0]))
            check.append((3, adaptive[1]))
    if index is not None:
        check.append((4, index))
    sublist = [r for r in results
               if all(r[k] == v for k, v in check)]
    if len(sublist) == 0:
        return None
    elif len(sublist) == 1:
        return res2func(*results[sublist[0]])
    else:
        return [res2func(*results[r]) for r in sublist]


def run(cores, in_file, out_file, adaptive, iters, w, scale, stop, index):
    data = imread(os.path.join(HOME_DIR,'STORM_data','sequence-as-stack-MT4.N2.HD-2D-Exp.tif')).astype(FTYPE) / 2000 - 0.07
#     data = imread(in_file).astype(FTYPE)
    index = index[0], min(index[1], data.shape[0])
    data = data[index[0]:index[1]]
    out = os.path.join(HOME_DIR,'STORM_data',out_file)

    setaffinity(cores=cores)
    A = kernelMapMesh2D(x=linspace(0, 1, 64 + 1)[:-1] + 1 / 128,
                    sigma=2 / 64 / sqrt(2 * log(2)), Norm=63.593735, _multicore=(len(getaffinity()) > 1))

    try:
        with load(out, allow_pickle=True) as tmp:
            results = tmp['results'].item(0).copy()
    except Exception:
        results = {}
        savez(out, results=results)

    for i, I in enumerate(range(index[0], index[1])):
        if get_result(results, adaptive, iters, I) is None:
            recon, record = Lasso(A, data[i], w, adaptive, iters, scale=scale, stop=stop,
                                  vid=None, plot=False, prnt=False)
            set_result(results, recon, record, adaptive, iters, I)
        print(str(i) + '\r', flush=True)
        if (i + 1) % 100 == 0:
            with load(out, allow_pickle=True) as tmp:
                results.update(tmp['results'].item(0).copy())
            savez(out, results=results)
            results = {}

    with load(out, allow_pickle=True) as tmp:
        results.update(tmp['results'].item(0).copy())
    savez(out, results=results)


def merge_results(chunks, directory=''):
    if len(directory) > 0:
        little_files = [os.path.join(HOME_DIR,'STORM_data', directory, 'store_%d.npz' % i) for i in range(chunks)]
    else:
        little_files = [os.path.join(HOME_DIR,'STORM_data', 'store_%d.npz' % i) for i in range(chunks)]
    results = {}
    for f in little_files:
        with load(f, allow_pickle=True) as tmp:
            results.update(tmp['results'].item(0).copy())

    tmp = {}
    for k, v in results.items():
        ind = ((v[0] > 1e-10 * v[0].max()) * (v[0] > 1e-16)).ravel()
        tmp[k] = (v[0][ind], v[1][ind], v[2], v[3]) # same arrays with zero pixels removed
    results = tmp

    x, DM = List(v[0].ravel() for _, v in results.items()), List(v[1] for _, v in results.items())
    while len(x) > 1:
        N = len(x) // 2
        sz = array([x[2 * i].size + x[2 * i + 1].size for i in range(N)], dtype=int)
        new_x = List(empty(sz[i], dtype=FTYPE) for i in range(N))
        new_DM = List(empty((sz[i], 3), dtype=FTYPE) for i in range(N))
    
        _batch_merge(x, DM, new_x, new_DM, sz)
    
        new_x = List(r[:sz[i]] for i, r in enumerate(new_x))
        new_DM = List(r[:sz[i]] for i, r in enumerate(new_DM))
        if 2 * len(new_x) < len(x):
            new_x.append(x[-1])
            new_DM.append(DM[-1])
        x, DM = new_x, new_DM

    return sparse_func(x[0].reshape(-1,1), sparse_mesh(init=DM[0], dim=2)), results


@jit(parallel=True, nopython=True, cache=True)
def _batch_merge(old, old_DM, new, new_DM, sz):
    for i in prange(len(new)):
        sz[i] = _merge2D(old[2 * i], old_DM[2 * i],
                         old[2 * i + 1], old_DM[2 * i + 1],
                         new[i], new_DM[i])


def params2commands(in_file=None, adaptive=None, iters=None, w=None, scale=None, stop=1e-10, chunks=1, max_index=None, directory=''):
    assert all(t is not None for t in (in_file, adaptive, iters, w, scale, max_index))

    cores = [str(s) for s in range(os.cpu_count())] # each core
    if chunks == 1:
        cores = [','.join(cores)] # all cores
        index = [0, max_index]
    else:
        index, j = [0], 0
        while index[-1] < max_index:
            index = [i * (max_index // chunks + j) for i in range(chunks + 1)]
            j += 1
        index[-1] = max_index
    os.makedirs(os.path.join('STORM_data',directory), exist_ok=True)
    in_file, out_file = '0', [os.path.join(directory,'store_%d.npz') % i for i in range(chunks)]
    adaptive = '%d %d' % (adaptive[0], int(adaptive[1]))
    iters = '%d %s' % (iters[0], str(iters[1]))
    w, scale, stop = str(w), str(scale), str(stop)
    return ['python run_STORM.py' + (' %s'*8)%(cores[i % len(cores)], in_file, out_file[i],
                                                adaptive, iters, w, scale, stop)
                                  + ' %d %d'%(index[i], index[i + 1]) for i in range(chunks)]


if __name__ == '__main__':
    from sys import argv
    cores, in_file, out_file = argv[1:4]
    adaptive, iters = (int(argv[4]), bool(argv[5])), (int(argv[6]), float(argv[7]))
    w, scale, stop = [float(x) for x in argv[8:11]]
    index = int(argv[11]), int(argv[12])

    run(cores, in_file, out_file, adaptive, iters, w, scale, stop, index)