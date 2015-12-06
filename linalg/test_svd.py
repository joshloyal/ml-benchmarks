import numpy as np
import pandas as pd
from time import time
import seaborn as sns
import matplotlib.pyplot as plt
from memory_profiler import memory_usage
from scipy.sparse import csc_matrix

from concurrent.futures import ProcessPoolExecutor

def sparse_matrix(N1, N2, f, conversion=np.asarray, rseed=0):
    """create NxN matrix with an approximate fraction f of nonzero entries"""
    rng = np.random.RandomState(rseed)
    M = rng.rand(N1, N2)
    M[M > f] = 0
    return conversion(M)

def time_svd(svdfunc, N1, N2, f, rseed=0, bestof=3, args=None, matfunc=np.asarray, **kwargs):
    if args is None:
        args = ()

    N1_N2_f = np.broadcast(N1, N2, f)
    times = []
    memory = []
    for (N1, N2, f) in N1_N2_f:
        M = sparse_matrix(N1, N2, f, matfunc, rseed)
        t_best = np.inf
        mem_best = np.inf

        for i in range(bestof):
            t0 = time()
            if args:
                _args = [M]
                _args.extend(list(args))
            else:
                _args = (M,)
            mem_usage = max(memory_usage((svdfunc, _args, kwargs)))
            t1 = time()
            t_best = min(t_best, t1 - t0)
            mem_best = min(mem_best, mem_usage)

        times.append(t_best)
        memory.append(mem_best)

    return np.array(times).reshape(N1_N2_f.shape), np.array(memory).reshape(N1_N2_f.shape)


# we spawn child processes to make sure memory is cleaned after each call

def plot_propack(N1, N2, f, k):
    from pypropack import svdp
    print "computing execution times for propack..."
    with ProcessPoolExecutor(max_workers=1) as executor:
        t, m = executor.submit(time_svd, *(svdp, N1, N2, f), **dict(k=k, kmax=500,
                     matfunc=csc_matrix)).result()
    return pd.DataFrame({'N': N, 't': t, 'm': m, 'name': 'propack'})


def plot_arpack(N1, N2, f, k):
    from scipy.sparse.linalg import svds
    print "computing execution times for arpack..."
    with ProcessPoolExecutor(max_workers=1) as executor:
        t, m = executor.submit(time_svd, *(svds, N1, N2, f), **dict(k=k,
                     matfunc=csc_matrix)).result()
    return pd.DataFrame({'N': N, 't': t, 'm': m, 'name': 'arpack'})


def plot_svdlibc(N1, N2, f, k):
    from sparsesvd import sparsesvd
    print "computing execution times for svdlibc..."
    with ProcessPoolExecutor(max_workers=1) as executor:
        t, m = executor.submit(time_svd, *(sparsesvd, N1, N2, f), **dict(args=(5,),
                     matfunc=csc_matrix)).result()
    #t, m = time_svd(sparsesvd, N1, N2, f, args=(5,), matfunc=csc_matrix)
    return pd.DataFrame({'N': N, 't': t, 'm': m, 'name': 'svdlibc'})


def plot_lapack(N1, N2, f, k):
    from scipy.linalg import svd
    print "computing execution times for lapack..."
    with ProcessPoolExecutor(max_workers=1) as executor:
        t, m = executor.submit(time_svd, *(sparsesvd, N1, N2, f), **dict(full_matrices=True)).result()
    return pd.DataFrame({'N': N, 't': t, 'm':m, 'name': 'lapack'})


if __name__ == '__main__':
    N = 2 ** np.arange(3, 12)
    f = 0.6
    k = 5

    df = pd.DataFrame()

    try:
        df = df.append(plot_propack(N, N, f, k))
    except ImportError:
        print "propack cannot be loaded"

    try:
        df = df.append(plot_arpack(N, N, f, k))
    except ImportError:
        print "scipy arpack wrapper cannot be loaded"

    try:
        df = df.append(plot_svdlibc(N, N, f, k))
    except ImportError:
        print "svdlibc cannot be loaded"

    #try:
    #    df = df.append(plot_lapack(N, N, f, k))
    #except ImportError:
    #    print "scipy lapack wrapper cannot be loaded"


    sns.set_style('white')
    sns.factorplot(x='N', y='t', data=df, hue='name', legend_out=False)
    plt.show()

    sns.factorplot(x='N', y='m', data=df, hue='name', legend_out=False)
    plt.show()
