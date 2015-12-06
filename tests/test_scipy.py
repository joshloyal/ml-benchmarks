import timeit

setup = "import numpy as np;\
         import scipy.linalg as linalg;\
         x = np.random.random((1000,1000));\
         z = np.dot(x, x.T)"
count = 5

t = timeit.Timer('linalg.cholesky(z, lower=True)', setup=setup)
print 'cholesky: {} sec'.format(t.timeit(count)/count)

t = timeit.Timer('linalg.svd(z)', setup=setup)
print 'svd: {} sec'.format(t.timeit(count)/count)
