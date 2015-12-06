import numpy as np
import sys
import timeit

print 'version: {}'.format(np.__version__)
print 'maxint: {}'.format(sys.maxint)
print

setup = 'import numpy; x = numpy.random.random((5000,5000))'
count = 5

t = timeit.Timer('numpy.dot(x, x.T)', setup=setup)
print 'dot: {} sec'.format(t.timeit(count)/count)
