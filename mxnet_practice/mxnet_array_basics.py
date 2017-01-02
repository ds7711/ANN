import mxnet as mx
import numpy as np

# create array
a = mx.nd.array(np.arange(6).reshape((2, 3)))
print a.shape
print a.dtype
assert a.dtype is np.float32, "NDArray data type inconsistency." # assert raise an error when the first expression is wrong, display the string

b = mx.nd.array([0, 1, 2], dtype=np.int32)
assert b.dtype is np.int32

# ways to initialize array with known size but unknow values
# zeros, ones, full (uniform value), asnumpy (random values)
def print_val(mx_array):
    print mx_array.asnumpy()
c = mx.nd.zeros((2, 3))
d = mx.nd.ones((3, 4))
print c, d
e = mx.nd.full((5, 6), np.infty)
print_val(e)
f = mx.nd.empty((5, 6))
print_val(f)
print e + f
# to print a mx nd array, we first convert it to a numpy array and then print

# arithmetic operations are similar to numpy array, including maximum, default elementwise operation and dot product

# indexing and slicing
mmm = mx.nd.array(np.random.randn(4, 5))
print_val(mmm)
print np.array_equal(mmm[1].asnumpy(), mmm.asnumpy()[1])

mmm2 = mx.nd.slice_axis(mmm, axis=0, begin=1, end=2)
print_val(mmm2)

a = mx.nd.array(np.arange(6).reshape(3,2))
a[:].asnumpy()
d = mx.nd.slice_axis(a, axis=1, begin=0, end=1)
d.asnumpy()
a[1:2, 0:2].asnumpy()
# unlike numpy, mx nd array could only be sliced along the first dimension!!!

# broadcast
aaa = mx.nd.array(np.arange(3)[:, np.newaxis])
bbb = aaa.broadcast_to((3, 2))
ccc = bbb.reshape((1, 6))
print_val(ccc)
ddd = ccc.broadcast_to((2, 6))
print_val(ddd)
aaa = mx.nd.array(np.arange(24).reshape(3, 2, 4))
print_val(aaa)
bbb = mx.nd.array(np.arange(12).reshape(3, 1, 4))
print_val(bbb)
print_val(aaa + bbb)

# Copies: assignment DOESN'T Make a copy of data, copy method does that explicityly
# pay special attention to [] and copyto operator
aaa = mx.nd.array([1, 2, 3])
bbb = aaa
print aaa is bbb
ccc = mx.nd.array([0, 0, 0])
print_val(ccc)
ddd = ccc.copyto(aaa)
print_val(aaa)
print_val(ddd)
print ddd is ccc, aaa is ccc

a = mx.nd.zeros((3))
b = mx.nd.ones(a.shape)
c = b
print_val(c)
print c is b
c[:] = a
print c is a, c is b
print_val(a), print_val(c)
d = b
a.copyto(d)
e = mx.nd.array([[1, 2, 3], [4, 5, 6]])
a.copyto(e)

# advaned features: GPU can be used to do arithmetic operations
# ability to push computation backwards and thus enable automatic parallel computations