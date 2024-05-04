import numpy as np

a = np.random.rand(4096).astype(np.float32)
b = np.random.rand(4096).astype(np.float32)

print(a[:5])
print(b[:5])

c = a + b
print(c[:5])
a.tofile('vec1.bin')
b.tofile('vec2.bin')
