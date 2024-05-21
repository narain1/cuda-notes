import ctypes
import numpy as np

lib = ctypes.CDLL("./mm.so")

lib.mm.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32),
        np.ctypeslib.ndpointer(dtype=np.float32),
        np.ctypeslib.ndpointer(dtype=np.float32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int
        ]

lib.mm.restype = None

aw, ah = 2048, 2048
bw, bh = 2048, 2048

a = np.random.rand(ah, aw).astype(np.float32)
b = np.random.rand(bh, bw).astype(np.float32)
c = np.zeros((ah, bw), dtype=np.float32)

lib.mm(a, b, c, aw, ah, bw, bh)

print("result matrix c:")
print(c)

print(a @ b)
