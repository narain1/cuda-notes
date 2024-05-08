from PIL import Image
import numpy as np
import ctypes

lib = ctypes.CDLL("./libblur.so")

lib.blur.argtypes = [
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.POINTER(ctypes.c_ubyte),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]


img = Image.open("sample.jpg")
img = np.array(img)
out_img = np.zeros_like(img)
h, w, c = img.shape
print(img.shape)

input_ptr = img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
output_ptr = out_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

lib.blur(input_ptr, output_ptr, w, h, c)

out = Image.fromarray(out_img)
out.save('blurred.png')