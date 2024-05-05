import ctypes
import numpy as np
from PIL import Image

# Load the shared library
lib = ctypes.CDLL('./librgb2gray.so')

# Define the function prototypes
rgb_to_grayscale = lib.rgbToGrayscale
rgb_to_grayscale.restype = None
rgb_to_grayscale.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=3, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_int
]

# Example usage with numpy arrays
height, width = 640, 480  # Example dimensions
img = Image.open("sample.jpg")
img = img.resize((height, width))

img = np.array(img)
gray_array = np.zeros((width, height), dtype=np.uint8)
# Call the function
rgb_to_grayscale(img, gray_array, width, height)

# Now `gray_array` contains the grayscale image
gray = Image.fromarray(gray_array)
gray.save("gray.png")