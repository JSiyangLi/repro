import numpy as np
#import matplotlib.pyplot as plt
#https://www.askpython.com/python-modules/numpy/bilinear-interpolation-python

def bi_interpolation_psfsize(psf_data, x, y):
    height = len(psf_data) # number of rows in the picture
    width = len(psf_data[0]) # number of columns in the picture
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + 1
    y2 = y1 + 1
    if x2 >= width:
        x2 = x1
    if y2 >= height:
        y2 = y1
    p11 = psf_data[y1 - 1][x1 - 1] # -1 since array indexing starts from 0
    p12 = psf_data[y2 - 1][x1 - 1]
    p21 = psf_data[y1 - 1][x2 - 1]
    p22 = psf_data[y2 - 1][x2 - 1]
    x_diff = x - x1
    y_diff = y - y1
    interpolated = (p11 * (1 - x_diff) * (1 - y_diff) +
                    p21 * x_diff * (1 - y_diff) +
                    p12 * (1 - x_diff) * y_diff +
                    p22 * x_diff * y_diff)
    return interpolated

# deciding which level of index correspond to x axis
# np.max([np.max(b32_psf[0].data[i]) for i in range(1024)]) # find the largest psf size
# first_index = np.argmax([np.max(b32_psf[0].data[i]) for i in range(1024)]
# second_index = np.argmax(b32_psf[0].data[first_index])
# b32_psf[0].data[first_index][second_index] # this should match the largest psf size above

def vectorized_bi_interpolation(psf_data, x, y):
    """
    Vectorized bilinear interpolation on the PSF data.

    Parameters:
    psf_data (numpy.ndarray): 2D array of PSF/exposure data
    x (numpy.ndarray): x-coordinates (pixel)
    y (numpy.ndarray): y-coordinates (pixel)

    Returns:
    numpy.ndarray: Interpolated values at (x,y) coordinates
    """
    # Get integer parts of coordinates
    x1 = np.floor(x).astype(int)
    y1 = np.floor(y).astype(int)
    x2 = x1 + 1
    y2 = y1 + 1

    # Handle edge cases
    height, width = psf_data.shape
    x2 = np.where(x2 >= width, x1, x2)
    y2 = np.where(y2 >= height, y1, y2)

    # Ensure we don't go below 0 in indices
    x1 = np.maximum(x1, 0)
    y1 = np.maximum(y1, 0)
    x2 = np.maximum(x2, 0)
    y2 = np.maximum(y2, 0)

    # Get the four neighboring values for all points
    # Using advanced indexing to get all values at once
    p11 = psf_data[y1, x1]
    p12 = psf_data[y2, x1]
    p21 = psf_data[y1, x2]
    p22 = psf_data[y2, x2]

    # Calculate differences
    x_diff = x - x1
    y_diff = y - y1

    # Perform bilinear interpolation for all points
    interpolated = (p11 * (1 - x_diff) * (1 - y_diff) +
                    p21 * x_diff * (1 - y_diff) +
                    p12 * (1 - x_diff) * y_diff +
                    p22 * x_diff * y_diff)

    return interpolated