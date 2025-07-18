from circfun import n_overlap, overlap_correct, compute_area_only
import os
from astropy.io import fits
import numpy as np
from catfun import parse_circle_file_pixel, extract_theta
from astropy.wcs import WCS

# Should return π*r²
test_area = compute_area_only(
    i=0,
    xc=np.array([0]),
    yc=np.array([0]),
    rc=np.array([1]),
    overlap_data=None
)[1][0]
print(f"Circle area: {test_area:.6f} (expected: {np.pi:.6f})")

# Two touching circles
d = 1.0
r1, r2 = 1.0, 1.0
overlap = circolap(d, r1, r2)['overlap_area']
expected = (2*np.pi/3 - np.sqrt(3)/2) * r1**2  # Exact solution
print(f"Overlap area: {overlap:.6f} (expected: {expected:.6f})")