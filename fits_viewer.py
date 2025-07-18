import os
from astropy.io import fits
import numpy as np
from circfun import filter_evt_by_roigroup, evt_response, make_evt_design, save_roi_to_txt
from catfun import parse_circle_file_pixel, extract_theta
from astropy.wcs import WCS

# 1. Get absolute paths to all files
base_dir = os.path.dirname(os.path.abspath(__file__))  # Script's directory
emap_path = os.path.join(base_dir, "27_b32.emap")
secondary_emap = os.path.join(base_dir, "27_b08.emap")
evtfile = os.path.join(base_dir, "hrcf00027_repro_evt2.fits")

with fits.open(evtfile) as hdul:
    data = hdul[1].data
    head = hdul[1].header
    evtX = data['x']
    evtY = data['y']
    header = hdul[1].header
    centreRa = head['ra_nom']
    centreDec = head['dec_nom']

print(centreRa)
print(centreDec)
print(np.min(evtX))
print(np.max(evtX))
print(np.min(evtY))
print(np.max(evtY))
print(evtfile)