from circfun import filter_evt_for_bkgd, save_bkgd_to_txt
import os
from catfun import parse_circle_file_pixel, extract_theta
import numpy as np
from astropy.io import fits
from ciao_contrib.runtool import dmcoords

# 1. Get absolute paths to all files
base_dir = os.path.dirname(os.path.abspath(__file__))  # Script's directory
evtfile = os.path.join(base_dir, "hrcf00027_repro_evt2.fits")
region_file = os.path.join(base_dir, "all+phantom_sources.xyreg")
theta_file = os.path.join(base_dir, "all+phantom_sources.xyoffx")

# 2. Verify files exist before processing
if not all(os.path.exists(f) for f in [evtfile, region_file]):
    missing = [f for f in [evtfile, region_file] if not os.path.exists(f)]
    raise FileNotFoundError(f"Missing files: {missing}")

# 3. Load circle data
circ_df = parse_circle_file_pixel(region_file)
xpos = np.array([circle[0] for circle in circ_df])
ypos = np.array([circle[1] for circle in circ_df])
psfsiz_pix = np.array([circle[2] for circle in circ_df])
theta = extract_theta(theta_file)

# 4. Load event data with explicit verification
with fits.open(evtfile) as hdul:
    data = hdul[1].data
    head = hdul[1].header
    evtX = data['x']
    evtY = data['y']
    centreRa = head['ra_nom']
    centreDec = head['dec_nom']

# Convert to pixel coordinates using dmcoords
dmcoords(infile=evtfile, asolfile=None, opt="cel", ra=centreRa, dec=centreDec)
centre_x = float(dmcoords.x)
centre_y = float(dmcoords.y)

print(centre_x)
print(centre_y)
print("start")
# 5. Run analysis with full path handling
result = filter_evt_for_bkgd(
    evtX, evtY,
    xpos, ypos, psfsiz_pix,
    theta, centre_x, centre_y,
    verbose=True,
)

print(result['nroi'])
print(result['bkg_rate'])
print(np.sum(result['nroi']))

# 6. Save results
save_bkgd_to_txt(
    result,
    "cbind_bkgd.txt",
    save_radius=True,
    verbose=True
)