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
region_file = os.path.join(base_dir, "unique_xray.xyreg")
theta_file = os.path.join(base_dir, "unique_xray.xyoffx")

# 2. Verify files exist before processing
if not all(os.path.exists(f) for f in [emap_path, evtfile, region_file]):
    missing = [f for f in [emap_path, evtfile, region_file] if not os.path.exists(f)]
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
    evtX = data['x']
    evtY = data['y']

print("start")
# 5. Run analysis with full path handling
result = filter_evt_by_roigroup(
    evtX, evtY,
    xpos, ypos, psfsiz_pix,
    emap_path=emap_path,  # Use absolute path
    computePropArea=True,
    verbose=True,
    sparse=False,
    parallel=False,
    secondary_emap=secondary_emap
)

result = make_evt_design(result)
result = evt_response(result)

# 6. Save results
save_roi_to_txt(
    result,
    "xrayYroi.txt",
    "xrayAroi.txt",
    "xrayRmat.txt",
    "xrayEroi.txt",
    design_filename = "xrayDesign.txt",
    sparse=False
)
