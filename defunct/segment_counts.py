from astropy.io import fits
import numpy as np

# Step 1: Load event data from a FITS file
evt_file = 'hrcf00027_repro_evt2.fits'
evt = fits.open(evt_file)[1].data  # Assuming the data is in the second HDU
evtx = evt['X']  # Extract X coordinates
evty = evt['Y']  # Extract Y coordinates

# Step 2: Load circle data from the region file
region_file = 'tmp/optreg_emap_grp0.reg'

# Initialize lists to store circle data
xc, yc, rc = [], [], []

# Read the region file and parse circle data
with open(region_file, 'r') as file:
    for line in file:
        if line.startswith('circle'):
            # Extract the content inside the parentheses
            content = line.split('(')[1].split(')')[0]
            # Split into xc, yc, rc
            xc_val, yc_val, rc_val = map(float, content.split(','))
            xc.append(xc_val)
            yc.append(yc_val)
            rc.append(rc_val)

# Convert lists to numpy arrays
xc = np.array(xc)
yc = np.array(yc)
rc = np.array(rc)

# Step 3: Calculate the bounding box
xmin = np.min(xc) - np.max(rc)  # Minimum X coordinate of the bounding box
xmax = np.max(xc) + np.max(rc)  # Maximum X coordinate of the bounding box
ymin = np.min(yc) - np.max(rc)  # Minimum Y coordinate of the bounding box
ymax = np.max(yc) + np.max(rc)  # Maximum Y coordinate of the bounding box

# Step 4: Filter events within the bounding box
mask = (evtx >= xmin) & (evtx <= xmax) & (evty >= ymin) & (evty <= ymax)
evtx_filtered = evtx[mask]
evty_filtered = evty[mask]

# Print results
print(f"Filtered events: {len(evtx_filtered)}")
print(f"Circle centers (xc): {xc}")
print(f"Circle centers (yc): {yc}")
print(f"Circle radii (rc): {rc}")