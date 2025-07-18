import re
import os
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from coordfun import dm_xyreg_to_file, check_circle_overlaps_file, center_inside, circles_overlap, should_remove_circle
import ciao_contrib.runtool
from ciao_contrib.runtool import *
from ciao_contrib.runtool import dmcoords

# Function to extract numbers from a line
def extract_numbers(line):
    # Use regular expression to find all numbers in the line
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
    # Convert the extracted strings to floats
    return [float(num) for num in numbers]

# List of file names in order of resolution (highest to lowest)
file_names = [
    "b08_coord.txt",
    "b08_odegap_coord.txt",
    "b16_coord_clear.txt",
    "b32_coord_clear.txt"
]

# Corresponding FITS files for WCS
fits_names = [
    "27_b08_psf.fits",
    "27_b08_odegap_psf.fits",
    "27_b16_psf.fits",
    "27_b32_psf.fits"
]

# Pixel scale in arcseconds per pixel
pixel_scale = 0.1318

# Bin sizes corresponding to each file (8, 8, 16, 32)
bin_sizes = [8, 8, 16, 32]

# Read all files and store circles with their resolution priority
circles = []
for priority, (file_name, fits_name) in enumerate(zip(file_names, fits_names)):
    if os.path.exists(file_name) and os.path.exists(fits_name):
        # Load the WCS from the corresponding FITS file
        with fits.open(fits_name) as hdul:
            wcs = WCS(hdul[0].header)

        with open(file_name, 'r') as file:
            for line in file:
                ra, dec, r_arcsec = extract_numbers(line)
                circles.append({"ra": ra, "dec": dec, "r_arcsec": r_arcsec, "priority": priority, "file": file_name})
    else:
        print(f"File {file_name} or {fits_name} does not exist.")

# Sort circles by priority (highest resolution first)
circles.sort(key=lambda x: x["priority"])

# Merge overlapping circles
unique_circles = []
for circle in circles:
    is_duplicate = False
    for unique_circle in unique_circles:
        # Check if the current circle is from a larger binning
        if circle["priority"] > unique_circle["priority"]:
            # Case 1: If the center of the current circle is inside the unique circle, remove it
            if center_inside(circle, unique_circle):
                is_duplicate = True
                break
            # Case 2: If the circles overlap and the diameter condition is met, remove it
            elif circles_overlap(circle, unique_circle) and should_remove_circle(circle, bin_sizes[circle["priority"]]):
                is_duplicate = True
                break
    if not is_duplicate:
        unique_circles.append(circle)

# Count how many stars from each file are included in the unique list
file_counts = {file_name: 0 for file_name in file_names}
for star in unique_circles:
    file_counts[star["file"]] += 1

# Print the summary
print("Summary of unique stars:")
for file_name, count in file_counts.items():
    print(f"  {file_name}: {count} stars included in the unique list")

# Optionally, save the unique stars to a new file
with open("unique_stars.txt", "w") as output_file:
    for star in unique_circles:
        output_file.write(f"circle({star['ra']}d,{star['dec']}d,{star['r_arcsec']}'')\n")

# save the xpos, ypos, theta and phi of x-ray detections
# Extract ra and dec from unique_circles
ra = [circle["ra"] for circle in unique_circles]
dec = [circle["dec"] for circle in unique_circles]
psfsiz = [circle["r_arcsec"] for circle in unique_circles]

# Define the output file name
xyregfile = "unique_xray.xyreg"
xyoffxfile = "unique_xray.xyoffx"
evtfile = "hrcf00027_repro_evt2.fits"

# Call the function
dm_xyreg_to_file(ra, dec, psfsiz, xyregfile, evtfile)
dm_xyoffx_to_file(ra, dec, evtfile, xyoffxfile)

# Check for overlaps
overlaps = check_circle_overlaps_file(writefile)

# Print the results
if overlaps:
    print("Overlapping circles found:")
    for i, j in overlaps:
        print(f"Circle {i} overlaps with Circle {j}")
else:
    print("No overlapping circles found.")