import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as u
from bilinear_interpolation import bi_interpolation_psfsize
from circfun import deltang, circcumulate
import os
from catfun import parse_circle_file, plot_cumulative_properties, plot_group_properties, remove_groups, get_min_theta

# Read the relevant files
# Read the xyoffx file
# Load the data from the file
xyoffx_df = np.loadtxt('J_A+A_375_863_sub.xyoffx')

# Transpose the array and unpack into separate variables
xpos, ypos, theta, phi = xyoffx_df.T

# Load group properties from file
data = np.loadtxt('group_properties.txt', skiprows=1, unpack=True)
group_id = data[0]
max_distance = data[1]
total_distance = data[2]
num_circles = data[3]
group_min_theta = data[4]
isrc = data[5]
nsrc = len(group_id)

# Sort groups by area and number of circles
max_dist_order = np.argsort(-max_distance)  # Descending order
total_dist_order = np.argsort(-total_distance)  # Descending order
number_order = np.argsort(-num_circles)  # Descending order

# Create cumulative area and cumulative number of groups
cumulative_maxdist = np.cumsum(max_distance[max_dist_order])
cumulative_totaldist = np.cumsum(total_distance[total_dist_order])
cumulative_number = np.cumsum(num_circles[number_order])

# Plot group properties
plot_cumulative_properties(max_dist_order, total_dist_order, number_order, cumulative_maxdist, cumulative_totaldist, cumulative_number, num_circles)
plot_group_properties(max_dist_order, total_dist_order, number_order, max_distance, total_distance, num_circles)

# Perform group removal

global_min_theta = get_min_theta(max_dist_order, total_dist_order, number_order, group_id, group_min_theta)
oki = theta < global_min_theta

# re-load all the coordinates
radec_df = parse_circle_file('J_A+A_375_863_sub.reg')
# Extract RA, DEC, and radii from the optical catalogues
ra = np.array([circle[0] for circle in radec_df])  # RA in degrees
dec = np.array([circle[1] for circle in radec_df])  # DEC in degrees
psfsiz = np.array([circle[2] for circle in radec_df])  # Radii in arcseconds
psfsiz_pix = psfsiz / 0.1318 # converting from arcsec to pixels
cc = [f"circle({ra[i]}d, {dec[i]}d, {psfsiz[i]}\'\')" for i in range(len(ra))]

print(f"theta length: {len(theta)}")
print(f"ra length: {len(ra)}")

ra = ra[oki]
dec = dec[oki]
psfsiz_pix = psfsiz_pix[oki]
psfsiz = psfsiz[oki]
xpos = xpos[oki]
ypos = ypos[oki]
theta = theta[oki]
phi = phi[oki]

# Write RA and Dec to a new file
with open('cbind_sub_cleared.reg', 'w') as f:
    for i in range(len(ra)):
        f.write(f"circle({ra[i]}d, {dec[i]}d, {psfsiz[i]}\'\')\n")

with open('cbind_sub_cleared.xyoffx', 'w') as f:
    for i in range(len(xpos)):
        f.write(f"{xpos[i]}\t{ypos[i]}\t{theta[i]}\t{phi[i]}\n")

cc_pix = [f"circle({xpos[i]}, {ypos[i]}, {psfsiz_pix[i]})" for i in range(len(xpos))]
# Write regions to a file
with open('cbind_sub_cleared.xyreg', 'w') as f:
    for reg in cc_pix:
        f.write(f"{reg}\n")
# Filter based on theta
