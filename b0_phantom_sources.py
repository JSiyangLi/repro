import numpy as np
from catfun import parse_circle_file
from merge_files import merge_txt_files
# Read both files and extract theta values
theta_values = []
# Load the data from the file
xyoffx_df = np.loadtxt('J_A+A_375_863_sub.xyoffx')

# Transpose the array and unpack into separate variables
xpos, ypos, theta, phi = xyoffx_df.T

# Read b1_xray.xyoffx
with open('b1_xray.xyoffx', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            theta_values.append(float(parts[2]))

# Read cbind_sub_cleared.xyoffx
with open('cbind_sub_cleared.xyoffx', 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3:
            theta_values.append(float(parts[2]))

# Find maximum theta
max_source_theta = max(theta_values)
print(f"Maximum theta: {max_source_theta}")

# Apply the same filtering logic as in your original code
oki = theta < max_source_theta

# Re-load all the coordinates (assuming you have theta, xpos, ypos, phi arrays already loaded)
radec_df = parse_circle_file('J_A+A_375_863_sub.reg')

# Extract RA, DEC, and radii from the optical catalogues
ra = np.array([circle[0] for circle in radec_df])  # RA in degrees
dec = np.array([circle[1] for circle in radec_df])  # DEC in degrees
psfsiz = np.array([circle[2] for circle in radec_df])  # Radii in arcseconds
psfsiz_pix = psfsiz / 0.1318  # converting from arcsec to pixels
cc = [f"circle({ra[i]}d, {dec[i]}d, {psfsiz[i]}\'\')" for i in range(len(ra))]

print(f"theta length: {len(theta)}")
print(f"ra length: {len(ra)}")

# Apply filtering
ra = ra[oki]
dec = dec[oki]
psfsiz_pix = psfsiz_pix[oki]
psfsiz = psfsiz[oki]
xpos = xpos[oki]
ypos = ypos[oki]
theta = theta[oki]
phi = phi[oki]

# Write RA and Dec to a new file
with open('cbind_bkgd_complement.reg', 'w') as f:
    for i in range(len(ra)):
        f.write(f"circle({ra[i]}d, {dec[i]}d, {psfsiz[i]}\'\')\n")

with open('cbind_bkgd_complement.xyoffx', 'w') as f:
    for i in range(len(xpos)):
        f.write(f"{xpos[i]}\t{ypos[i]}\t{theta[i]}\t{phi[i]}\n")

cc_pix = [f"circle({xpos[i]}, {ypos[i]}, {psfsiz_pix[i]})" for i in range(len(xpos))]
# Write regions to a file
with open('cbind_bkgd_complement.xyreg', 'w') as f:
    for reg in cc_pix:
        f.write(f"{reg}\n")

print(f"Filtered data points: {len(ra)}")

merge_txt_files("b1_xray.xyreg", "cbind_bkgd_complement.xyreg", "all+phantom_sources.xyreg")
merge_txt_files("b1_xray.xyoffx", 'cbind_bkgd_complement.xyoffx', "all+phantom_sources.xyoffx")