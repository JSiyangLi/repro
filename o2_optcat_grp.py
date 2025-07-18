import numpy as np
from circfun import deltang, circcumulate
from catfun import parse_circle_file

# Load the data from the file
xyoffx_df = np.loadtxt('J_A+A_375_863_sub.xyoffx')

# Transpose the array and unpack into separate variables
xpos, ypos, theta, phi = xyoffx_df.T

radec_df = parse_circle_file('J_A+A_375_863_sub.reg')
# Extract RA, DEC, and radii from the optical catalogues
ra = np.array([circle[0] for circle in radec_df])  # RA in degrees
dec = np.array([circle[1] for circle in radec_df])  # DEC in degrees
psfsiz = np.array([circle[2] for circle in radec_df])  # Radii in arcseconds
psfsiz_pix = psfsiz / 0.1318 # converting from arcsec to pixels
cc = [f"circle({ra[i]}d, {dec[i]}d, {psfsiz[i]}\'\')" for i in range(len(ra))]

#####################
# Group overlapping regions and compute group properties
ig, group_max_distance, group_total_distance, group_num_circles, group_min_theta = circcumulate(xpos, ypos, psfsiz_pix, theta, thresh=0.05)
iug = np.unique(ig)
nsrc = len(iug)
isrc = np.zeros(nsrc, dtype=int) - 1
ksrc = np.zeros(nsrc, dtype=int)
grpreg = [""] * nsrc
grpidx = [""] * nsrc

for i in range(nsrc):
    oo = np.where(ig == iug[i])[0]
    isrc[i] = oo[0]
    ksrc[i] = len(oo)
    grpidx[i] = " ".join(map(str, oo))
    grpreg[i] = "+".join([cc[j] for j in oo])

# Save group properties to a file
with open('group_properties.txt', 'w') as f:
    f.write("Group_ID\tMax_Distance\tTotal_Distance\tNum_Circles\tMin_Theta\tisrc\n")
    for i in range(nsrc):
        f.write(f"{iug[i]}\t{group_max_distance[i]}\t{group_total_distance[i]}\t{group_num_circles[i]}\t{group_min_theta[i]}\t{isrc[i]}\n")

# Write groups to a new file
with open('J_A+A_375_863_sub_group.reg', 'w') as f:
    for reg in grpreg:
        f.write(f"{reg}\n")

# Write remaining group indices to a new file
with open('J_A+A_375_863_sub_group.idx', 'w') as f:
    for idx in grpidx:
        f.write(f"{idx}\n")

with open('circle_psf.txt', 'w') as f:
    f.write("psfsiz_pix\tpsfsiz\n")
    for i in range(len(theta)):
        f.write(f"\t{psfsiz_pix[i]}\t{psfsiz[i]}\n")

print("Grouping complete. Intermediate files saved.")