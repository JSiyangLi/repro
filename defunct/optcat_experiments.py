import numpy as np
import matplotlib.pyplot as plt
import os

# Function to calculate angular separation
def deltang(ra1, dec1, ra2, dec2):
    coord1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame='icrs')
    coord2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame='icrs')
    return coord1.separation(coord2).arcminute


# Read the relevant files
data0 = np.loadtxt('circle_info.txt', skiprows=1, unpack=True)
xpos = data0[0]
ypos = data0[1]
psfsiz_pix = data0[2]
theta = data0[3]

# Read grpreg as a list of strings
with open('J_A+A_375_863_sub_group.reg', 'r') as f:
    grpreg = f.read().splitlines()  # Read lines as strings

# Read grpidx as a list of strings
with open('J_A+A_375_863_sub_group.idx', 'r') as f:
    grpidx = f.read().splitlines()  # Read lines as strings

# Load group properties from file
data = np.loadtxt('group_properties.txt', skiprows=1, unpack=True)
group_id = data[0]
total_area = data[1]
num_circles = data[2]
group_min_theta = data[3]
isrc = data[4]
nsrc = len(group_id)

# Sort groups by area and number of circles
area_order = np.argsort(-total_area)  # Descending order
number_order = np.argsort(-num_circles)  # Descending order

# Create cumulative area and cumulative number of groups
cumulative_area = np.cumsum(total_area[area_order])
cumulative_number = np.cumsum(num_circles[number_order])



# Define column names for the table3.dat file
column_names = [
    'num', 'rah', 'ram', 'ras', 'decd', 'decm', 'decs',
    'vmag', 'b_v', 'v_i', 'f1', 'f2', 'f3', 'f4', 'f5'
]

# Read the optical catalog
optcatfil = '/Users/jl1023/NGC2516/J_A+A_375_863/table3.dat'
data1 = np.genfromtxt(optcatfil, dtype=None, names=column_names)

# Extract columns
num = data1['num']
rah = data1['rah']
ram = data1['ram']
ras = data1['ras']
decd = data1['decd']
decm = data1['decm']
decs = data1['decs']
vmag = data1['vmag']
b_v = data1['b_v']
v_i = data1['v_i']
f1 = data1['f1']
f2 = data1['f2']
f3 = data1['f3']
f4 = data1['f4']
f5 = data1['f5']

# Convert RA and Dec to degrees
ra = 15.0 * (rah + (ram + ras / 60.0) / 60.0)
dec = np.abs(decd) + (decm + decs / 60.0) / 60.0
dec[decd < 0] *= -1

# Calculate angular separation from a reference point
ra_nom = 1.1953556646780e+02
dec_nom = -6.0763788181428e+01
angsep = deltang(ra, dec, ra_nom, dec_nom)

# Function to read the group indices from the .idx file
def read_group_indices(filename):
    """
    Read the group indices from the .idx file.
    Each line corresponds to a group, and the line contains the indices of circles in that group.
    Returns a list of lists, where each inner list contains the circle indices for a group.
    """
    group_indices = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into individual circle indices and convert them to integers
            circle_indices = list(map(int, line.strip().split()))
            group_indices.append(circle_indices)
    return group_indices

# Interactive group removal
def remove_groups(area_order, number_order, group_id, total_area, num_circles, group_min_theta, theta, grpreg, grpidx, isrc):
    # Read the group indices from the .idx file
    group_indices = read_group_indices('J_A+A_375_863_sub_group.idx')

    while True:
        user_input = input("Enter removal command (e.g., 'rm_area_order=N', 'rm_number_order=N', or 'done'): ").strip()

        if user_input.lower() == 'done':
            break

        if user_input.startswith('rm_area_order='):
            try:
                rm_order = int(user_input.split('=')[1])
                # Find the group IDs corresponding to the area order (up to rm_order)
                rm_ids = [group_id[area_order[i]] for i in range(rm_order)]
                # Find the smallest theta among the specified groups
                min_theta = min(group_min_theta[np.where(group_id == rm_id)[0][0]] for rm_id in rm_ids)

                print(f"Removing circles with theta > {min_theta}: circles")
            except (IndexError, ValueError):
                print("Invalid input. Please use 'rm_area_order=N' where N is a valid order.")
                continue

        elif user_input.startswith('rm_number_order='):
            try:
                rm_order = int(user_input.split('=')[1])
                # Find the group IDs corresponding to the number order (up to rm_order)
                rm_ids = [group_id[number_order[i]] for i in range(rm_order)]
                # Find the smallest theta among the specified groups
                min_theta = min(group_min_theta[np.where(group_id == rm_id)[0][0]] for rm_id in rm_ids)

                print(f"Removing circles with theta > {min_theta}: circles")
            except (IndexError, ValueError):
                print("Invalid input. Please use 'rm_number_order=N' where N is a valid order.")
                continue

        else:
            print("Invalid command. Please try again.")
            continue

        # Remove the specified circles
        # Filter based on theta
        ok = theta < min_theta
        xpos = xpos[ok]
        ypos = ypos[ok]
        theta = theta[ok]
        phi = phi[ok]

        # Load the PSF FITS file
        fits_name = "27_b32_psf.fits"
        f = fits.open(fits_name)

        # Extract WCS information from the FITS header
        w = WCS(f[0].header)

        # Convert xpos and ypos (physical coordinates) to pixel coordinates
        sc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
        xpos_pix, ypos_pix = skycoord_to_pixel(sc, w)

        # Load the PSF size using bilinear interpolation with pixel coordinates
        psfsiz = np.array([bi_interpolation_psfsize(f[0].data, xpos_pix[j], ypos_pix[j]) for j in range(len(xpos_pix))])
        psfsiz_pix = psfsiz / 0.1318  # converting from arcsec to pixels

        # Create region strings
        cc = [f"circle({ra[i]}d, {dec[i]}d, {psfsiz[i]}\'\')" for i in range(len(xpos))]

        # Write regions to a file
        with open('J_A+A_375_863_sub_clearedexp.reg', 'w') as f:
            for reg in cc:
                f.write(f"{reg}\n")

        grpreg = [grpreg[i] for i in range(len(grpreg)) if mask[i]]  # Filter grpreg
        grpidx = [grpidx[i] for i in range(len(grpidx)) if mask[i]]  # Filter grpidx

        # Parse grpidx to extract all remaining circle indices
        remaining_circle_indices = set()
        for group in grpidx:
            # Split the string into individual integers and add them to the set
            remaining_circle_indices.update(map(int, group.split()))

        # Filter isrc based on the remaining circles
        isrc = isrc[np.isin(isrc, list(remaining_circle_indices))]  # Use NumPy for efficient filtering
        if len(isrc) == 0:
            print("Warning: No sources left after filtering.")

    return group_id, total_area, num_circles, group_min_theta, grpreg, grpidx, isrc

# Perform group removal
group_id, total_area, num_circles, group_min_theta, grpreg, grpidx, isrc = remove_groups(
    area_order, number_order, group_id, total_area, num_circles, group_min_theta, theta, grpreg, grpidx, isrc
)

# Write remaining groups to a new file
with open('J_A+A_375_863_sub_group_cleared.reg', 'w') as f:
    for reg in grpreg:
        f.write(f"{reg}\n")

# Write remaining group indices to a new file
with open('J_A+A_375_863_sub_group_cleared.idx', 'w') as f:
    for idx in grpidx:
        f.write(f"{idx}\n")

# Generate commands for exposure calculations
with open('optreg_emap08.cmd', 'w') as f08, open('optreg_emap16.cmd', 'w') as f16, open('optreg_emap32.cmd', 'w') as f32:
    for i in range(min(nsrc, len(isrc))):
        ir = int(isrc[i])
        print(f"Processing group {i + 1}: ir = {ir}")
        cmd08 = f'dmstat "27_b08.emap[sky=region(tmp/optreg_emap_grp{ir}.reg)][opt null=0]" centroid=no | grep mean | awk \'{{print "{ir}", "{xpos[ir]}", "{ypos[ir]}", $2}}\''
        cmd16 = f'dmstat "27_b16.emap[sky=region(tmp/optreg_emap_grp{ir}.reg)][opt null=0]" centroid=no | grep mean | awk \'{{print "{ir}", "{xpos[ir]}", "{ypos[ir]}", $2}}\''
        cmd32 = f'dmstat "27_b32.emap[sky=region(tmp/optreg_emap_grp{ir}.reg)][opt null=0]" centroid=no | grep mean | awk \'{{print "{ir}", "{xpos[ir]}", "{ypos[ir]}", $2}}\''
        f08.write(f"{cmd08}\n")
        f16.write(f"{cmd16}\n")
        f32.write(f"{cmd32}\n")

# Make commands executable
os.chmod('optreg_emap08.cmd', 0o755)
os.chmod('optreg_emap16.cmd', 0o755)
os.chmod('optreg_emap32.cmd', 0o755)

# Generate alternate extraction commands
with open('alt_optcat_reg.cmd', 'w') as f:
    f.write("touch alt_optcat_reg_counts ; rm alt_optcat_reg_counts ; touch alt_optcat_reg_counts\n")
    for i in range(min(nsrc, len(isrc))):
        ir = int(isrc[i])
        nreg = len(open(f"tmp/optreg_emap_grp{ir}.reg").readlines())
        if nreg > 5:
            extrctfil = "27_rng.img"
            stampcmd = f"cat tmp/optreg_grp{ir}.rng\n"
        else:
            extrctfil = "27.evt2"
            stampcmd = ""
        f.write(stampcmd)
        f.write(f'dmextract infile="{extrctfil}[bin sky=region(tmp/optreg_emap_grp{ir}.reg)]" '
                f'outfile=tmp/alt_optreg_{ir}.counts clobber=yes verbose=2\n')
        f.write(f'dmlist "tmp/alt_optreg_{ir}.counts[cols sky,r,counts,area]" data,raw | grep -v ^# >> alt_optcat_reg_counts\n')

# Make alternate command executable
os.chmod('alt_optcat_reg.cmd', 0o755)
print("ungrouped optical catalog regions are in the file J_A+A_375_863_sub.reg")