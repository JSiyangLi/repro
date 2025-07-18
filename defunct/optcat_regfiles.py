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
from catfun import plot_cumulative_properties, remove_groups, plot_group_properties

# Read the relevant files
# Read the xyoffx file
xpos, ypos, theta, phi = np.loadtxt('J_A+A_375_863_sub.xyoffx', unpack=True)
data0 = np.loadtxt('circle_psf.txt', skiprows=1, unpack=True)
psfsiz_pix = data0[0]
psfsiz = data0[1]

# Read grpreg as a list of strings
with open('J_A+A_375_863_sub_group.reg', 'r') as f:
    grpreg = f.read().splitlines()  # Read lines as strings

# Read grpidx as a list of strings
with open('J_A+A_375_863_sub_group.idx', 'r') as f:
    grpidx = f.read().splitlines()  # Read lines as strings

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
oki = remove_groups(
    max_dist_order, total_dist_order, number_order, group_id, group_min_theta, theta
)

# Define column names for the table3.dat file
column_names = [
    'num', 'rah', 'ram', 'ras', 'decd', 'decm', 'decs',
    'vmag', 'b_v', 'v_i', 'f1', 'f2', 'f3', 'f4', 'f5'
]

# Read the optical catalog
optcatfil = '/Users/jl1023/NGC2516/J_A+A_375_863/table3.dat'
data = np.genfromtxt(optcatfil, dtype=None, names=column_names)

# Extract columns
num = data['num']
rah = data['rah']
ram = data['ram']
ras = data['ras']
decd = data['decd']
decm = data['decm']
decs = data['decs']
vmag = data['vmag']
b_v = data['b_v']
v_i = data['v_i']
f1 = data['f1']
f2 = data['f2']
f3 = data['f3']
f4 = data['f4']
f5 = data['f5']

# Convert RA and Dec to degrees
ra = 15.0 * (rah + (ram + ras / 60.0) / 60.0)
dec = np.abs(decd) + (decm + decs / 60.0) / 60.0
dec[decd < 0] *= -1

# Calculate angular separation from a reference point
ra_nom = 1.1953556646780e+02
dec_nom = -6.0763788181428e+01
angsep = deltang(ra, dec, ra_nom, dec_nom)

# Filter sources based on angular separation
ok = angsep < 16
num = num[ok]
ra = ra[ok]
dec = dec[ok]
vmag = vmag[ok]
b_v = b_v[ok]
v_i = v_i[ok]
f1 = f1[ok]
f2 = f2[ok]
f3 = f3[ok]
f4 = f4[ok]
f5 = f5[ok]
num = num[oki]
ra = ra[oki]
dec = dec[oki]
vmag = vmag[oki]
b_v = b_v[oki]
v_i = v_i[oki]
f1 = f1[oki]
f2 = f2[oki]
f3 = f3[oki]
f4 = f4[oki]
f5 = f5[oki]

# Write filtered data to a new file
with open('J_A+A_375_863_sub_cleared.dat', 'w') as f:
    for i in range(len(num)):
        f.write(f"{num[i]} {rah[i]} {ram[i]} {ras[i]} {decd[i]} {decm[i]} {decs[i]} "
                f"{vmag[i]} {b_v[i]} {v_i[i]} {f1[i]} {f2[i]} {f3[i]} {f4[i]} {f5[i]}\n")

# Write RA and Dec to a new file
with open('J_A+A_375_863_sub_cleared.radec', 'w') as f:
    for i in range(len(ra)):
        f.write(f"{ra[i]} {dec[i]}\n")

# Filter based on theta
xpos = xpos[oki]
ypos = ypos[oki]
theta = theta[oki]
phi = phi[oki]
psfsiz_pix = psfsiz_pix[oki]
psfsiz = psfsiz[oki]

# Create region strings
cc = [f"circle({ra[i]}d, {dec[i]}d, {psfsiz[i]}\'\')" for i in range(len(ra))]
cc_pix = [f"circle({xpos[i]}, {ypos[i]}, {psfsiz_pix[i]})" for i in range(len(xpos))]

# Write regions to a file
with open('J_A+A_375_863_sub_cleared.reg', 'w') as f:
    for reg in cc:
        f.write(f"{reg}\n")

# Group overlapping regions and compute group properties
ig, group_max_distance, group_total_distance, group_num_circles, group_min_theta = circcumulate(xpos, ypos, psfsiz_pix, theta, thresh=0.05)
print('end circcumulate')
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
    grpreg[i] = "+".join([cc_pix[j] for j in oo])

    # Write grouped regions to files
    with open(f"tmp/optreg_emap_grp{isrc[i]}.reg", "w") as f:
        f.write(cc_pix[oo[0]])
        for j in range(1, len(oo)):
            f.write(f"\n+{cc_pix[oo[j]]}")

    # Write range files
    xmin = np.min(xpos[oo])
    xmax = np.max(xpos[oo])
    ymin = np.min(ypos[oo])
    ymax = np.max(ypos[oo])
    rmax = np.max(psfsiz[oo])
    rngmin = min(xmin, ymin)
    rngmax = max(xmax, ymax)

    with open(f"tmp/optreg_grp{isrc[i]}.rng", "w") as f:
        f.write(f'dmcopy "27.evt2[bin x={rngmin - rmax}:{rngmax + rmax}:#1024, '
                f'y={rngmin - rmax}:{rngmax + rmax}:#1024]" 27_rng.img clobber=yes verbose=2\n')

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
print("ds9 27.evt2 -regionfile J_A+A_375_863_sub.reg &")
print("grouped optical catalog regions are in the file J_A+A_375_863_sub_group_cleared.reg")
print("grouped, separated, regions are in files tmp/optreg_emap_grp*.reg")
print("grouped optical catalog source indices are in the file J_A+A_375_863_sub_group_cleared.idx")
print("")
print("run the command --")
print("optreg_emap08.cmd >! optreg_emap08.lst")
print("optreg_emap16.cmd >! optreg_emap16.lst")
print("optreg_emap32.cmd >! optreg_emap32.lst")
print("idl < optreg_emap.com")
print("to get the exposure values in optreg_emap.lst, and")
print("")
print("and to extract the counts, run the command --")
print("./alt_optcat_reg.cmd")
print("")
print("as an alternative to the interminably slow --")
print('dmextract infile="27.evt2[bin sky=@J_A+A_375_863_sub_group_cleared.reg]" outfile=J_A+A_375_863_sub_group_cleared.counts clobber=yes verbose=2')
print('dmlist "J_A+A_375_863_sub_group_cleared.counts[cols sky,r,counts,area]" data,raw >! J_A+A_375_863_sub_group_cleared_counts')