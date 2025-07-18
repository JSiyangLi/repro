import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
import astropy.units as u
from bilinear_interpolation import bi_interpolation_psfsize
from coordfun import dm_xyoffx_to_file
from circfun import deltang
import os

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
angsep_mask = angsep < 16
filtered_indices = np.where(angsep_mask)[0]

# Apply the filter to all arrays
num = num[angsep_mask]
ra = ra[angsep_mask]
dec = dec[angsep_mask]
vmag = vmag[angsep_mask]
b_v = b_v[angsep_mask]
v_i = v_i[angsep_mask]
f1 = f1[angsep_mask]
f2 = f2[angsep_mask]
f3 = f3[angsep_mask]
f4 = f4[angsep_mask]
f5 = f5[angsep_mask]

# Write filtered data to a new file
with open('J_A+A_375_863_sub.dat', 'w') as f:
    for i in range(len(num)):
        f.write(f"{num[i]} {rah[filtered_indices[i]]} {ram[filtered_indices[i]]} {ras[filtered_indices[i]]} "
                f"{decd[filtered_indices[i]]} {decm[filtered_indices[i]]} {decs[filtered_indices[i]]} "
                f"{vmag[i]} {b_v[i]} {v_i[i]} {f1[i]} {f2[i]} {f3[i]} {f4[i]} {f5[i]}\n")

# Write RA and Dec to a new file
with open('J_A+A_375_863_sub.radec', 'w') as f:
    for i in range(len(ra)):
        f.write(f"{ra[i]} {dec[i]}\n")

# Define event file
evtfile = "hrcf00027_repro_evt2.fits"

# Calculate xpos, ypos, theta, phi using dmcoords
print('dmcoords radec to xy start')
dm_xyoffx_to_file(ra, dec, evtfile, 'J_A+A_375_863.xyoffx')

# Read the xyoffx file and apply theta filter
xy_data = np.loadtxt('J_A+A_375_863.xyoffx', unpack=True)
theta_mask = xy_data[2] < 16
final_indices = filtered_indices[theta_mask]  # Track original indices through both filters

# Apply the theta filter to all arrays
xpos = xy_data[0][theta_mask]
ypos = xy_data[1][theta_mask]
theta = xy_data[2][theta_mask]
phi = xy_data[3][theta_mask]
ra = ra[theta_mask]
dec = dec[theta_mask]
num = num[theta_mask]

# Load the PSF FITS file
fits_name = "27_b32_psf.fits"
with fits.open(fits_name) as f:
    # Extract WCS information from the FITS header
    w = WCS(f[0].header)

    # Convert RA/Dec to pixel coordinates
    sc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    xpos_pix, ypos_pix = skycoord_to_pixel(sc, w)

    # Load the PSF size using bilinear interpolation with pixel coordinates
    psfsiz = np.array([bi_interpolation_psfsize(f[0].data, xpos_pix[j], ypos_pix[j])
                      for j in range(len(xpos_pix))])
    psfsiz_pix = psfsiz / 0.1318  # converting from arcsec to pixels

# Create region strings only for sources that passed both filters
cc = [f"circle({ra[i]}d, {dec[i]}d, {psfsiz[i]}\'\')" for i in range(len(ra))]
cc_pix = [f"{xpos[i]}\t{ypos[i]}\t{theta[i]}\t{phi[i]}" for i in range(len(xpos))]

# Write regions to a file
with open('J_A+A_375_863_sub.reg', 'w') as f:
    for reg in cc:
        f.write(f"{reg}\n")

with open('J_A+A_375_863_sub.xyoffx', 'w') as f:
    for reg in cc_pix:
        f.write(f"{reg}\n")

print(f"Successfully created files with {len(ra)} matching entries")