import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from catfun import parse_circle_file, iterate_prop_values  # Import the required functions
from bilinear_interpolation import bi_interpolation_psfsize  # Import the interpolation function

# Fix random seed for reproducibility
np.random.seed(42)

# Step 1: Read optical catalogues from J_A+A_375_863_sub.reg
optical_catalogues = parse_circle_file('J_A+A_375_863_sub.reg')

# Extract RA, DEC, and radii from the optical catalogues
ra = np.array([circle[0] for circle in optical_catalogues])  # RA in degrees
dec = np.array([circle[1] for circle in optical_catalogues])  # DEC in degrees
radii = np.array([circle[2] for circle in optical_catalogues])  # Radii in arcseconds

# Step 2: Convert all optical source locations from RA and DEC coordinates to polar coordinates
# Define the nominal center
ra_nom = 1.1953556646780e+02  # in degrees
dec_nom = -6.0763788181428e+01  # in degrees

# Convert RA and DEC to pixel coordinates first
fits_name = "27_b32_psf.fits"
f = fits.open(fits_name)
w = WCS(f[0].header)

# Create SkyCoord object for the optical sources
sc = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
xpos_pix, ypos_pix = w.world_to_pixel(sc)

# Convert pixel coordinates to polar coordinates relative to the nominal center
sc_nom = SkyCoord(ra=ra_nom * u.degree, dec=dec_nom * u.degree)
xpos_nom, ypos_nom = w.world_to_pixel(sc_nom)

# Ensure xpos_nom and ypos_nom are arrays with shape (1,)
xpos_nom = np.array([xpos_nom])  # Shape: (1,)
ypos_nom = np.array([ypos_nom])  # Shape: (1,)

# Calculate the relative positions
xpos_rel = xpos_pix - xpos_nom[:, np.newaxis]  # Shape: (1, num_sources)
ypos_rel = ypos_pix - ypos_nom[:, np.newaxis]  # Shape: (1, num_sources)

# Convert to polar coordinates
angles = np.arctan2(ypos_rel, xpos_rel)  # Angles in radians, shape: (1, num_sources)
angles = np.mod(angles, 2 * np.pi)  # Ensure angles are between 0 and 2*pi

# Calculate the modulus (radial distance)
modulus = np.sqrt(xpos_rel**2 + ypos_rel**2)  # Euclidean distance in pixel coordinates, shape: (1, num_sources)

# Step 3: Randomly draw null_siz points from a uniform distribution from 0 to 2*pi
null_siz = int(1e3)
random_angles = np.random.uniform(0, 2 * np.pi, null_siz)  # Shape: (null_siz,)

# Step 4: Read X-ray detections from unique_stars.txt
xray_detections = parse_circle_file('unique_stars.txt')

# Step 5: Define constants
distance_threshold = 0.1318 * 8 * np.sqrt(2)  # Distance threshold in arcseconds
prop_values = np.arange(0.01, 1, 0.01)  # Example property values

# Step 6: Vectorize the rotation and coordinate conversion
# Create arrays to hold all rotated coordinates
rotated_x = modulus.T * np.cos(angles.T + random_angles[np.newaxis, :])  # Shape: (num_sources, null_siz)
rotated_y = modulus.T * np.sin(angles.T + random_angles[np.newaxis, :])  # Shape: (num_sources, null_siz)

# Convert back to pixel coordinates relative to the nominal center
x_rot_pix = rotated_x + xpos_nom[:, np.newaxis]  # Shape: (num_sources, null_siz)
y_rot_pix = rotated_y + ypos_nom[:, np.newaxis]  # Shape: (num_sources, null_siz)

# Step 7: Initialize arrays to store results for each prop_value
expected_angsep_samples = np.zeros((null_siz, len(prop_values)))  # Shape: (null_siz, num_prop_values)
expected_adjangsep_samples = np.zeros((null_siz, len(prop_values)))  # Shape: (null_siz, num_prop_values)
num_matches_samples = np.zeros((null_siz, len(prop_values)))      # Shape: (null_siz, num_prop_values)
var_angsep_samples = np.zeros((null_siz, len(prop_values)))       # Shape: (null_siz, num_prop_values)

# Step 8: Loop over each rotation
print("begin rotation loop")
for r in range(null_siz):
    # Convert pixel coordinates back to RA and DEC for this rotation
    rotated_coords = w.pixel_to_world(x_rot_pix[:, r], y_rot_pix[:, r])

    # Calculate radii for the rotated sources using bi_interpolation_psfsize
    radii_rot = np.array([bi_interpolation_psfsize(f[0].data, x_rot_pix[j, r], y_rot_pix[j, r]) for j in range(len(x_rot_pix))])

    # Combine RA, DEC, and interpolated radii
    rotated_ra_dec = np.column_stack((rotated_coords.ra.deg, rotated_coords.dec.deg, radii_rot))

    # Call iterate_prop_values for all prop_values
    result = iterate_prop_values(xray_detections, rotated_ra_dec, distance_threshold, prop_values)

    # Store results for this rotation
    expected_angsep_samples[r, :] = result['expected_angsep']
    expected_adjangsep_samples[r, :] = result["expected_adjangsep"]
    num_matches_samples[r, :] = result['num_matches']
    var_angsep_samples[r, :] = result['var_angsep']

print('end rotation loop')
# Step 9: Calculate sample means and variances across all rotations for each prop_value, excluding NaN values
mean_expected_angsep = np.nanmean(expected_angsep_samples, axis=0)  # Shape: (num_prop_values,)
mean_num_matches = np.nanmean(num_matches_samples, axis=0)          # Shape: (num_prop_values,)
mean_var_angsep = np.nanmean(var_angsep_samples, axis=0)           # Shape: (num_prop_values,)
mean_expected_adjangsep = np.nanmean(expected_adjangsep_samples, axis=0)
var_expected_angsep = np.nanvar(expected_angsep_samples, axis=0)
var_num_matches = np.nanvar(num_matches_samples, axis=0)
var_expected_adjangsep = np.nanvar(expected_adjangsep_samples, axis=0)

# Count the number of NaNs in mean_expected_angsep and mean_num_matches
num_nans_expected_angsep = np.sum(np.isnan(expected_angsep_samples), axis=0)  # Shape: (num_prop_values,)
num_nans_num_matches = np.sum(np.isnan(num_matches_samples), axis=0)          # Shape: (num_prop_values,)

# Step 10: Compute results for the original optical catalogues
print('start expectation plotting')
optresults = iterate_prop_values(xray_detections, optical_catalogues, distance_threshold, prop_values=prop_values)

# Ensure optresults['expected_angsep'] and optresults['num_matches'] are 1D arrays
optresults_expected_angsep = np.squeeze(optresults['expected_angsep'])
optresults_expected_adjangsep = np.squeeze(optresults['expected_adjangsep'])
optresults_num_matches = np.squeeze(optresults['num_matches'])

# Step 11: Find the point where mean_expected_angsep is closest to optresults_expected_angsep
diff_expected_angsep = np.abs(mean_expected_angsep - optresults_expected_angsep)
closest_expected_angsep_idx = np.nanargmin(diff_expected_angsep)
closest_expected_angsep_prop = prop_values[closest_expected_angsep_idx]
closest_expected_angsep_value = mean_expected_angsep[closest_expected_angsep_idx]

diff_expected_adjangsep = np.abs(mean_expected_adjangsep - optresults_expected_adjangsep)
closest_expected_adjangsep_idx = np.nanargmin(diff_expected_adjangsep)
closest_expected_adjangsep_prop = prop_values[closest_expected_adjangsep_idx]
closest_expected_adjangsep_value = mean_expected_adjangsep[closest_expected_adjangsep_idx]

# Step 12: Find intersection points
def find_intersection(x, y1, y2):
    """
    Find the intersection points between two curves y1 and y2.
    """
    # Ensure inputs are valid and exclude NaN values
    valid = ~np.isnan(y1) & ~np.isnan(y2)
    x_valid = x[valid]
    y1_valid = y1[valid]
    y2_valid = y2[valid]

    # Calculate the difference between the two curves
    diff = y1_valid - y2_valid

    # Find where the sign of the difference changes
    sign_change = np.where(np.diff(np.sign(diff)))[0]

    # Interpolate to find the exact intersection points
    intersections = []
    for idx in sign_change:
        x0, x1 = x_valid[idx], x_valid[idx + 1]
        y0, y1 = diff[idx], diff[idx + 1]
        x_interp = x0 - y0 * (x1 - x0) / (y1 - y0)
        y_interp = np.interp(x_interp, x_valid, y1_valid)
        intersections.append((x_interp, y_interp))
    return intersections

# Find intersections for expected_angsep and num_matches
intersections_expected_angsep = find_intersection(prop_values, mean_expected_angsep, optresults_expected_angsep)
intersections_expected_adjangsep = find_intersection(prop_values, mean_expected_adjangsep, optresults_expected_adjangsep)
intersections_num_matches = find_intersection(prop_values, mean_num_matches, optresults_num_matches)

# Step 13: Save the necessary outputs for plotting
np.savez('plotting_data.npz',
         prop_values=prop_values,
         mean_expected_angsep=mean_expected_angsep,
         mean_expected_adjangsep=mean_expected_adjangsep,
         mean_num_matches=mean_num_matches,
         mean_var_angsep=mean_var_angsep,
         var_expected_angsep=var_expected_angsep,
         var_expected_adjangsep=var_expected_adjangsep,
         var_num_matches=var_num_matches,
         num_nans_expected_angsep=num_nans_expected_angsep,
         num_nans_num_matches=num_nans_num_matches,
         optresults_expected_angsep=optresults_expected_angsep,
         optresults_expected_adjangsep=optresults_expected_adjangsep,
         optresults_num_matches=optresults_num_matches,
         closest_expected_angsep_prop=closest_expected_angsep_prop,
         closest_expected_angsep_value=closest_expected_angsep_value,
         closest_expected_adjangsep_prop=closest_expected_adjangsep_prop,
         closest_expected_adjangsep_value=closest_expected_adjangsep_value,
         intersections_expected_angsep=intersections_expected_angsep,
         intersections_expected_adjangsep=intersections_expected_adjangsep,
         intersections_num_matches=intersections_num_matches,
         null_siz=null_siz)

# Close the FITS file
f.close()