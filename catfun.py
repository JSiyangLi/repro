import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import skycoord_to_pixel
from astropy import units as u
from circfun import circolap
import matplotlib.pyplot as plt
from scipy.stats import t
from astropy.io import fits
from astropy.wcs import WCS
from bilinear_interpolation import bi_interpolation_psfsize

# Function to calculate angular separation between two points on the sky
def angular_separation(ra1, dec1, ra2, dec2):
    """
    Calculate the angular separation between two points on the sky.

    Parameters:
        ra1 (float): Right ascension of the first point (in degrees).
        dec1 (float): Declination of the first point (in degrees).
        ra2 (float): Right ascension of the second point (in degrees).
        dec2 (float): Declination of the second point (in degrees).

    Returns:
        float: Angular separation (in arcseconds).
    """
    coord1 = SkyCoord(ra=ra1 * u.degree, dec=dec1 * u.degree, frame='icrs')
    coord2 = SkyCoord(ra=ra2 * u.degree, dec=dec2 * u.degree, frame='icrs')
    return coord1.separation(coord2).arcsecond

# Function to parse circle data from a file
def parse_circle_file(filename):
    """
    Parse a file containing circle data in the format 'circle(ra, dec, radius)'.

    Parameters:
        filename (str): Path to the file.

    Returns:
        list: List of tuples containing (ra, dec, radius).
    """
    circles = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('circle'):
                # Remove 'circle(' and ')' and split by commas
                parts = line.strip().replace('circle(', '').replace(')', '').split(',')
                # Extract RA, Dec, and radius, removing 'd' and '"' suffixes
                ra = float(parts[0].replace('d', ''))  # Remove 'd' from RA
                dec = float(parts[1].replace('d', ''))  # Remove 'd' from Dec
                radius = float(parts[2].replace('"', '').replace("''", ''))  # Remove '"' or "''" from radius
                circles.append((ra, dec, radius))
    return circles

def parse_circle_file_pixel(filename):
    """
    Parse a file containing circle data in the format 'circle(x, y, radius)',
    where x and y are pixel coordinates.

    Parameters:
        filename (str): Path to the file.

    Returns:
        list: List of tuples containing (x, y, radius).
    """
    circles = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('circle'):
                # Remove 'circle(' and ')' and split by commas
                parts = line.strip().replace('circle(', '').replace(')', '').split(',')
                # Extract x, y, and radius
                x = float(parts[0])  # Pixel x-coordinate
                y = float(parts[1])  # Pixel y-coordinate
                radius = float(parts[2])  # Radius in pixels
                circles.append((x, y, radius))
    return circles


def extract_theta(filename):
    """
    Extract theta values (third column) from a space/tab-delimited text file.

    Parameters:
        filename: Path to the .xyoffx file (default: "unique_xray.xyoffx")

    Returns:
        numpy.ndarray: Array of theta values in radians
    """
    try:
        # Load all columns, unpack=False keeps columns together
        data = np.loadtxt(filename, unpack=False)

        # Extract third column (index 2) which contains theta values
        theta = data[:, 2]

        return theta

    except Exception as e:
        print(f"Error reading {filename}: {str(e)}")
        raise

def match_xray_optical_sources(xray_detections, optical_catalogues, prop, distance_threshold=0.1, filtering=True, verbose=False):
    """
    Match X-ray detections with optical sources and compute normalized differences.

    Parameters:
        xray_detections (list of tuples): List of X-ray sources as (ra, dec, radius).
        optical_catalogues (list of tuples): List of optical sources as (ra, dec, radius).
        prop (float): Proportion threshold for overlap area.
        distance_threshold (float): Threshold for considering a match (in arcseconds, default: 0.1).
        filtering (bool): If True, remove matched optical sources. If False, only compute statistics.
        verbose (bool): If True, include detailed differences in the output.

    Returns:
        dict: A dictionary containing:
            - num_matches (int): Number of matches (removals if filtering=True).
            - expected_ra_diff (float): Expected RA difference.
            - expected_dec_diff (float): Expected DEC difference.
            - expected_abs_ra_diff (float): Expected absolute RA difference.
            - expected_abs_dec_diff (float): Expected absolute DEC difference.
            - expected_angsep (float): Expected angular separation (normalized by xray_radius).
            - var_ra_diff (float): Variance of RA difference.
            - var_dec_diff (float): Variance of DEC difference.
            - var_abs_ra_diff (float): Variance of absolute RA difference.
            - var_abs_dec_diff (float): Variance of absolute DEC difference.
            - var_angsep (float): Variance of normalized angular separation.
            - filtered_catalogues (list of tuples): Updated optical catalogues (only if filtering=True).
            - optical_removals (list): Indices of removed optical sources (only if filtering=True).
            - ra_differences (list): List of RA differences (only if verbose=True).
            - dec_differences (list): List of DEC differences (only if verbose=True).
            - abs_ra_differences (list): List of absolute RA differences (only if verbose=True).
            - abs_dec_differences (list): List of absolute DEC differences (only if verbose=True).
            - angsep (list): List of normalized angular separations (only if verbose=True).
    """
    # Convert inputs to numpy arrays for vectorized operations
    xray_detections = np.array(xray_detections)
    optical_catalogues = np.array(optical_catalogues)

    # Start with a copy of the original optical catalogues
    filtered_catalogues = optical_catalogues.copy()

    # Lists to store differences (only for matches)
    ra_differences = []
    dec_differences = []
    abs_ra_differences = []
    abs_dec_differences = []
    angsep = []
    adjangsep = []  # List to store normalized angular separations

    # List to store indices of removed optical sources
    optical_removals = []

    # Counter for the number of matches (removals)
    num_matches = 0

    # Loop through each X-ray source
    for xray_ra, xray_dec, xray_radius in xray_detections:
        # Extract optical coordinates
        optical_ra = optical_catalogues[:, 0]
        optical_dec = optical_catalogues[:, 1]
        optical_radius = optical_catalogues[:, 2]

        # Calculate angular separations in a vectorized way
        xray_coord = SkyCoord(ra=xray_ra * u.degree, dec=xray_dec * u.degree, frame='icrs')
        optical_coords = SkyCoord(ra=optical_ra * u.degree, dec=optical_dec * u.degree, frame='icrs')
        dsep = xray_coord.separation(optical_coords).arcsecond

        # Find the closest optical source
        min_dsep_idx = np.argmin(dsep)
        min_dsep = dsep[min_dsep_idx]

        # Check if the closest optical source is within the distance threshold
        if min_dsep < distance_threshold:
            # Record the differences (without dividing by xray_radius)
            ra_diff = optical_ra[min_dsep_idx] - xray_ra
            dec_diff = optical_dec[min_dsep_idx] - xray_dec
            abs_ra_diff = np.abs(ra_diff)
            abs_dec_diff = np.abs(dec_diff)

            # Append to lists
            ra_differences.append(ra_diff)
            dec_differences.append(dec_diff)
            abs_ra_differences.append(abs_ra_diff)
            abs_dec_differences.append(abs_dec_diff)
            angsep.append(min_dsep)
            adjangsep.append(min_dsep / xray_radius)  # Normalize angular separation by xray_radius

            # Remove this optical source from filtered_catalogues (if filtering=True)
            if filtering:
                filtered_catalogues = np.delete(filtered_catalogues, min_dsep_idx, axis=0)
                optical_removals.append(min_dsep_idx)  # Record the index of the removed source
            num_matches += 1  # Increment the match counter
            continue  # Move to the next X-ray source

        # Check for overlap
        if min_dsep < (xray_radius + optical_radius[min_dsep_idx]):
            # Calculate overlap properties using the new circolap function
            overlap_properties = circolap(min_dsep, xray_radius, optical_radius[min_dsep_idx])
            overlap_area = overlap_properties['overlap_area']
            # Check if the overlap area is above the proportion threshold
            if overlap_area > prop * (np.pi * xray_radius**2):
                # Record the differences (without dividing by xray_radius)
                ra_diff = optical_ra[min_dsep_idx] - xray_ra
                dec_diff = optical_dec[min_dsep_idx] - xray_dec
                abs_ra_diff = np.abs(ra_diff)
                abs_dec_diff = np.abs(dec_diff)

                # Append to lists
                ra_differences.append(ra_diff)
                dec_differences.append(dec_diff)
                abs_ra_differences.append(abs_ra_diff)
                abs_dec_differences.append(abs_dec_diff)
                angsep.append(min_dsep)
                adjangsep.append(min_dsep / xray_radius)  # Normalize angular separation by xray_radius

                # Remove this optical source from filtered_catalogues (if filtering=True)
                if filtering:
                    filtered_catalogues = np.delete(filtered_catalogues, min_dsep_idx, axis=0)
                    optical_removals.append(min_dsep_idx)  # Record the index of the removed source
                num_matches += 1  # Increment the match counter

    # Convert lists to numpy arrays for easier calculations
    ra_differences = np.array(ra_differences)
    dec_differences = np.array(dec_differences)
    abs_ra_differences = np.array(abs_ra_differences)
    abs_dec_differences = np.array(abs_dec_differences)
    adjangsep = np.array(adjangsep)  # Convert angular separations to numpy array
    angsep = np.array(angsep)  # Convert angular separations to numpy array

    # Calculate expected values (means) and variances (only if there are matches)
    if num_matches > 0:
        expected_ra_diff = np.mean(ra_differences)
        expected_dec_diff = np.mean(dec_differences)
        expected_abs_ra_diff = np.mean(abs_ra_differences)
        expected_abs_dec_diff = np.mean(abs_dec_differences)
        expected_adjangsep = np.mean(adjangsep)  # Expected normalized angular separation
        expected_angsep = np.mean(angsep)  # Expected normalized angular separation

        var_ra_diff = np.var(ra_differences)
        var_dec_diff = np.var(dec_differences)
        var_abs_ra_diff = np.var(abs_ra_differences)
        var_abs_dec_diff = np.var(abs_dec_differences)
        var_adjangsep = np.var(adjangsep)  # Variance of normalized angular separation
        var_angsep = np.var(angsep)  # Variance of normalized angular separation
    else:
        # If no matches, set expectations and variances to NaN
        expected_ra_diff = np.nan
        expected_dec_diff = np.nan
        expected_abs_ra_diff = np.nan
        expected_abs_dec_diff = np.nan
        expected_angsep = np.nan
        expected_adjangsep = np.nan

        var_ra_diff = np.nan
        var_dec_diff = np.nan
        var_abs_ra_diff = np.nan
        var_abs_dec_diff = np.nan
        var_angsep = np.nan
        var_adjangsep = np.nan

    # Prepare the results
    results = {
        'num_matches': num_matches,
        'expected_ra_diff': expected_ra_diff,
        'expected_dec_diff': expected_dec_diff,
        'expected_abs_ra_diff': expected_abs_ra_diff,
        'expected_abs_dec_diff': expected_abs_dec_diff,
        'expected_angsep': expected_angsep,
        'expected_adjangsep': expected_adjangsep,
        'var_ra_diff': var_ra_diff,
        'var_dec_diff': var_dec_diff,
        'var_abs_ra_diff': var_abs_ra_diff,
        'var_abs_dec_diff': var_abs_dec_diff,
        'var_angsep': var_angsep,
        'var_adjangsep': var_adjangsep,
        'filtered_catalogues': filtered_catalogues if filtering else None,
        'optical_removals': optical_removals if filtering else None,  # Add optical_removals to results
    }

    # Add verbose details if requested
    if verbose:
        results['ra_differences'] = ra_differences
        results['dec_differences'] = dec_differences
        results['abs_ra_differences'] = abs_ra_differences
        results['abs_dec_differences'] = abs_dec_differences
        results['angsep'] = angsep  # Add normalized angular separations
        results['adjangsep'] = adjangsep  # Add normalized angular separations

    return results

def iterate_prop_values(xray_detections, optical_catalogues, distance_threshold, prop_values=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    Iterate over prop values and compute expectations and variances.

    Parameters:
        xray_detections (list of tuples): List of X-ray sources as (ra, dec, radius).
        optical_catalogues (list of tuples): List of optical sources as (ra, dec, radius).
        distance_threshold (float): Threshold for considering a match (in arcseconds).
        prop_values (list of float): List of proportion values to iterate over.

    Returns:
        dict: A dictionary containing results for each prop value, including the prop value that minimizes expected_angsep.
    """
    filtering = False
    # Initialize lists to store results
    results = {
        'prop_values': prop_values,
        'expected_ra_diff': [],
        'expected_dec_diff': [],
        'expected_abs_ra_diff': [],
        'expected_abs_dec_diff': [],
        'expected_angsep': [],  # Add expected angular separation
        'expected_adjangsep': [],
        'var_ra_diff': [],
        'var_dec_diff': [],
        'var_abs_ra_diff': [],
        'var_abs_dec_diff': [],
        'var_angsep': [],  # Add variance of angular separations
        'num_matches': [],
        'min_angsep_prop': None,  # Add prop value that minimizes expected_angsep
        'min_angsep_value': None,  # Add the minimum expected angular separation
    }

    # Iterate over prop values
    for prop in prop_values:
        # Call the matching function
        match_results = match_xray_optical_sources(xray_detections, optical_catalogues, prop, distance_threshold, filtering)

        # Append results to lists
        results['expected_ra_diff'].append(match_results['expected_ra_diff'])
        results['expected_dec_diff'].append(match_results['expected_dec_diff'])
        results['expected_abs_ra_diff'].append(match_results['expected_abs_ra_diff'])
        results['expected_abs_dec_diff'].append(match_results['expected_abs_dec_diff'])
        results['expected_angsep'].append(match_results['expected_angsep'])  # Add expected angular separation
        results['expected_adjangsep'].append(match_results['expected_adjangsep'])
        results['var_ra_diff'].append(match_results['var_ra_diff'])
        results['var_dec_diff'].append(match_results['var_dec_diff'])
        results['var_abs_ra_diff'].append(match_results['var_abs_ra_diff'])
        results['var_abs_dec_diff'].append(match_results['var_abs_dec_diff'])
        results['var_angsep'].append(match_results['var_angsep'])  # Add variance of angular separations
        results['num_matches'].append(match_results['num_matches'])

    # Find the prop value that minimizes expected_angsep
    min_angsep_index = np.nanargmin(results['expected_adjangsep'])
    results['min_angsep_prop'] = prop_values[min_angsep_index]
    results['min_angsep_value'] = results['expected_adjangsep'][min_angsep_index]

    return results

def plot_results(results):
    """
    Plot the results for expectations, variances, and number of matches.

    Parameters:
        results (dict): Dictionary containing results from iterate_prop_values.
    """
    prop_values = results['prop_values']

    # Plot 1: Expected and variance of RA and DEC differences
    plt.figure(figsize=(14, 10))

    # Facet 1: Expected RA difference (in degrees)
    plt.subplot(2, 2, 1)
    plt.plot(prop_values, results['expected_ra_diff'], 'b-', label='Expected RA Diff')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Expected RA Difference (degrees)')
    plt.title('Expected RA Difference vs Proportion Threshold')
    plt.grid(True)

    # Facet 2: Expected DEC difference (in degrees)
    plt.subplot(2, 2, 2)
    plt.plot(prop_values, results['expected_dec_diff'], 'r-', label='Expected DEC Diff')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Expected DEC Difference (degrees)')
    plt.title('Expected DEC Difference vs Proportion Threshold')
    plt.grid(True)

    # Facet 3: Variance of RA difference (in degrees²)
    plt.subplot(2, 2, 3)
    plt.plot(prop_values, results['var_ra_diff'], 'g-', label='Variance RA Diff')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Variance of RA Difference (degrees²)')
    plt.title('Variance of RA Difference vs Proportion Threshold')
    plt.grid(True)

    # Facet 4: Variance of DEC difference (in degrees²)
    plt.subplot(2, 2, 4)
    plt.plot(prop_values, results['var_dec_diff'], 'm-', label='Variance DEC Diff')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Variance of DEC Difference (degrees²)')
    plt.title('Variance of DEC Difference vs Proportion Threshold')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot 2: Expected and variance of absolute RA and DEC differences
    plt.figure(figsize=(14, 10))

    # Facet 1: Expected absolute RA difference (in degrees)
    plt.subplot(2, 2, 1)
    plt.plot(prop_values, results['expected_abs_ra_diff'], 'b-', label='Expected |RA| Diff')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Expected |RA| Difference (degrees)')
    plt.title('Expected |RA| Difference vs Proportion Threshold')
    plt.grid(True)

    # Facet 2: Expected absolute DEC difference (in degrees)
    plt.subplot(2, 2, 2)
    plt.plot(prop_values, results['expected_abs_dec_diff'], 'r-', label='Expected |DEC| Diff')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Expected |DEC| Difference (degrees)')
    plt.title('Expected |DEC| Difference vs Proportion Threshold')
    plt.grid(True)

    # Facet 3: Variance of absolute RA difference (in degrees²)
    plt.subplot(2, 2, 3)
    plt.plot(prop_values, results['var_abs_ra_diff'], 'g-', label='Variance |RA| Diff')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Variance of |RA| Difference (degrees²)')
    plt.title('Variance of |RA| Difference vs Proportion Threshold')
    plt.grid(True)

    # Facet 4: Variance of absolute DEC difference (in degrees²)
    plt.subplot(2, 2, 4)
    plt.plot(prop_values, results['var_abs_dec_diff'], 'm-', label='Variance |DEC| Diff')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Variance of |DEC| Difference (degrees²)')
    plt.title('Variance of |DEC| Difference vs Proportion Threshold')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Plot 4: Angular separation metrics
    plt.figure(figsize=(14, 10))

    # Facet 1: Expected angular separation (dimensionless) with error bars (sqrt(variance))
    plt.subplot(2, 2, 1)
    plt.errorbar(
        prop_values,
        results['expected_angsep'],
        yerr=np.sqrt(results['var_angsep']),  # Error bars = sqrt(variance)
        fmt='bo-',  # Blue circles with lines
        capsize=5,  # Cap size for error bars
        label='Expected Angular Separation ± √(Variance)'
    )
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Expected Angular Separation (dimensionless)')
    plt.title('Expected Angular Separation vs Proportion Threshold')
    plt.grid(True)
    plt.legend()

    # Facet 2: Variance of angular separation (dimensionless)
    plt.subplot(2, 2, 2)
    plt.plot(prop_values, results['var_angsep'], 'r-', label='Variance of Angular Separation')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Variance of Angular Separation (dimensionless)')
    plt.title('Variance of Angular Separation vs Proportion Threshold')
    plt.grid(True)

    # Facet 3: Number of matches
    plt.subplot(2, 2, 3)
    plt.plot(prop_values, results['num_matches'], 'k-', label='Number of Matches')
    plt.xlabel('Proportion Threshold (prop)')
    plt.ylabel('Number of Matches')
    plt.title('Number of Matches vs Proportion Threshold')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_density_results(results):
    """
    Plot density distributions for the results at a specific prop value.

    Parameters:
        results (dict): Dictionary containing results from match_xray_optical_sources.
    """
    # Extract results
    ra_differences = results['ra_differences']
    dec_differences = results['dec_differences']
    abs_ra_differences = results['abs_ra_differences']
    abs_dec_differences = results['abs_dec_differences']
    num_matches = results['num_matches']

    # Plot: Densities of RA, DEC, absolute RA, and absolute DEC differences
    plt.figure(figsize=(14, 10))

    # Facet 1: Density of RA differences
    plt.subplot(2, 2, 1)
    plt.hist(ra_differences, bins=20, density=True, alpha=0.6, color='b', label='RA Differences')
    plt.xlabel('RA Differences')
    plt.ylabel('Density')
    plt.title(f'Density of RA Differences')
    plt.grid(True)

    # Facet 2: Density of DEC differences
    plt.subplot(2, 2, 2)
    plt.hist(dec_differences, bins=20, density=True, alpha=0.6, color='r', label='DEC Differences')
    plt.xlabel('DEC Differences')
    plt.ylabel('Density')
    plt.title(f'Density of DEC Differences')
    plt.grid(True)

    # Facet 3: Density of absolute RA differences
    plt.subplot(2, 2, 3)
    plt.hist(abs_ra_differences, bins=20, density=True, alpha=0.6, color='g', label='|RA| Differences')
    plt.xlabel('|RA| Differences')
    plt.ylabel('Density')
    plt.title(f'Density of |RA| Differences')
    plt.grid(True)

    # Facet 4: Density of absolute DEC differences
    plt.subplot(2, 2, 4)
    plt.hist(abs_dec_differences, bins=20, density=True, alpha=0.6, color='m', label='|DEC| Differences')
    plt.xlabel('|DEC| Differences')
    plt.ylabel('Density')
    plt.title(f'Density of |DEC| Differences')
    plt.grid(True)

    # Add num_matches at the bottom of the plot
    plt.figtext(0.5, 0.01, f'number of matches = {num_matches}', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()

def conduct_t_tests(results):
    """
    Conduct t-tests for expected differences and variances.

    Parameters:
        results (dict): Dictionary containing results from match_xray_optical_sources.

    Returns:
        bool: True if both t-tests are significant, False otherwise.
    """
    # Extract results
    expected_ra_diff = results['expected_ra_diff']
    expected_dec_diff = results['expected_dec_diff']
    var_ra_diff = results['var_ra_diff']
    var_dec_diff = results['var_dec_diff']
    num_matches = results['num_matches']

    # Degrees of freedom
    df = num_matches - 1

    # T-test for expected RA difference
    t_stat_ra = expected_ra_diff / np.sqrt(var_ra_diff / num_matches)
    p_val_ra = 2 * (1 - t.cdf(np.abs(t_stat_ra), df))  # Two-tailed test

    # T-test for expected DEC difference
    t_stat_dec = expected_dec_diff / np.sqrt(var_dec_diff / num_matches)
    p_val_dec = 2 * (1 - t.cdf(np.abs(t_stat_dec), df))  # Two-tailed test

    # Print results
    print(f"T-test for expected RA difference: t-statistic = {t_stat_ra}, p-value = {p_val_ra}")
    print(f"T-test for expected DEC difference: t-statistic = {t_stat_dec}, p-value = {p_val_dec}")

    # Determine if adjustments are needed
    need_adjust = (p_val_ra < 0.05) and (p_val_dec < 0.05)
    print(f"need_adjust = {need_adjust}")

    return need_adjust

import numpy as np

def is_mean_significantly_different(prop_value, data, tolerance=1e-10):
    """
    Check if, for a given prop_value, the mean_expected_angsep is >= 2 * std_error_expected_angsep
    away from optresults_expected_angsep.

    Parameters:
        prop_value (float): The proportion threshold value to check.
        data (dict): The data loaded from 'plotting_data.npz'.
        tolerance (float): Tolerance for floating-point comparison. Default is 1e-10.

    Returns:
        bool: True if the condition is met, False otherwise.
    """
    # Extract necessary arrays from the data
    prop_values = data['prop_values']
    mean_expected_angsep = data['mean_expected_angsep']
    optresults_expected_angsep = data['optresults_expected_angsep']
    std_error_expected_angsep = np.sqrt(data['mean_var_angsep']) / np.sqrt(data['null_siz'] - data['num_nans_expected_angsep'])

    # Find the index corresponding to the given prop_value (with tolerance)
    idx = np.where(np.isclose(prop_values, prop_value, atol=tolerance))[0]
    if len(idx) == 0:
        raise ValueError(f"prop_value {prop_value} not found in prop_values array within tolerance {tolerance}.")

    idx = idx[0]  # Get the first matching index

    # Calculate the absolute difference between mean and original expected angular separation
    difference = np.abs(mean_expected_angsep[idx] - optresults_expected_angsep[idx])

    # Check if the difference is >= 2 * std_error_expected_angsep
    return difference >= 2 * std_error_expected_angsep[idx]

def adjust_optical_catalogues(optical_catalogues, expected_ra_diff, expected_dec_diff, fits_name="27_b32_psf.fits"):
    """
    Adjust the RA, DEC, and radii values in optical_catalogues based on expected differences and PSF size.

    Parameters:
        optical_catalogues (list of tuples): List of optical sources as (ra, dec, radius).
        expected_ra_diff (float): Expected normalized RA difference.
        expected_dec_diff (float): Expected normalized DEC difference.
        fits_name (str): Path to the PSF FITS file.

    Returns:
        tuple: A tuple containing:
            - ra_list: List of adjusted RA values.
            - dec_list: List of adjusted DEC values.
            - adj_optcat: Adjusted optical catalogues as a list of tuples (ra, dec, radius).
    """
    # Load the PSF FITS file
    f = fits.open(fits_name)

    # Extract WCS information from the FITS header
    w = WCS(f[0].header)

    ra_list = []
    dec_list = []
    adj_optcat = []

    for ra, dec, radius in optical_catalogues:
        # Adjust RA and DEC (subtract to reduce the differences)
        adjusted_ra = ra - (expected_ra_diff * radius)
        adjusted_dec = dec - (expected_dec_diff * radius)

        # Convert adjusted RA and DEC to pixel coordinates
        sc = SkyCoord(ra=adjusted_ra * u.degree, dec=adjusted_dec * u.degree)
        xpos_pix, ypos_pix = skycoord_to_pixel(sc, w)

        # Load the PSF size using bilinear interpolation with pixel coordinates
        psfsiz = bi_interpolation_psfsize(f[0].data, xpos_pix, ypos_pix)

        # Use the PSF size as the adjusted radius
        adjusted_radius = psfsiz  # Directly use the PSF size as the new radius

        # Append to lists
        ra_list.append(adjusted_ra)
        dec_list.append(adjusted_dec)
        adj_optcat.append((adjusted_ra, adjusted_dec, adjusted_radius))

    # Close the FITS file
    f.close()

    return ra_list, dec_list, adj_optcat

def filter_xray_optical_xyoffx(xray_xyoffx, optical_xyoffx, optical_removals):
    """
    Filter X-ray and optical coordinates based on removed optical sources.

    Parameters:
        xray_xyoffx (str): Path to the file containing X-ray coordinates (output of dm_xyoffx_to_file).
        optical_xyoffx (str): Path to the file containing optical coordinates (output of dm_xyoffx_to_file).
        optical_removals (list): Indices of removed optical sources (from match_xray_optical_sources).

    Returns:
        list: A list of tuples containing (xpos, ypos, theta, phi) for the remaining optical sources.
    """
    # Load X-ray coordinates
    xray_coords = np.loadtxt(xray_xyoffx)
    print(f"Number of X-ray coordinates: {len(xray_coords)}")

    # Load optical coordinates
    optical_coords = np.loadtxt(optical_xyoffx)
    print(f"Number of optical coordinates before filtering: {len(optical_coords)}")

    # Filter out removed optical sources
    filtered_optical_coords = np.delete(optical_coords, optical_removals, axis=0)
    print(f"Number of optical coordinates after filtering: {len(filtered_optical_coords)}")

    # Combine X-ray and filtered optical coordinates
    combined_coords = np.vstack((xray_coords, filtered_optical_coords))
    print(f"Total number of combined coordinates: {len(combined_coords)}")

    # Convert to a list of tuples (xpos, ypos, theta, phi)
    result = [tuple(coord) for coord in combined_coords]

    return result

def filter_xyoffx_file(xray_xyoffx, optical_xyoffx, optical_removals, output_file):
    """
    Filter X-ray and optical coordinates based on removed optical sources and write the results to a file.

    Parameters:
        xray_xyoffx (str): Path to the file containing X-ray coordinates (output of dm_xyoffx_to_file).
        optical_xyoffx (str): Path to the file containing optical coordinates (output of dm_xyoffx_to_file).
        optical_removals (list): Indices of removed optical sources (from match_xray_optical_sources).
        output_file (str): Path to the output file where filtered coordinates will be saved.
    """
    # Get the filtered coordinates using filter_xray_optical_xyoffx
    filtered_coords = filter_xray_optical_xyoffx(xray_xyoffx, optical_xyoffx, optical_removals)

    # Write the filtered coordinates to the output file
    with open(output_file, 'w') as f:
        for coord in filtered_coords:
            f.write(f"{coord[0]}\t{coord[1]}\t{coord[2]}\t{coord[3]}\n")

# Plotting functionality
def plot_cumulative_properties(max_dist_order, total_dist_order, number_order, cumulative_maxdist, cumulative_totaldist, cumulative_number, num_circles):
    # Plot 1: Cumulative maximum distance
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(max_dist_order) + 1), cumulative_maxdist, linestyle='-', linewidth=1, color='blue')
    plt.xlabel('Max Distance Order')
    plt.ylabel('Cumulative Max Distance (arcseconds)')
    plt.title('Cumulative Max Distance of Groups (Sorted by Max Distance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_maxdist_plot.png')
    plt.show()

    # Plot 2: Cumulative total distance
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(total_dist_order) + 1), cumulative_totaldist, linestyle='-', linewidth=1, color='blue')
    plt.xlabel('Total Distance Order')
    plt.ylabel('Cumulative Total Distance (arcseconds)')
    plt.title('Cumulative Total Distance of Groups (Sorted by Total Distance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_totaldist_plot.png')
    plt.show()

    # Plot 3: Cumulative Number of Groups
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(number_order) + 1), cumulative_number, linestyle='-', linewidth=1, color='green')
    plt.xlabel('Number Order')
    plt.ylabel('Cumulative Number of Groups')
    plt.title('Cumulative Number of Groups (Sorted by Number of Circles)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cumulative_number_plot.png')
    plt.show()

    # Plot 4: Bar Plot for Group Sizes
    unique_counts, counts = np.unique(num_circles, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique_counts, counts, color='purple')
    plt.xlabel('Number of Circles in Group')
    plt.ylabel('Number of Groups')
    plt.title('Distribution of Group Sizes')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('group_size_distribution.png')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_group_properties(max_dist_order, total_dist_order, number_order, max_distance, total_distance, num_circles):
    """
    Plots non-cumulative properties of circle groups.

    Parameters:
        max_distance (np.array): Array of maximum distances for each group.
        total_distance (np.array): Array of total distances for each group.
        num_circles (np.array): Array of the number of circles in each group.
    """

    # Plot 1: Non-cumulative maximum distance
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(max_dist_order) + 1), max_distance[max_dist_order], linestyle='-', linewidth=1, color='blue')
    plt.xlabel('Group Index (Sorted by Max Distance)')
    plt.ylabel('Max Distance (arcseconds)')
    plt.title('Max Distance of Groups (Sorted by Max Distance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('max_distance_plot.png')
    plt.show()

    # Plot 2: Non-cumulative total distance
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(total_dist_order) + 1), total_distance[total_dist_order], linestyle='-', linewidth=1, color='blue')
    plt.xlabel('Group Index (Sorted by Total Distance)')
    plt.ylabel('Total Distance (arcseconds)')
    plt.title('Total Distance of Groups (Sorted by Total Distance)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('total_distance_plot.png')
    plt.show()

    # Plot 3: Non-cumulative number of circles
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(1, len(number_order) + 1), num_circles[number_order], linestyle='-', linewidth=1, color='green')
    plt.xlabel('Group Index (Sorted by Number of Circles)')
    plt.ylabel('Number of Circles')
    plt.title('Number of Circles in Groups (Sorted by Number of Circles)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('number_of_circles_plot.png')
    plt.show()

def remove_groups(max_dist_order, total_dist_order, number_order, group_id, group_min_theta, theta):
    """
    Remove circles based on the minimum theta of specified groups.

    Parameters:
        max_dist_order (np.ndarray): Order of groups sorted by maximum pairwise distance.
        total_dist_order (np.ndarray): Order of groups sorted by total pairwise distance.
        number_order (np.ndarray): Order of groups sorted by number of circles.
        group_id (np.ndarray): IDs of the groups.
        max_distance (np.ndarray): Maximum pairwise distance of each group.
        total_distance (np.ndarray): Total pairwise distance of each group.
        num_circles (np.ndarray): Number of circles in each group.
        group_min_theta (np.ndarray): Minimum theta of each group.
        theta (np.ndarray): Off-axis angles of the circles.

    Returns:
        np.ndarray: Indices of circles that should NOT be removed.
    """
    while True:
        user_input = input("Enter removal command (e.g., 'rm_max_dist_order=N', 'rm_total_dist_order=N', 'rm_number_order=N', or 'done'): ").strip()

        if user_input.lower() == 'done':
            break

        if user_input.startswith('rm_max_dist_order='):
            try:
                rm_order = int(user_input.split('=')[1])
                # Find the group IDs corresponding to the area order (up to rm_order)
                rm_ids = [group_id[max_dist_order[i]] for i in range(rm_order)]
                # Find the smallest theta among the specified groups
                min_theta = min(group_min_theta[np.where(group_id == rm_id)[0][0]] for rm_id in rm_ids)
                # Remove all circles with theta > min_theta
                remove_indices = np.where(theta > min_theta)[0]
                print(f"Removing circles with theta > {min_theta}: {len(remove_indices)} circles")
            except (IndexError, ValueError):
                print("Invalid input. Please use 'rm_max_dist_order=N' where N is a valid order.")
                continue

        elif user_input.startswith('rm_total_dist_order='):
            try:
                rm_order = int(user_input.split('=')[1])
                # Find the group IDs corresponding to the area order (up to rm_order)
                rm_ids = [group_id[total_dist_order[i]] for i in range(rm_order)]
                # Find the smallest theta among the specified groups
                min_theta = min(group_min_theta[np.where(group_id == rm_id)[0][0]] for rm_id in rm_ids)
                # Remove all circles with theta > min_theta
                remove_indices = np.where(theta > min_theta)[0]
                print(f"Removing circles with theta > {min_theta}: {len(remove_indices)} circles")
            except (IndexError, ValueError):
                print("Invalid input. Please use 'rm_total_dist_order=N' where N is a valid order.")
                continue

        elif user_input.startswith('rm_number_order='):
            try:
                rm_order = int(user_input.split('=')[1])
                # Find the group IDs corresponding to the number order (up to rm_order)
                rm_ids = [group_id[number_order[i]] for i in range(rm_order)]
                # Find the smallest theta among the specified groups
                min_theta = min(group_min_theta[np.where(group_id == rm_id)[0][0]] for rm_id in rm_ids)
                # Remove all circles with theta > min_theta
                remove_indices = np.where(theta > min_theta)[0]
                print(f"Removing circles with theta > {min_theta}: {len(remove_indices)} circles")
            except (IndexError, ValueError):
                print("Invalid input. Please use 'rm_number_order=N' where N is a valid order.")
                continue

        else:
            print("Invalid command. Please try again.")
            continue

        # Update the mask to exclude circles with theta > min_theta
        mask = np.ones(len(theta), dtype=bool)
        mask[remove_indices] = False
        theta = theta[mask]  # Update theta for the next iteration

    # Return the indices of circles that should NOT be removed
    return np.where(mask)[0]

def get_min_theta(max_dist_order, total_dist_order, number_order, group_id, group_min_theta):
    """
    Calculate the minimum theta based on the user's removal command input.

    Parameters:
        max_dist_order (np.ndarray): Order of groups sorted by maximum pairwise distance.
        total_dist_order (np.ndarray): Order of groups sorted by total pairwise distance.
        number_order (np.ndarray): Order of groups sorted by number of circles.
        group_id (np.ndarray): IDs of the groups.
        group_min_theta (np.ndarray): Minimum theta of each group.
        theta (np.ndarray): Off-axis angles of the circles.

    Returns:
        float: The minimum theta value based on the user's input.
    """
    while True:
        user_input = input("Enter removal command (e.g., 'rm_max_dist_order=N', 'rm_total_dist_order=N', 'rm_number_order=N', or 'done'): ").strip()

        if user_input.lower() == 'done':
            print("Exiting without calculating min_theta.")
            return None

        if user_input.startswith('rm_max_dist_order='):
            try:
                rm_order = int(user_input.split('=')[1])
                # Find the group IDs corresponding to the max_dist_order (up to rm_order)
                rm_ids = [group_id[max_dist_order[i]] for i in range(rm_order)]
                # Find the smallest theta among the specified groups
                min_theta = min(group_min_theta[np.where(group_id == rm_id)[0][0]] for rm_id in rm_ids)
                print(f"Minimum theta for rm_max_dist_order={rm_order}: {min_theta}")
                return min_theta
            except (IndexError, ValueError):
                print("Invalid input. Please use 'rm_max_dist_order=N' where N is a valid order.")
                continue

        elif user_input.startswith('rm_total_dist_order='):
            try:
                rm_order = int(user_input.split('=')[1])
                # Find the group IDs corresponding to the total_dist_order (up to rm_order)
                rm_ids = [group_id[total_dist_order[i]] for i in range(rm_order)]
                # Find the smallest theta among the specified groups
                min_theta = min(group_min_theta[np.where(group_id == rm_id)[0][0]] for rm_id in rm_ids)
                print(f"Minimum theta for rm_total_dist_order={rm_order}: {min_theta}")
                return min_theta
            except (IndexError, ValueError):
                print("Invalid input. Please use 'rm_total_dist_order=N' where N is a valid order.")
                continue

        elif user_input.startswith('rm_number_order='):
            try:
                rm_order = int(user_input.split('=')[1])
                # Find the group IDs corresponding to the number_order (up to rm_order)
                rm_ids = [group_id[number_order[i]] for i in range(rm_order)]
                # Find the smallest theta among the specified groups
                min_theta = min(group_min_theta[np.where(group_id == rm_id)[0][0]] for rm_id in rm_ids)
                print(f"Minimum theta for rm_number_order={rm_order}: {min_theta}")
                return min_theta
            except (IndexError, ValueError):
                print("Invalid input. Please use 'rm_number_order=N' where N is a valid order.")
                continue

        else:
            print("Invalid command. Please try again.")
            continue

def compare_row_counts(file_or_array1, file_or_array2):
    """
    Compare the number of rows in two files or between a file and the length of an array.

    Parameters:
        file_or_array1 (str or list/np.ndarray): Path to the first file or an array.
        file_or_array2 (str or list/np.ndarray or int): Path to the second file, an array, or an integer.

    Returns:
        bool: True if the row counts match, False otherwise.
    """
    # Get row count for the first input
    if isinstance(file_or_array1, str):
        with open(file_or_array1, 'r') as f1:
            row_count1 = sum(1 for _ in f1)
    elif isinstance(file_or_array1, (list, np.ndarray)):
        row_count1 = len(file_or_array1)
    elif isinstance(file_or_array1, int):
        row_count1 = file_or_array1
    else:
        raise TypeError("file_or_array1 must be a file path (str), an array, or an integer.")

    # Get row count for the second input
    if isinstance(file_or_array2, str):
        with open(file_or_array2, 'r') as f2:
            row_count2 = sum(1 for _ in f2)
    elif isinstance(file_or_array2, (list, np.ndarray)):
        row_count2 = len(file_or_array2)
    elif isinstance(file_or_array2, int):
        row_count2 = file_or_array2
    else:
        raise TypeError("file_or_array2 must be a file path (str), an array, or an integer.")

    # Compare row counts
    if row_count1 == row_count2:
        print(f"Row counts match: {row_count1}")
        return True
    else:
        print(f"Row counts differ: {row_count1} vs {row_count2}")
        return False

def combine_txt_files(file1, file2, output_file):
    """
    Row-combine two .txt files and write the result to an output file.

    Parameters:
        file1 (str): Path to the first input file.
        file2 (str): Path to the second input file.
        output_file (str): Path to the output file.
    """
    # Load data from the first file
    data1 = np.loadtxt(file1)
    print(f"Number of rows in {file1}: {len(data1)}")

    # Load data from the second file
    data2 = np.loadtxt(file2)
    print(f"Number of rows in {file2}: {len(data2)}")

    # Combine the data row-wise
    combined_data = np.vstack((data1, data2))
    print(f"Total number of rows in combined file: {len(combined_data)}")

    # Write the combined data to the output file
    np.savetxt(output_file, combined_data, delimiter="\t", fmt="%.6f")