import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import sys
from itertools import combinations
from scipy.stats import chi2, multivariate_normal
from scipy.sparse import csr_matrix, coo_matrix, save_npz, lil_matrix, issparse
from scipy.sparse import diags, issparse, csr_matrix, hstack, vstack
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from multiprocessing import Pool, cpu_count
from scipy.spatial import ConvexHull
import psutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from math import floor
from tqdm import tqdm
from bilinear_interpolation import bi_interpolation_psfsize, vectorized_bi_interpolation
from ciao_contrib.runtool import dmstat
from numpy import pi
import pickle  # <-- Added missing import
import warnings
from math import pi
from scipy.sparse import lil_matrix, issparse
import subprocess
import re

################################
### other IDL-translated codes
################################
# Function to calculate angular separation
def deltang(ra1, dec1, ra2, dec2):
    coord1 = SkyCoord(ra1 * u.deg, dec1 * u.deg, frame='icrs')
    coord2 = SkyCoord(ra2 * u.deg, dec2 * u.deg, frame='icrs')
    return coord1.separation(coord2).arcminute

def circolap(dsep, rad1, rad2):
    """
    Compute the area of overlap between two circles and additional geometric properties.

    Parameters:
        dsep (float or np.ndarray): Separation between the centers of the two circles.
        rad1 (float or np.ndarray): Radius of the first circle.
        rad2 (float or np.ndarray): Radius of the second circle.

    Returns:
        dict: A dictionary containing:
            - 'overlap_area': Area of overlap between the two circles.
            - 'chord_length': Length of the chord connecting the intersection points.
            - 'theta1': Opening angle on circle 1 that subtends the chord [degrees].
            - 'theta2': Opening angle on circle 2 that subtends the chord [degrees].
            - 'd2ch1': Perpendicular distance from the center of circle 1 to the chord.
            - 'd2ch2': Perpendicular distance from the center of circle 2 to the chord.
    """
    # Ensure inputs are numpy arrays for vectorized operations
    dsep = np.asarray(dsep)
    rad1 = np.asarray(rad1)
    rad2 = np.asarray(rad2)

    # Initialize outputs
    overlap_area = np.zeros_like(dsep)
    chord_length = np.zeros_like(dsep)
    theta1 = np.zeros_like(dsep)
    theta2 = np.zeros_like(dsep)
    d2ch1 = np.zeros_like(dsep)
    d2ch2 = np.zeros_like(dsep)

    # Special cases
    no_overlap = dsep >= (rad1 + rad2)  # No overlap
    fully_enclosed = dsep <= np.abs(rad1 - rad2)  # One circle fully inside the other
    regular_case = ~no_overlap & ~fully_enclosed  # Regular overlapping case

    # No overlap case
    overlap_area[no_overlap] = 0.0
    chord_length[no_overlap] = 0.0
    theta1[no_overlap] = 0.0
    theta2[no_overlap] = 0.0
    d2ch1[no_overlap] = np.nan
    d2ch2[no_overlap] = np.nan

    # Fully enclosed case
    overlap_area[fully_enclosed] = np.minimum(np.pi * rad1[fully_enclosed]**2, np.pi * rad2[fully_enclosed]**2)
    chord_length[fully_enclosed] = np.nan
    theta1[fully_enclosed] = 360.0
    theta2[fully_enclosed] = 360.0
    d2ch1[fully_enclosed] = np.nan
    d2ch2[fully_enclosed] = np.nan

    # Regular overlapping case
    if np.any(regular_case):
        d = dsep[regular_case]
        r1 = rad1[regular_case]
        r2 = rad2[regular_case]

        # Intermediate calculations
        xx = (d**2 - r1**2 + r2**2) / (2 * d)
        yy = np.sqrt((-d + r1 - r2) * (-d - r1 + r2) * (-d + r1 + r2) * (d + r1 + r2)) / d

        # Overlap area
        term1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        term2 = r2**2 * np.arccos((d**2 - r1**2 + r2**2) / (2 * d * r2))
        term3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        overlap_area[regular_case] = term1 + term2 - term3

        # Chord length
        chord_length[regular_case] = yy

        # Perpendicular distances
        d2ch1[regular_case] = xx
        d2ch2[regular_case] = d - xx

        # Opening angles (in degrees)
        theta1[regular_case] = np.arctan2(yy / 2, xx) * 180 / np.pi
        theta2[regular_case] = np.arctan2(yy / 2, d - xx) * 180 / np.pi

    # Return results as a dictionary
    return {
        'overlap_area': overlap_area,
        'chord_length': chord_length,
        'theta1': theta1,
        'theta2': theta2,
        'd2ch1': d2ch1,
        'd2ch2': d2ch2
    }

def circcumulate(xpos, ypos, rad, theta, thresh=0.1):
    """
    Vectorized version of circle grouping with overlap calculation.

    Parameters:
        xpos, ypos, rad, theta: Arrays of circle properties
        thresh: Overlap threshold for grouping

    Returns:
        Tuple: (group_indices, group_max_distance, group_total_distance,
               group_num_circles, group_min_theta)
    """
    nx = len(xpos)
    if nx != len(ypos) or nx != len(rad) or nx != len(theta):
        raise ValueError("All input arrays must have the same length")

    # Initialize group assignments
    ig = np.full(nx, -1, dtype=int)

    # Pre-compute all pairwise distances
    dx = xpos[:, None] - xpos
    dy = ypos[:, None] - ypos
    dsep = np.sqrt(dx ** 2 + dy ** 2)

    # Pre-compute all radius sums and differences
    rad_sum = rad[:, None] + rad
    rad_diff = np.abs(rad[:, None] - rad)

    # Create masks for different overlap cases
    no_overlap = dsep >= rad_sum
    fully_enclosed = dsep <= rad_diff

    # Calculate overlap areas for all pairs
    overlap_area = np.zeros_like(dsep)

    # Regular overlap case (vectorized)
    regular_case = ~no_overlap & ~fully_enclosed
    if np.any(regular_case):
        # Get indices where regular_case is True
        rows, cols = np.where(regular_case)
        d = dsep[rows, cols]
        r1 = rad[rows]
        r2 = rad[cols]

        term1 = r1 ** 2 * np.arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
        term2 = r2 ** 2 * np.arccos((d ** 2 - r1 ** 2 + r2 ** 2) / (2 * d * r2))
        term3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) *
                              (d - r1 + r2) * (d + r1 + r2))

        # Update only the regular_case positions
        overlap_area[rows, cols] = term1 + term2 - term3

    # Fully enclosed case (vectorized)
    # Get indices where fully_enclosed is True
    rows, cols = np.where(fully_enclosed)
    r1 = rad[rows]
    r2 = rad[cols]
    overlap_area[rows, cols] = np.minimum(np.pi * r1 ** 2, np.pi * r2 ** 2)

    # Calculate fractional overlap areas
    with np.errstate(divide='ignore', invalid='ignore'):
        farea = overlap_area / (np.pi * rad ** 2)

    # Create adjacency matrix based on threshold
    adj_matrix = (farea >= thresh) & ~np.eye(nx, dtype=bool)

    # Group connected components
    n_components, ig = connected_components(adj_matrix, directed=False)

    # Compute group properties
    group_max_distance = []
    group_total_distance = []
    group_num_circles = []
    group_min_theta = []

    for group_id in range(n_components):
        members = (ig == group_id)
        group_size = np.sum(members)
        group_num_circles.append(group_size)

        if group_size == 1:
            group_max_distance.append(0.0)
            group_total_distance.append(0.0)
        else:
            # Extract pairwise distances for this group
            group_dists = dsep[members][:, members]
            group_max_distance.append(np.max(group_dists))
            group_total_distance.append(np.sum(group_dists) / 2)

        group_min_theta.append(np.min(theta[members]))

    return ig, group_max_distance, group_total_distance, group_num_circles, group_min_theta

def ccircle(xc, yc=None, rad=None, xrange=None, yrange=None, nx=512, ny=512, norm=False, small=False, rand=False):
    """
    Returns a bit image of a circle.

    Parameters:
        xc (float): X-coordinate of the circle's center.
        yc (float): Y-coordinate of the circle's center. If not provided, defaults to xc.
        rad (float): Radius of the circle. If not provided, defaults to 10.
        xrange (list): Range of x-values [xmin, xmax]. Defaults to [1, 512].
        yrange (list): Range of y-values [ymin, ymax]. Defaults to [1, 512].
        nx (int): Number of elements in the x-direction. Defaults to 512.
        ny (int): Number of elements in the y-direction. Defaults to 512.
        norm (bool): If True, normalizes the image by its maximum value.
        small (bool): If True, returns a smaller image containing only the circle.
        rand (bool): If True, uses Monte Carlo sampling for small circles.

    Returns:
        np.ndarray: A 2D array representing the circle image.
    """
    # Handle default values for yc and rad
    if yc is None:
        yc = xc
    if rad is None:
        rad = 10.0

    # Handle default values for xrange and yrange
    if xrange is None:
        xrange = [1.0, 512.0]
    if yrange is None:
        yrange = [1.0, 512.0]

    # Create the grid
    xstp = (xrange[1] - xrange[0]) / float(nx - 1)
    ystp = (yrange[1] - yrange[0]) / float(ny - 1)
    x = np.arange(nx) * xstp + xrange[0]
    y = np.arange(ny) * ystp + yrange[0]

    # Initialize the image
    img = np.zeros((nx, ny), dtype=int)

    # Define the bounding box for the circle
    x0 = xc - rad - xstp
    x1 = xc + rad + xstp
    y0 = yc - rad - ystp
    y1 = yc + rad + ystp

    # Clip the bounding box to the grid range
    x0 = max(x0, xrange[0])
    x1 = min(x1, xrange[1])
    y0 = max(y0, yrange[0])
    y1 = min(y1, yrange[1])

    # Find the indices corresponding to the bounding box
    i0 = np.searchsorted(x, x0, side='left')
    i1 = np.searchsorted(x, x1, side='right') - 1
    j0 = np.searchsorted(y, y0, side='left')
    j1 = np.searchsorted(y, y1, side='right') - 1

    # Handle small circles with Monte Carlo sampling
    if rand and (i1 - i0 + 1) * (j1 - j0 + 1) < 100:
        npt = 10 * (i1 - i0 + 1) * (j1 - j0 + 1)
        xpt = (x[i1] - x[i0] + 1.0) * np.random.random(npt) + x[i0] - 0.5
        ypt = (y[j1] - y[j0] + 1.0) * np.random.random(npt) + y[j0] - 0.5
        rpt = (xpt - xc)**2 + (ypt - yc)**2
        inside = rpt <= rad**2
        ix = np.floor(xpt[inside] + 0.5).astype(int)
        iy = np.floor(ypt[inside] + 0.5).astype(int)
        for i in range(len(ix)):
            img[ix[i], iy[i]] += 1
    else:
        # Draw the circle using a grid-based approach
        for i in range(i0, i1 + 1):
            z = np.arange(j0, j1 + 1)
            tmp = (x[i] - xc)**2 + (y[j0:j1 + 1] - yc)**2
            inside = tmp <= rad**2
            img[i, z[inside]] = 1

    # Normalize the image if requested
    if norm:
        img = img / np.max(img)

    # Extract a smaller image if requested
    if small:
        img = img[i0:i1 + 1, j0:j1 + 1]

    return img

###########################
### Functions for design-matrix construction
### Part 1: basic properties of input
##########################

def should_sparse(xc, yc, rc, mem_threshold=0.1, overlap_ratio_threshold=0.2, verbose=True):
    """
    Determine if sparse matrices should be used, with decision printing.

    Parameters:
        xc, yc, rc (np.ndarray): Circle parameters
        mem_threshold (float): Max memory fraction for dense (0-1)
        overlap_ratio_threshold (float): Overlap density threshold
        verbose (bool): Whether to print decision details

    Returns:
        bool: True if sparse recommended, False otherwise
    """
    n_circles = len(xc)
    decision = False
    reason = ""

    # 1. Small system check
    if n_circles < 50:
        decision = False
        reason = f"System too small (N={n_circles} < 50)"

    # 2. Memory requirement check
    else:
        dense_mem_gb = (n_circles ** 2) * 8 / (1024 ** 3)
        avail_mem_gb = 0.8 * (sys.getsizeof([0] * 1000000) / (1024 ** 3))  # Approximate available memory

        if dense_mem_gb > mem_threshold * avail_mem_gb:
            decision = True
            reason = f"Memory savings ({dense_mem_gb:.2f}GB dense > {mem_threshold * 100:.0f}% of available {avail_mem_gb:.2f}GB)"

        # 3. Overlap density check
        else:
            avg_radius = np.mean(rc)
            area = (np.max(xc) - np.min(xc) + 2 * avg_radius) * \
                   (np.max(yc) - np.min(yc) + 2 * avg_radius)
            circle_area = np.pi * avg_radius ** 2
            overlap_ratio = (n_circles * circle_area) / area

            if overlap_ratio < overlap_ratio_threshold:
                decision = True
                reason = f"Low expected overlap density ({overlap_ratio:.2f} < {overlap_ratio_threshold})"
            else:
                decision = False
                reason = f"High expected overlap density ({overlap_ratio:.2f} >= {overlap_ratio_threshold})"

    if verbose:
        print(f"SPARSE DECISION: {'USE SPARSE' if decision else 'USE DENSE'}")
        print(f"Reason: {reason}")
        print(f"System stats: N={n_circles}, avg_r={np.mean(rc):.2f}")

    return decision

def n_overlap(xc, yc, rc, evtX, evtY, sparse=None):
    """
    Optimized geometric overlap calculation with optional sparse outputs.

    Parameters:
        xc, yc, rc: Arrays of circle centers and radii
        evtX, evtY: Arrays of event coordinates
        sparse: If True, returns overlap_matrix as CSR format

    Returns:
        dict: {
            'dsep_matrix': (n,n) center distances (always dense),
            'overlap_matrix': (n,n) binary overlap (sparse/dense),
            'overlap_indices': [array] per-circle overlaps,
            'evtbyt': (n_events, n_circles) membership matrix (always dense),
            'uroi': unique ROI patterns,
            'nroi': counts per ROI
        }
    """
    if sparse is None:
        sparse = should_sparse(xc, yc, rc)

    # Convert inputs to numpy arrays
    xc = np.asarray(xc, dtype=np.float64)
    yc = np.asarray(yc, dtype=np.float64)
    rc = np.asarray(rc, dtype=np.float64)
    evtX = np.asarray(evtX, dtype=np.float64)
    evtY = np.asarray(evtY, dtype=np.float64)

    # Compute centers once
    centers = np.column_stack((xc, yc))
    evt_points = np.column_stack((evtX, evtY))

    # Compute all distances at once
    dsep_matrix = cdist(centers, centers)
    dd = cdist(evt_points, centers)

    # Compute overlap matrix
    rc_sum = rc[:, None] + rc[None, :]
    if sparse:
        # Use triu_indices for efficient upper triangle access
        rows, cols = np.triu_indices(len(xc), k=1)
        mask = dsep_matrix[rows, cols] < rc_sum[rows, cols]

        # Build COO matrix directly for better efficiency
        overlap_matrix = coo_matrix(
            (np.ones(2 * np.sum(mask), dtype=np.uint8),
             (np.concatenate([rows[mask], cols[mask]]),
              np.concatenate([cols[mask], rows[mask]]))),
            shape=(len(xc), len(xc))
        ).tocsr()
    else:
        overlap_matrix = (dsep_matrix < rc_sum).astype(np.uint8)
        np.fill_diagonal(overlap_matrix, 0)

    # Compute event memberships
    evtbyt = (dd <= rc).astype(np.uint8)

    # Find unique ROIs
    uroi, nroi = np.unique(evtbyt, axis=0, return_counts=True)

    return {
        'dsep_matrix': dsep_matrix,
        'overlap_matrix': overlap_matrix,
        'overlap_indices': [overlap_matrix.getrow(i).indices if sparse else
                            np.where(overlap_matrix[i])[0] for i in range(len(xc))],
        'evtbyt': evtbyt,
        'uroi': uroi,
        'nroi': nroi
    }


def overlap_correct(n_overlap_result, sparse=None):
    """
    Corrects missing circle indices in n_overlap() results by adding zero-count ROIs.
    Accepts direct output from n_overlap() and handles missing indices internally.

    Parameters:
        n_overlap_result: Dictionary output from n_overlap()
        sparse: Optional override for sparse format (default: keep original format)

    Returns: Corrected dictionary with same structure as n_overlap()
    """
    # Extract all needed parameters from input dict
    xc = n_overlap_result.get('xc', None)  # Not in original output but may be needed
    yc = n_overlap_result.get('yc', None)
    rc = n_overlap_result.get('rc', None)
    evtX = n_overlap_result.get('evtX', None)
    evtY = n_overlap_result.get('evtY', None)
    dsep_matrix = n_overlap_result['dsep_matrix']
    overlap_matrix = n_overlap_result['overlap_matrix']
    overlap_indices = n_overlap_result['overlap_indices']
    evtbyt = n_overlap_result['evtbyt']
    uroi = n_overlap_result['uroi']
    nroi = n_overlap_result['nroi']

    # Determine sparse mode (use original if not specified)
    if sparse is None:
        sparse = issparse(overlap_matrix)

    # Find missing circle indices (columns with all zeros in uroi)
    mis_indices = np.where(uroi.sum(axis=0) == 0)[0]

    if len(mis_indices) == 0:
        return n_overlap_result  # No correction needed

    # 1. Create new ROI patterns for missing circles
    n_circles = uroi.shape[1]
    new_uroi_rows = np.zeros((len(mis_indices), n_circles), dtype=uroi.dtype)

    for i, idx in enumerate(mis_indices):
        new_uroi_rows[i, idx] = 1  # Pattern: circle exists in isolation

    # 2. Merge with existing uroi
    corrected_uroi = np.vstack([uroi, new_uroi_rows])
    corrected_nroi = np.concatenate([nroi, np.zeros(len(mis_indices), dtype=nroi.dtype)])

    # 3. Maintain original ordering by ROI size then position
    roi_sizes = corrected_uroi.sum(axis=1)
    first_circle = np.argmax(corrected_uroi, axis=1)
    sort_order = np.lexsort((first_circle, roi_sizes))

    # Return corrected result (preserving original sparse/dense format)
    print("corrected for no-signal source")
    return {
        'dsep_matrix': dsep_matrix,
        'overlap_matrix': overlap_matrix.tocsr() if sparse else overlap_matrix,
        'overlap_indices': overlap_indices,
        'evtbyt': evtbyt,
        'uroi': corrected_uroi[sort_order],
        'nroi': corrected_nroi[sort_order],
        # Preserve any additional keys from original
        **{k: v for k, v in n_overlap_result.items()
           if k not in ['dsep_matrix', 'overlap_matrix', 'overlap_indices', 'evtbyt', 'uroi', 'nroi']}
    }

###########################
### Functions for design-matrix construction
### Part 2: area and exposure calculations
### single circles
##########################

def exposure_mc(x_coords, y_coords, emap_path, exposure_bin=32):
    """
    Vectorized exposure calculation with configurable binning factor.

    Parameters:
        x_coords (array-like): X coordinates in exposure_bin*pixel units
        y_coords (array-like): Y coordinates in exposure_bin*pixel units
        emap_path (str): Path to exposure map FITS file
        exposure_bin (float): Binning factor (default: 32)

    Returns:
        numpy.ndarray: Interpolated exposure values
    """
    # Convert from binned to pixel coordinates
    xpix = x_coords / float(exposure_bin)
    ypix = y_coords / float(exposure_bin)

    with fits.open(emap_path) as hdul:
        emap_data = hdul[0].data

    # Vectorized bilinear interpolation
    height, width = emap_data.shape

    # Get integer coordinates
    x1 = np.floor(xpix).astype(int)
    y1 = np.floor(ypix).astype(int)
    x2 = x1 + 1
    y2 = y1 + 1

    # Handle edge cases
    x1 = np.clip(x1, 0, width - 1)
    y1 = np.clip(y1, 0, height - 1)
    x2 = np.clip(x2, 0, width - 1)
    y2 = np.clip(y2, 0, height - 1)

    # Calculate weights
    x_diff = xpix - x1
    y_diff = ypix - y1
    w11 = (1 - x_diff) * (1 - y_diff)
    w21 = x_diff * (1 - y_diff)
    w12 = (1 - x_diff) * y_diff
    w22 = x_diff * y_diff

    # Perform interpolation
    return (emap_data[y1, x1] * w11 +
            emap_data[y1, x2] * w21 +
            emap_data[y2, x1] * w12 +
            emap_data[y2, x2] * w22)


def area_exposure_circle_mc(i, xc, yc, rc, emap_path=None, circMCn=1e5, verbose=False,
                            overlap_data=None, max_attempts=5, exposure_bin=32,
                            find_rmat=False, expected_proportion=0.9, rmat_tol=0.1):
    """
    Monte Carlo calculation for circles with multiple overlaps.
    Modified to:
    - Make emap_path optional (default None)
    - Skip exposure calculation when emap_path is None
    - Still computes areas and (optionally) density proportions

    Returns:
        segments: (n_seg, n_circles) binary arrays
        areas: (n_seg,) physical areas
        exposures: (n_seg,) mean exposures (None if emap_path is None)
        rmat_means: (n_seg,) volume proportions (if find_rmat=True)
    """
    # 1. Setup circle parameters
    x0, y0, r0 = xc[i], yc[i], rc[i]
    overlap_indices = overlap_data['overlap_indices'][i]
    n_circles = len(xc)

    # 2. Generate segments (original logic)
    circles_involved = np.concatenate(([i], overlap_indices))
    segments = []
    for k in range(1, len(circles_involved) + 1):
        for subset in combinations(circles_involved, k):
            if i in subset:
                seg = np.zeros(n_circles, dtype=np.uint8)
                seg[list(subset)] = 1
                segments.append(seg)
    segments = np.unique(np.array(segments), axis=0)
    n_segments = len(segments)

    # 3. Initialize bivariate normal if needed
    if find_rmat:
        chi2_val = chi2.ppf(expected_proportion, df=2)
        cov = np.array([[r0 ** 2 / chi2_val, 0], [0, r0 ** 2 / chi2_val]])
        rv = multivariate_normal(mean=[x0, y0], cov=cov)

    # 4. Adaptive MC sampling
    base_samples = max(1000, int(circMCn))
    total_counts = np.zeros(n_segments)
    all_points = []
    all_masks = []

    for attempt in range(max_attempts):
        n_samples = base_samples * (2 ** attempt)
        points = _generate_points_in_circle(x0, y0, r0, n_samples)

        # Vectorized circle membership test
        dx = points[:, 0, None] - xc
        dy = points[:, 1, None] - yc
        in_circles = (dx ** 2 + dy ** 2) <= (rc ** 2)

        # Vectorized segment matching
        matched = np.all(in_circles[:, None, :] == segments, axis=2)

        all_points.append(points)
        all_masks.append(matched)
        total_counts += matched.sum(axis=0)

        if np.all(total_counts > 0):
            break

    # 5. Combine results
    points = np.concatenate(all_points)
    matched = np.concatenate(all_masks, axis=0)
    total_samples = len(points)

    # 6. Compute areas (always needed)
    counts = matched.sum(axis=0)
    areas = (counts / total_samples) * (pi * r0 ** 2)

    # 7. Compute exposures only if emap_path is provided
    if emap_path is not None:
        exposure_vals = exposure_mc(points[:, 0], points[:, 1], emap_path, exposure_bin)
        valid_mask = ~np.isnan(exposure_vals)
        eroi = np.zeros(n_segments)

        for j in range(n_segments):
            seg_mask = matched[:, j] & valid_mask
            if np.any(seg_mask):
                eroi[j] = np.mean(exposure_vals[seg_mask])
    else:
        eroi = None

    # 8. Compute volume proportions if requested
    if find_rmat:
        rmat_means = np.zeros(n_segments)
        total_volume = 0.0

        for j in range(n_segments):
            seg_mask = matched[:, j]
            seg_points = points[seg_mask]

            if len(seg_points) > 0:
                # Mean density in segment
                mean_density = np.mean(rv.pdf(seg_points))
                # Segment area (already calculated in areas[j])
                # Volume proportion = density × area
                rmat_means[j] = mean_density * areas[j]
                total_volume += rmat_means[j]

        # Validate total volume
        if not np.isclose(total_volume, expected_proportion, atol=rmat_tol):
            warnings.warn(
                f"Total volume proportion {total_volume:.4f} ≠ "
                f"expected {expected_proportion} for circle {i}"
            )

        return segments, areas, eroi, rmat_means

    return segments, areas, eroi

def binary_r(d=None, rc1=1.0, rc2=1.0, circMCn=int(1e3), expected_proportion=0.9,
             overlap_area=None, xc1=0.0, yc1=0.0, xc2=None, yc2=None):
    """Bivariate normal volume calculation for overlapping circles"""
    if d is not None:
        actual_d = d
        xc2, yc2 = xc1 + d, yc1
    else:
        if xc2 is None or yc2 is None:
            raise ValueError("Either (d) or both (xc2,yc2) must be specified")
        actual_d = np.sqrt((xc2 - xc1) ** 2 + (yc2 - yc1) ** 2)

    chi2_val = chi2.ppf(expected_proportion, df=2)
    cov = np.array([[rc1 ** 2 / chi2_val, 0], [0, rc1 ** 2 / chi2_val]])
    rv = multivariate_normal(mean=[xc1, yc1], cov=cov)

    if overlap_area is None:
        if actual_d >= rc1 + rc2 or actual_d <= abs(rc1 - rc2):
            overlap_area = 0.0
        else:
            term1 = rc1 ** 2 * np.arccos((actual_d ** 2 + rc1 ** 2 - rc2 ** 2) / (2 * actual_d * rc1))
            term2 = rc2 ** 2 * np.arccos((actual_d ** 2 + rc2 ** 2 - rc1 ** 2) / (2 * actual_d * rc2))
            term3 = 0.5 * np.sqrt(abs(4 * actual_d ** 2 * rc1 ** 2 - (actual_d ** 2 + rc1 ** 2 - rc2 ** 2) ** 2))
            overlap_area = term1 + term2 - term3

    if overlap_area <= 0:
        return {
            'overlap_proportion': 0.0,
            'non_overlap_proportion': expected_proportion,
            'total_volume': expected_proportion
        }

    center_inside = (actual_d <= rc2)
    if center_inside:
        samples = []
        while len(samples) < circMCn:
            theta = 2 * np.pi * np.random.rand()
            r = rc1 * np.sqrt(np.random.rand())
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            if (x - actual_d) ** 2 + y ** 2 <= rc2 ** 2:
                samples.append([x, y])
        samples = np.array(samples)
    else:
        x_int = (actual_d ** 2 + rc1 ** 2 - rc2 ** 2) / (2 * actual_d)
        y_int = np.sqrt(rc1 ** 2 - x_int ** 2)
        theta1 = np.arctan2(y_int, x_int)
        theta2 = np.arctan2(-y_int, x_int)

        n_half = int(np.ceil(circMCn / 2))
        theta_samples = np.random.uniform(theta2, theta1, size=n_half)
        r_samples = rc1 * np.sqrt(np.random.uniform(0, 1, size=n_half))
        x = r_samples * np.cos(theta_samples)
        y = r_samples * np.sin(theta_samples)

        x_mirror = 2 * x_int - x
        samples = np.vstack([np.column_stack([x, y]),
                             np.column_stack([x_mirror, y])])
        samples = samples[(samples[:, 0] - actual_d) ** 2 + samples[:, 1] ** 2 <= rc2 ** 2]

    if len(samples) == 0:
        overlap_proportion = 0.0
    else:
        densities = rv.pdf(samples)
        overlap_proportion = np.mean(densities) * overlap_area

    return {
        'overlap_proportion': overlap_proportion,
        'non_overlap_proportion': expected_proportion - overlap_proportion,
        'total_volume': expected_proportion
    }


def _generate_points_in_circle(x0, y0, r, n):
    """Generate points uniformly within a circle"""
    theta = np.random.uniform(0, 2 * np.pi, n)
    rad = r * np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack((
        x0 + rad * np.cos(theta),
        y0 + rad * np.sin(theta)))


def area_exposure_single(i, xc, yc, rc, emap_path, circMCn=1e5, verbose=False,
                         overlap_data=None, secondary_emap=None,
                         find_rmat=False, expected_proportion=0.9):
    """
    Process single circle with optional bivariate normal density calculation.
    Minimal changes from original version.
    """
    # Original exposure map handling
    emap_to_use = emap_path
    try:
        with fits.open(emap_path) as f:
            if not isinstance(f[0].data, np.ndarray):
                raise ValueError("Primary exposure map data not found")
    except Exception as e:
        if secondary_emap:
            if verbose:
                print(f"Primary map failed, using secondary: {str(e)}")
            emap_to_use = secondary_emap
        else:
            raise ValueError(f"Could not read exposure map {emap_path}: {str(e)}")

    overlap_data = overlap_data or overlap_correct(n_overlap(xc, yc, rc))
    overlap_indices = overlap_data['overlap_indices'][i]

    # Case 1: No overlaps (minimal change)
    if len(overlap_indices) == 0:
        seg = np.zeros(len(xc), dtype=np.uint8)
        seg[i] = 1
        area = np.pi * rc[i] ** 2
        exposure = _get_exposure(emap_to_use, f"circle({xc[i]},{yc[i]},{rc[i]})", secondary_emap)

        if find_rmat:
            return np.array([seg]), np.array([area]), np.array([exposure]), np.array([expected_proportion])
        else:
            return np.array([seg]), np.array([area]), np.array([exposure])

    # Case 2: Single overlap (minimal change)
    if len(overlap_indices) == 1:
        j = overlap_indices[0]
        d = overlap_data['dsep_matrix'][i, j]
        overlap = circolap(d, rc[i], rc[j])['overlap_area']

        segments = np.zeros((2, len(xc)), dtype=np.uint8)
        segments[0, i] = 1
        segments[1, [i, j]] = 1

        exp_i_only = _get_exposure(
            emap_to_use,
            f"circle({xc[i]},{yc[i]},{rc[i]})-circle({xc[j]},{yc[j]},{rc[j]})",
            secondary_emap
        )
        exp_overlap = _get_exposure(
            emap_to_use,
            f"circle({xc[i]},{yc[i]},{rc[i]})&circle({xc[j]},{yc[j]},{rc[j]})",
            secondary_emap
        )

        areas = np.array([np.pi * rc[i] ** 2 - overlap, overlap])
        exposures = np.array([exp_i_only, exp_overlap])

        if find_rmat:
            res = binary_r(d, rc[i], rc[j], expected_proportion=expected_proportion)
            rmat_means = np.array([
                res['non_overlap_proportion'],
                res['overlap_proportion']
            ])
            return segments, areas, exposures, rmat_means
        else:
            return segments, areas, exposures

    # Case 3: Multiple overlaps (only added find_rmat parameter)
    return area_exposure_circle_mc(
        i, xc, yc, rc, emap_to_use, circMCn, verbose,
        overlap_data, max_attempts=5, exposure_bin=32,
        find_rmat=find_rmat, expected_proportion=expected_proportion
    )

###################
### multiple circles iteration
###################
def _process_single_wrapper(i, xc, yc, rc, emap_path, circMCn, overlap_data,
                          use_secondary=False, secondary_emap=None,
                          find_rmat=False, expected_proportion=0.9):
    """Standalone processing function for parallel execution"""
    current_emap = secondary_emap if use_secondary else emap_path
    return area_exposure_single(
        i, xc, yc, rc, current_emap, circMCn, False, overlap_data,
        secondary_emap, find_rmat, expected_proportion
    )


def area_exposure_multiple(xc, yc, rc, emap_path, overlap_data, circMCn=1e5,
                           parallel=False, empty_cores=0.15, mem_threshold=0.9,
                           secondary_emap=None, find_rmat=False, expected_proportion=0.9,
                           validate_rmat=True, rmat_tol=0.1, faulty_circles_file="faulty_rmat.txt",
                           impossible_txt="impossible_roi.txt", verbose=False):
    """
    Complete ROI analysis pipeline with robust r_mat validation and error recovery.

    Parameters:
        xc, yc, rc: Circle parameters (arrays)
        emap_path: Primary exposure map path
        overlap_data: Precomputed overlap information
        circMCn: Base number of MC samples
        parallel: Enable parallel processing
        empty_cores: Fraction of CPU cores to leave unused
        mem_threshold: Memory usage threshold for fallback to serial
        secondary_emap: Backup exposure map path
        find_rmat: Calculate bivariate normal densities
        expected_proportion: Target proportion for density calculation
        validate_rmat: Verify column sums match expected_proportion
        rmat_tol: Tolerance for column sum validation
        faulty_circles_file: Path to save problematic circle coordinates

    Returns:
        Dictionary containing filtered results with consistent dimensions
    """
    # Input validation
    required_keys = {'uroi', 'overlap_matrix', 'overlap_indices'}
    if not all(k in overlap_data for k in required_keys):
        missing = required_keys - overlap_data.keys()
        raise KeyError(f"Missing keys in overlap_data: {missing}")

    uroi = np.asarray(overlap_data['uroi'], dtype=np.uint8)
    if uroi.ndim != 2:
        raise ValueError("uroi must be 2D array")

    n_roi, n_circles = uroi.shape
    is_sparse = issparse(overlap_data.get('overlap_matrix', None))

    # Initialize matrices
    area_mat = lil_matrix((n_roi, n_circles), dtype=np.float64) if is_sparse else \
        np.zeros((n_roi, n_circles), dtype=np.float64)
    exposure_mat = lil_matrix((n_roi, n_circles), dtype=np.float64) if is_sparse else \
        np.zeros((n_roi, n_circles), dtype=np.float64)
    r_mat = lil_matrix((n_roi, n_circles), dtype=np.float64) if (is_sparse and find_rmat) else \
        np.zeros((n_roi, n_circles), dtype=np.float64) if find_rmat else None

    # Memory check
    mem = psutil.virtual_memory()
    if mem.used / mem.total > mem_threshold:
        print(f"Memory usage {mem.used / mem.total:.1%} > threshold {mem_threshold:.0%} - forcing serial")
        parallel = False

    # Primary processing
    try:
        if parallel:
            max_workers = max(1, int((1 - empty_cores) * os.cpu_count()))
            print(f"Parallel processing with {max_workers} workers")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_single_wrapper,
                        i, xc, yc, rc, emap_path, circMCn, overlap_data,
                        False, secondary_emap, find_rmat, expected_proportion
                    ): i for i in range(n_circles)
                }

                results = []
                for future in tqdm(as_completed(futures), total=n_circles,
                                   desc="Processing area&exposure"):
                    mem = psutil.virtual_memory()
                    if mem.used / mem.total > mem_threshold:
                        raise MemoryError("Memory threshold exceeded")

                    i = futures[future]
                    if find_rmat:
                        segments, areas, exposures, r_means = future.result()
                        results.append((i, segments, areas, exposures, r_means))
                    else:
                        segments, areas, exposures = future.result()
                        results.append((i, segments, areas, exposures))

                # Sort and update matrices
                results.sort()
                for result in results:
                    i = result[0]
                    if find_rmat:
                        segments, areas, exposures, r_means = result[1:]
                        _update_dual_matrices(area_mat, exposure_mat, segments,
                                              areas, exposures, uroi, i, is_sparse)
                        _update_r_matrix(r_mat, segments, r_means, uroi, i, is_sparse)
                    else:
                        segments, areas, exposures = result[1:]
                        _update_dual_matrices(area_mat, exposure_mat, segments,
                                              areas, exposures, uroi, i, is_sparse)
        else:
            for i in tqdm(range(n_circles), desc="Processing area&exposure serially"):
                if find_rmat:
                    segments, areas, exposures, r_means = _process_single_wrapper(
                        i, xc, yc, rc, emap_path, circMCn, overlap_data,
                        False, secondary_emap, find_rmat, expected_proportion
                    )
                    _update_dual_matrices(area_mat, exposure_mat, segments,
                                          areas, exposures, uroi, i, is_sparse)
                    _update_r_matrix(r_mat, segments, r_means, uroi, i, is_sparse)
                else:
                    segments, areas, exposures = _process_single_wrapper(
                        i, xc, yc, rc, emap_path, circMCn, overlap_data
                    )
                    _update_dual_matrices(area_mat, exposure_mat, segments,
                                          areas, exposures, uroi, i, is_sparse)

    except (MemoryError, pickle.PicklingError) as e:
        print(f"Parallel failed ({str(e)}), falling back to serial")
        return _serial_area_exposure(xc, yc, rc, emap_path, overlap_data,
                                     circMCn, is_sparse, secondary_emap,
                                     find_rmat, expected_proportion)
    except Exception as e:
        print(f"Critical error: {traceback.format_exc()}")
        raise

    # Calculate means
    aroi = _calculate_adjusted_means(area_mat, is_sparse)
    eroi = _calculate_adjusted_means(exposure_mat, is_sparse) if exposure_mat is not None else None

    # Secondary exposure map processing
    if secondary_emap and eroi is not None:
        zero_mask = eroi == 0
        zero_indices = np.where(zero_mask)[0]
        impossible_rois = set()
        tangential_pairs = []

        if len(zero_indices) > 0:
            print(f"\nProcessing {len(zero_indices)} zero-exposure ROIs")

            for roi_idx in zero_indices:
                circle_indices = np.where(uroi[roi_idx] == 1)[0]

                # Skip if this is a single-circle ROI
                if len(circle_indices) == 1:
                    if verbose:
                        print(f"Skipping single-circle ROI {roi_idx} (circle {circle_indices[0]})")
                    continue

        if len(zero_indices) > 0:
            print(f"\nProcessing {len(zero_indices)} zero-exposure ROIs with secondary map")

            for roi_idx in zero_indices:
                circle_indices = np.where(uroi[roi_idx] == 1)[0]
                for circ_idx in circle_indices:
                    if find_rmat:
                        segments, areas, exposures, r_means = _process_single_wrapper(
                            circ_idx, xc, yc, rc, emap_path, circMCn, overlap_data,
                            use_secondary=True, secondary_emap=secondary_emap,
                            find_rmat=True, expected_proportion=expected_proportion
                        )
                        matches = np.all(segments[:, None, :] == uroi[None, :, :], axis=-1)
                        for seg_idx in range(matches.shape[0]):
                            matched_rois = np.where(matches[seg_idx])[0]
                            for matched_roi in matched_rois:
                                if matched_roi == roi_idx:
                                    exposure_mat[matched_roi, circ_idx] = exposures[seg_idx]
                                    if find_rmat:
                                        r_mat[matched_roi, circ_idx] = r_means[seg_idx]
                    else:
                        segments, areas, exposures = _process_single_wrapper(
                            circ_idx, xc, yc, rc, emap_path, circMCn, overlap_data,
                            use_secondary=True, secondary_emap=secondary_emap
                        )
                        matches = np.all(segments[:, None, :] == uroi[None, :, :], axis=-1)
                        for seg_idx in range(matches.shape[0]):
                            matched_rois = np.where(matches[seg_idx])[0]
                            for matched_roi in matched_rois:
                                if matched_roi == roi_idx:
                                    exposure_mat[matched_roi, circ_idx] = exposures[seg_idx]

            # Recalculate means
            eroi = _calculate_adjusted_means(exposure_mat, is_sparse)
            zero_mask = eroi == 0
            zero_indices = np.where(zero_mask)[0]

            # Identify impossible ROIs
            for roi_idx in zero_indices:
                if np.all(uroi[roi_idx] == 0):
                    continue

                impossible_rois.add(roi_idx)
                circle_mask = uroi[roi_idx] == 1
                superset_mask = np.all(uroi[:, circle_mask] == 1, axis=1)

                for superset_idx in np.where(superset_mask)[0]:
                    if not np.all(uroi[superset_idx] == 0):
                        impossible_rois.add(superset_idx)

            # Identify tangential pairs
            for roi_idx in impossible_rois:
                circle_indices = np.where(uroi[roi_idx] == 1)[0]
                if len(circle_indices) == 2:
                    tangential_pairs.append((circle_indices[0], circle_indices[1]))

            impossible_rois = np.array(sorted(impossible_rois), dtype=int)

            # After identifying all impossible ROIs:
            if len(impossible_rois) > 0:
                print(f"Saving {len(impossible_rois)} impossible ROIs to {impossible_txt}")
                with open(impossible_txt, 'w') as f:
                    # Write header
                    f.write("roi_index\tnum_circles\tcircle_indices\tx_coords\ty_coords\tradii\tarea\n")

                    # Write each impossible ROI
                    for roi_idx in impossible_rois:
                        circle_indices = np.where(uroi[roi_idx] == 1)[0]
                        f.write(
                            f"{roi_idx}\t"
                            f"{len(circle_indices)}\t"
                            f"{','.join(map(str, circle_indices))}\t"
                            f"{','.join(f'{xc[i]:.2f}' for i in circle_indices)}\t"
                            f"{','.join(f'{yc[i]:.2f}' for i in circle_indices)}\t"
                            f"{','.join(f'{rc[i]:.2f}' for i in circle_indices)}\t"
                            f"{aroi[roi_idx]:.2f}\n"
                        )

    # Create valid ROI mask
    valid_roi_mask = np.ones(n_roi, dtype=bool)
    if 'impossible_rois' in locals() and len(impossible_rois) > 0:
        valid_roi_mask[impossible_rois] = False

    # Filter outputs
    filtered_output = {
        'roi_means': aroi[valid_roi_mask],
        'roi_exposures': eroi[valid_roi_mask] if eroi is not None else None,
        'roi_patterns': uroi[valid_roi_mask],
        'roi_counts': overlap_data.get('nroi', np.zeros(n_roi, dtype=int))[valid_roi_mask],
        'zero_exposure_indices': zero_indices if 'zero_indices' in locals() else np.array([], dtype=int),
        'impossible_rois': impossible_rois if 'impossible_rois' in locals() else np.array([], dtype=int),
        'tangential_pairs': tangential_pairs if 'tangential_pairs' in locals() else []
    }

    # Handle r_mat with validation
    if find_rmat:
        # Convert to dense for consistent filtering
        r_mat_dense = r_mat.toarray() if issparse(r_mat) else r_mat
        filtered_output['r_mat'] = r_mat_dense[valid_roi_mask]

        # Validate column sums
        if validate_rmat:
            col_sums = filtered_output['r_mat'].sum(axis=0)
            deviations = np.abs(col_sums - expected_proportion)
            problematic = np.where(deviations > rmat_tol)[0]

            if len(problematic) > 0:
                # Initialize faulty circles file
                with open(faulty_circles_file, 'w') as f:
                    f.write("index\tx\ty\tradius\tdeviation\n")

                print("\nProblematic circles with r_mat deviations:")
                print(f"{'Index':<8}{'X':<12}{'Y':<12}{'Radius':<12}{'Deviation':<12}")
                for idx in problematic:
                    dev = deviations[idx]
                    print(f"{idx:<8}{xc[idx]:<12.2f}{yc[idx]:<12.2f}{rc[idx]:<12.2f}{dev:<12.2e}")
                    with open(faulty_circles_file, 'a') as f:
                        f.write(f"{idx}\t{xc[idx]}\t{yc[idx]}\t{rc[idx]}\t{dev}\n")

                # Attempt recovery
                for idx in problematic:
                    print(f"\nRecomputing circle {idx} at ({xc[idx]:.2f}, {yc[idx]:.2f})")
                    result = rmat_single_circle_mc(
                        idx, xc, yc, rc, circMCn * 2, overlap_data,
                        expected_proportion, tangential_pairs,
                        impossible_rois, verbose=verbose
                    )

                    if np.isclose(result['proportions'].sum(), expected_proportion, atol=rmat_tol):
                        # Update successful recomputations
                        for seg_idx, segment in enumerate(result['segments']):
                            roi_matches = np.all(filtered_output['roi_patterns'] == segment, axis=1)
                            filtered_output['r_mat'][roi_matches, idx] = result['proportions'][seg_idx]
                        print(f"Successfully recomputed circle {idx}")
                    else:
                        # Fallback to expected_proportion for non-overlapping segment
                        warnings.warn(
                            f"Circle {idx} at ({xc[idx]:.2f}, {yc[idx]:.2f}) failed recomputation. "
                            f"Assigning {expected_proportion} to non-overlapping segment."
                        )
                        non_overlap_segment = np.zeros(n_circles, dtype=np.uint8)
                        non_overlap_segment[idx] = 1
                        roi_matches = np.all(
                            filtered_output['roi_patterns'] == non_overlap_segment,
                            axis=1
                        )
                        filtered_output['r_mat'][roi_matches, idx] = expected_proportion

    return filtered_output

def _update_r_matrix(r_mat, segments, r_means, uroi, circle_idx, is_sparse):
    """Helper to update density matrix with new values for a circle"""
    if r_mat is None:
        return

    for seg_idx, segment in enumerate(segments):
        matches = np.all(uroi == segment, axis=1)
        matched_rois = np.where(matches)[0]

        for roi_idx in matched_rois:
            if is_sparse:
                r_mat[roi_idx, circle_idx] = r_means[seg_idx]
            else:
                r_mat[roi_idx, circle_idx] = r_means[seg_idx]

def _get_exposure(emap_path, region, fallback_emap=None):
    """Get exposure value with fallback to secondary map"""
    try:
        dmstat(f"{emap_path}[sky={region}]", centroid=False, sigma=False)
        result = float(dmstat.out_mean)
        if np.isnan(result) and fallback_emap:
            print(f"Primary map returned NaN, trying fallback for {region}")
            dmstat(f"{fallback_emap}[sky={region}]", centroid=False, sigma=False)
            return float(dmstat.out_mean)
        return result
    except Exception as e:
        print(f"WARNING: Exposure calculation failed for region {region}: {str(e)}")
        return float('nan')


# Calculate adjusted means with new factor (1/n_nonzeros)
def _calculate_adjusted_means(mat, is_sparse):
    """Calculate adjusted means with (1/n_nonzeros) factor"""
    if is_sparse:
        n_nonzeros = np.diff(mat.tocsr().indptr)
        row_means = np.array(mat.sum(axis=1)).flatten()
    else:
        n_nonzeros = np.count_nonzero(mat, axis=1)
        row_means = np.sum(mat, axis=1)

    # Handle division by zero
    valid = n_nonzeros > 0
    adjusted_means = np.zeros_like(row_means)
    adjusted_means[valid] = row_means[valid] * (1 / n_nonzeros[valid])
    return adjusted_means

def _update_dual_matrices(area_mat, exp_mat, segments, areas, exposures, uroi, circle_idx, is_sparse):
    """Update both matrices with segment data"""
    if len(segments) == 0:
        return

    matches = np.all(segments[:, None, :] == uroi[None, :, :], axis=-1)
    for seg_idx in range(matches.shape[0]):
        matched_rois = np.where(matches[seg_idx])[0]
        if len(matched_rois) > 0:
            if is_sparse:
                for roi_idx in matched_rois:
                    area_mat[roi_idx, circle_idx] = areas[seg_idx]
                    exp_mat[roi_idx, circle_idx] = exposures[seg_idx]
            else:
                area_mat[matched_rois, circle_idx] = areas[seg_idx]
                exp_mat[matched_rois, circle_idx] = exposures[seg_idx]


def _serial_area_exposure(xc, yc, rc, emap_path, overlap_data, circMCn, is_sparse, secondary_emap=None):
    """Fallback serial implementation"""
    uroi = np.asarray(overlap_data['uroi'])
    n_roi, n_circles = uroi.shape

    area_mat = lil_matrix((n_roi, n_circles), dtype=np.float64) if is_sparse else \
        np.zeros((n_roi, n_circles), dtype=np.float64)
    exposure_mat = lil_matrix((n_roi, n_circles), dtype=np.float64) if is_sparse else \
        np.zeros((n_roi, n_circles), dtype=np.float64)

    for i in range(n_circles):
        mem = psutil.virtual_memory()
        if mem.used / mem.total > 0.95:
            raise MemoryError("Memory critically high")

        segments, areas, exposures = area_exposure_single(
            i, xc, yc, rc, emap_path, circMCn, False, overlap_data, secondary_emap
        )
        _update_dual_matrices(area_mat, exposure_mat,
                              segments, areas, exposures, uroi, i, is_sparse)

    # Calculate means
    aroi = _calculate_adjusted_means(area_mat, is_sparse)
    eroi = _calculate_adjusted_means(exposure_mat, is_sparse) if exposure_mat is not None else None

    return {
        'roi_means': aroi,
        'roi_exposures': eroi,
        'roi_patterns': uroi,
        'roi_counts': overlap_data.get('nroi', np.zeros(uroi.shape[0], dtype=int))
    }

###########################
### Functions for design-matrix construction
### Part 3: rmat calculation
### single circles
##########################
"""
    Optimized MC estimation for r_mat entries with special handling for:
    - Nearly-tangential circle pairs
    - Impossible ROIs (zero exposure after both primary and secondary maps)

    Parameters:
        i (int): Target circle index
        xc, yc, rc (np.ndarray): Circle centers and radii
        circMCn (int): Initial MC samples
        overlap_data (dict): Precomputed overlaps from overlap_correct(n_overlap())
        expected_proportion (float): Target proportion for convergence
        tangential_pairs (list): List of (roi_idx, c1, c2, dist, rsum) tuples
        impossible_rois (np.ndarray): Indices of ROIs with zero exposure after both maps

    Returns:
        dict: {
            'segments': array of segment vectors,
            'proportions': array of proportions,
            'n_samples': total points generated,
            'status': convergence message
        }
    """

def rmat_single_circle_mc(i, xc, yc, rc, circMCn=1e5, overlap_data=None,
                          expected_proportion=0.9, tangential_pairs=None,
                          impossible_rois=None, verbose=False):
    """
    Robust MC estimation for r_mat entries with fixed tuple unpacking.
    """
    # Input validation
    if not (0 <= i < len(xc)):
        raise ValueError(f"Circle index {i} out of range [0, {len(xc)})")
    if len({len(xc), len(yc), len(rc)}) != 1:
        raise ValueError("xc, yc, rc must have equal lengths")
    if not (0 < expected_proportion <= 1):
        raise ValueError("expected_proportion must be in (0, 1]")

    # Initialize default return values
    default_segment = np.zeros(len(xc), dtype=np.uint8)
    default_segment[i] = 1
    default_return = {
        'segments': np.array([default_segment]),
        'proportions': np.array([expected_proportion]),
        'n_samples': 0,
        'status': 'Initialization'
    }

    # Get sparse overlap data
    if overlap_data is None:
        overlap_data = overlap_correct(n_overlap(xc, yc, rc, sparse=True), sparse=True)
    overlap_indices = overlap_data['overlap_indices'][i]
    uroi = overlap_data.get('uroi', None)

    # Handle non-overlapping case
    if len(overlap_indices) == 0:
        default_return['status'] = f'Non-overlapping circle (assigned {expected_proportion})'
        return default_return

    # Initialize variables
    if impossible_rois is None:
        impossible_rois = np.array([], dtype=int)
    if tangential_pairs is None:
        tangential_pairs = []

    # Generate expected segments
    circles_involved = np.concatenate(([i], overlap_indices))
    expected_segments = []

    for k in range(1, len(circles_involved) + 1):
        for subset in combinations(circles_involved, k):
            if i in subset:
                seg = np.zeros(len(xc), dtype=np.uint8)
                seg[np.array(subset)] = 1
                if uroi is not None and len(impossible_rois) > 0:
                    if not np.any(np.all(uroi[impossible_rois] == seg, axis=1)):
                        expected_segments.append(seg)
                else:
                    expected_segments.append(seg)

    # Handle tangential pairs - fixed unpacking
    if tangential_pairs:
        filtered_segments = []
        current_pairs = [pair for pair in tangential_pairs if i in (pair[0], pair[1])]  # Now expects (c1, c2) pairs

        for seg in expected_segments:
            include = True
            for c1, c2 in current_pairs:  # Now unpacking just circle indices
                if seg[c1] == 1 and seg[c2] == 1:
                    if uroi is not None:
                        # Find if this combination matches any impossible ROI
                        match_mask = np.all(uroi == seg, axis=1)
                        if np.any(match_mask) and np.any(np.isin(np.where(match_mask)[0], impossible_rois)):
                            include = False
                            break
            if include:
                filtered_segments.append(seg)
        expected_segments = filtered_segments

    expected_segments = np.unique(expected_segments, axis=0)

    if len(expected_segments) == 0:
        default_return['status'] = 'Only non-impossible segment is the circle itself'
        return default_return

    # MC sampling setup
    chi2_val = chi2.ppf(expected_proportion, df=2)
    cov = np.array([
        [rc[i] ** 2 / chi2_val, 0],
        [0, rc[i] ** 2 / chi2_val]
    ])
    total_proportion = 0.0
    circMCn_current = max(1000, int(circMCn))
    status_msgs = []
    final_proportions = np.zeros(len(expected_segments))

    for iteration in range(5):  # Reduced to 5 iterations
        try:
            points = multivariate_normal.rvs(
                mean=[xc[i], yc[i]],
                cov=cov,
                size=circMCn_current
            )

            # Vectorized classification
            diff = points[:, None] - np.column_stack((xc, yc))
            in_circles = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2) <= rc ** 2
            unique_segments = np.unique(in_circles, axis=0)

            # Check for missing segments
            missing = 0
            for seg in expected_segments:
                if not np.any(np.all(unique_segments == seg, axis=1)):
                    missing += 1

            if missing > 0:
                reason = f"Missing {missing} segment types"
                status_msgs.append(f"Iter {iteration}: {circMCn_current} samples - {reason}")
                if verbose:
                    print(status_msgs[-1])
                circMCn_current *= 2
                continue

            # Calculate proportions
            matched = np.all(in_circles[:, None] == expected_segments, axis=2)
            counts = np.sum(matched, axis=0)
            final_proportions = counts / circMCn_current
            total_proportion = np.sum(final_proportions)

            # Check convergence
            tolerance = 0.05
            if ((expected_proportion - tolerance) <= total_proportion <=
                    (expected_proportion + tolerance)):
                return {
                    'segments': expected_segments,
                    'proportions': final_proportions,
                    'n_samples': circMCn_current,
                    'status': " | ".join(status_msgs + [f"Converged after {iteration + 1} iterations"])
                }

            reason = f"Proportion {total_proportion:.4f} outside target range"
            status_msgs.append(f"Iter {iteration}: {circMCn_current} samples - {reason}")
            if verbose:
                print(status_msgs[-1])
            circMCn_current *= 2

        except Exception as e:
            status_msgs.append(f"Iter {iteration} failed: {str(e)}")
            if verbose:
                print(status_msgs[-1])
            circMCn_current *= 2
            continue

    final_status = " | ".join(status_msgs + [
        f"Final: {total_proportion:.4f} (target: {expected_proportion})"
    ])
    warnings.warn(
        f"Circle {i} at ({xc[i]:.2f}, {yc[i]:.2f}) did not converge\n{final_status}"
    )
    return {
        'segments': expected_segments,
        'proportions': final_proportions,
        'n_samples': circMCn_current,
        'status': final_status
    }

#############
### multiple circles iteration
#############
def _process_circle_wrapper(i, xc, yc, rc, circMCn, overlap_data, expected_proportion,
                            tangential_pairs=None, impossible_rois=None, verbose=True):
    """Standalone processing function for parallel execution with tangential pair handling"""
    return rmat_single_circle_mc(
        i, xc, yc, rc,
        circMCn=circMCn,
        overlap_data=overlap_data,
        expected_proportion=expected_proportion,
        tangential_pairs=tangential_pairs,
        impossible_rois=impossible_rois,
        verbose=verbose
    )


def rmat_multiple_mc(xc, yc, rc, overlap_data, circMCn=1e5, expected_proportion=0.9,
                     parallel=False, empty_cores=0.15, mem_threshold=0.9,
                     tangential_pairs=None, impossible_rois=None, verbose=True):
    """
    Compute density proportion matrix with special handling for:
    - Nearly-tangential circle pairs
    - Impossible ROIs (zero exposure after both primary and secondary maps)

    Parameters:
        xc, yc, rc: Circle parameters
        overlap_data: Must contain valid 'uroi' matrix
        circMCn: MC samples for complex cases
        expected_proportion: Target proportion
        parallel: Enable parallel processing
        empty_cores: Fraction of unused cores
        mem_threshold: Memory threshold
        tangential_pairs: List of (roi_idx, c1, c2, dist, rsum) tuples
        impossible_rois: Array of ROI indices with zero exposure after both maps

    Returns:
        csr_matrix or ndarray: r_mat matrix
    """
    # 1. Input validation
    required_keys = {'uroi', 'overlap_matrix', 'overlap_indices'}
    if not all(k in overlap_data for k in required_keys):
        missing = required_keys - overlap_data.keys()
        raise KeyError(f"Missing keys: {missing}")

    uroi = np.asarray(overlap_data['uroi'])
    if uroi.ndim != 2:
        raise ValueError("uroi must be 2D array")

    # Initialize tangential pairs and impossible ROIs if not provided
    if tangential_pairs is None:
        tangential_pairs = []
    if impossible_rois is None:
        impossible_rois = np.array([], dtype=int)

    n_roi, n_circles = uroi.shape
    is_sparse = issparse(overlap_data['overlap_matrix'])

    # 2. Initialize r_mat with special handling for impossible ROIs
    r_mat = lil_matrix((n_roi, n_circles), dtype=np.float64) if is_sparse else \
        np.zeros((n_roi, n_circles), dtype=np.float64)

    # Set zero for impossible ROIs
    for roi_idx in impossible_rois:
        if roi_idx < n_roi:
            circle_indices = np.where(uroi[roi_idx] == 1)[0]
            r_mat[roi_idx, circle_indices] = 0

    # 3. Memory check
    mem = psutil.virtual_memory()
    if mem.used / mem.total > mem_threshold:
        warnings.warn(f"Memory threshold exceeded - forcing serial")
        parallel = False

    # 4. Create circle processing groups (tangential vs normal)
    circle_groups = {}
    for i in range(n_circles):
        # Check if circle is part of any tangential pair
        is_tangential = any(i in (pair[1], pair[2]) for pair in tangential_pairs)
        group_key = 'tangential' if is_tangential else 'normal'
        circle_groups.setdefault(group_key, []).append(i)

    # 5. Process circles with appropriate parameters
    all_segments = []
    all_proportions = []
    circle_indices = []

    def process_circle_batch(circle_list, is_tangential=False, verbose=True):
        """Helper function to process a batch of circles"""
        batch_segments = []
        batch_proportions = []
        batch_indices = []

        current_tangential_pairs = tangential_pairs if is_tangential else []
        current_impossible_rois = impossible_rois if is_tangential else np.array([], dtype=int)

        if parallel:
            with ProcessPoolExecutor(max_workers=max(1, int((1 - empty_cores) * os.cpu_count()))) as executor:
                futures = {
                    executor.submit(
                        _process_circle_wrapper,
                        i, xc, yc, rc, circMCn, overlap_data, expected_proportion,
                        current_tangential_pairs, current_impossible_rois, verbose
                    ): i for i in circle_list
                }

                for future in tqdm(as_completed(futures), total=len(circle_list),
                                   desc=f"Processing {'tangential' if is_tangential else 'normal'} circles"):
                    res = future.result()
                    batch_segments.append(res['segments'])
                    batch_proportions.append(res['proportions'])
                    batch_indices.extend([res['circle_index']] * len(res['segments']))
        else:
            for i in tqdm(circle_list, desc=f"Processing {'tangential' if is_tangential else 'normal'} circles"):
                res = _process_circle_wrapper(
                    i, xc, yc, rc, circMCn, overlap_data, expected_proportion,
                    current_tangential_pairs, current_impossible_rois, verbose
                )
                batch_segments.append(res['segments'])
                batch_proportions.append(res['proportions'])
                batch_indices.extend([i] * len(res['segments']))

        return batch_segments, batch_proportions, batch_indices

    # Process normal circles first
    if 'normal' in circle_groups:
        seg, prop, idx = process_circle_batch(circle_groups['normal'], False, verbose=verbose)
        all_segments.extend(seg)
        all_proportions.extend(prop)
        circle_indices.extend(idx)

    # Then process tangential circles with special handling
    if 'tangential' in circle_groups:
        seg, prop, idx = process_circle_batch(circle_groups['tangential'], True, verbose=verbose)
        all_segments.extend(seg)
        all_proportions.extend(prop)
        circle_indices.extend(idx)

    # 6. Vectorized ROI matching with dimension safety
    for seg, prop, i in zip(np.concatenate(all_segments),
                            np.concatenate(all_proportions),
                            circle_indices):
        # Ensure consistent dimensions
        seg = np.squeeze(seg)
        if seg.ndim == 0:
            seg = seg[np.newaxis]

        if seg.shape[0] != uroi.shape[1]:
            seg = seg[:uroi.shape[1]]  # Truncate if necessary

        matches = np.all(uroi == seg[np.newaxis, :], axis=1)
        if np.any(matches):
            if is_sparse:
                for row in np.where(matches)[0]:
                    # Only update if not an impossible ROI
                    if row not in impossible_rois:
                        r_mat[row, i] = prop
            else:
                valid_rows = np.where(matches)[0]
                valid_rows = valid_rows[~np.isin(valid_rows, impossible_rois)]
                r_mat[valid_rows, i] = prop

    # 7. Enhanced validity check
    col_sums = np.array(r_mat.sum(axis=0)).flatten()
    valid_cols = ~np.isin(np.arange(n_circles), [p[1] for p in tangential_pairs] + [p[2] for p in tangential_pairs])

    bad_cols = valid_cols & ~np.isclose(col_sums, expected_proportion, atol=0.01)
    if np.any(bad_cols):
        warnings.warn(f"{bad_cols.sum()} normal columns deviate from expected proportion")

    # Check tangential columns separately
    tang_cols = np.isin(np.arange(n_circles), [p[1] for p in tangential_pairs] + [p[2] for p in tangential_pairs])
    if np.any(tang_cols):
        tang_sums = col_sums[tang_cols]
        print(f"\nTangential circle proportions (expected ~{expected_proportion:.2f}):")
        print(f"Min: {np.min(tang_sums):.4f}, Max: {np.max(tang_sums):.4f}, Mean: {np.mean(tang_sums):.4f}")

    return r_mat.tocsr() if is_sparse else r_mat

###########################
### Functions for design-matrix construction
### Part 4: background calculations
### single background
##########################
def background_area(xc, yc, rc, aroi, theta, centre_x, centre_y, verbose=False):
    """
    Calculate background area using circular region based on maximum off-axis angle.

    Parameters:
        evtX, evtY: Arrays of event coordinates
        aroi: Array of segment areas (background will be aroi[0])
        theta: Array of off-axis angles for each circle
        find_centre: Function that returns (x0, y0) center coordinates
        verbose: Print debug info if True

    Returns:
        tuple: (updated_aroi, (x0, y0, R_big), None)
               (last None maintains compatibility with original return signature)
    """
    # Find the circle with maximum off-axis angle
    imax = np.argmax(theta)
    max_theta = theta[imax]
    xfar, yfar, rfar = xc[imax], yc[imax], rc[imax]

    # Get center of big circle
    x0, y0 = centre_x, centre_y

    # Calculate distance from farthest circle to center
    dfar = np.sqrt((xfar - x0) ** 2 + (yfar - y0) ** 2)

    # Radius of big circle
    R_big = dfar + rfar
    print("xfar:\n")
    print(xfar)
    print("yfar:\n")
    print(yfar)
    print("rfar:\n")
    print(rfar)
    # Calculate circular area
    C_area = np.pi * R_big ** 2

    # Update background area (aroi[0])
    if len(aroi) > 1:
        aroi[0] = max(0, C_area - np.sum(aroi[1:]))
    else:
        aroi[0] = C_area

    if verbose:
        print(f"Big circle: center=({x0:.2f}, {y0:.2f}), R={R_big:.2f}")
        print(f"Background area: {aroi[0]:.2f} (total {C_area:.2f} - sources {np.sum(aroi[1:]):.2f})")

    return aroi, R_big, max_theta


def background_count(R_big, centre_x, centre_y, evtX, evtY, nroi):
    """
    Count photons in the big circular background region and update nroi[0].

    Parameters:
        R_big (float): Radius of the big circle
        centre_x, centre_y (float): Center coordinates of big circle
        evtX, evtY (arrays): Photon coordinates
        nroi (array): Array of counts per segment (background is nroi[0])

    Returns:
        array: Updated nroi with background count corrected
    """
    # Calculate distances from all photons to center
    distances = np.sqrt((evtX - centre_x) ** 2 + (evtY - centre_y) ** 2)

    # Count photons inside big circle
    circular_count = np.sum(distances <= R_big)

    # Update background count (nroi[0])
    if len(nroi) > 1:
        nroi[0] = max(0, circular_count - np.sum(nroi[1:]))
    else:
        nroi[0] = circular_count

    return nroi

def minimum_area_rectangle(points):
    """Vectorized rotating calipers implementation."""
    edges = points - np.roll(points, 1, axis=0)
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.unique(np.mod(angles, np.pi / 2))

    # Vectorized rotation matrix calculation
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    rotation_matrices = np.array([[cos_a, -sin_a], [sin_a, cos_a]]).transpose(2, 0, 1)

    # Vectorized bounding box calculation
    rot_points = points @ rotation_matrices
    min_xy = rot_points.min(axis=0)
    max_xy = rot_points.max(axis=0)
    areas = (max_xy[:, 0] - min_xy[:, 0]) * (max_xy[:, 1] - min_xy[:, 1])

    # Find optimal rotation
    best_idx = np.argmin(areas)
    R = rotation_matrices[best_idx]
    corners = np.array([[min_xy[best_idx, 0], min_xy[best_idx, 1]],
                        [max_xy[best_idx, 0], min_xy[best_idx, 1]],
                        [max_xy[best_idx, 0], max_xy[best_idx, 1]],
                        [min_xy[best_idx, 0], max_xy[best_idx, 1]]])

    # Rotate back and order points
    ordered = corners @ R.T
    centroid = ordered.mean(axis=0)
    return ordered[np.argsort(np.arctan2(ordered[:, 1] - centroid[1], ordered[:, 0] - centroid[0]))]

#########################
### multiple background
#########################
def stratified_bkgd_belong(theta_thresh, theta):
    """
    Stratify theta values into K categories based on threshold boundaries.

    Parameters:
        theta_thresh: Array of K-1 thresholds [b1, b2, ..., b(K-1)]
                     that define category boundaries
        theta: Array of theta values to be classified

    Returns:
        numpy.ndarray: Integer array (1 to K) indicating category membership
    """
    # Convert inputs to numpy arrays if they aren't already
    theta_thresh = np.asarray(theta_thresh)
    theta = np.asarray(theta)

    # Add -inf and +inf to thresholds for edge cases
    full_thresh = np.concatenate([[-np.inf], theta_thresh, [np.inf]])

    # Digitize the theta values into bins
    category = np.digitize(theta, full_thresh)

    # np.digitize returns 1-based indexing (1 to K)
    return category


###########################
### Functions for design-matrix construction
### Part 5: main function filter_evt_by_roigroup
##########################

"""
    Compute ROI statistics with optional parallel MC sampling.

    Parameters:
        evtX, evtY: Event coordinates (np.ndarray)
        xc, yc, rc: Circle parameters (np.ndarray)
        computePropArea: Whether to compute areas/proportions (bool)
        circMCn: MC samples for density proportions (int)
        expected_proportion: Target proportion for convergence (default: 0.9)
        verbose: Print progress info (bool)
        sparse: Sparse mode (bool/None for auto)
        parallel: Enable parallel MC (bool)

    Returns:
        dict: {
            'evtbyt': Event membership matrix,
            'uroi': Unique ROI patterns,
            'broi': Transposed uroi,
            'nroi': Counts per ROI,
            'aroi': ROI and background areas,
            'ncir': Counts per circle,
            'ntot': Total events,
            'r_mat': Density proportion matrix
            'bkg_rate': background event rate (nroi[0]/aroi[0]),
            'bkg_vertices': background rectangle vertices,
            'bkg_angles': rectangle corner angles
        }
    """


def filter_evt_by_roigroup(evtX, evtY, xc, yc, rc, theta, centre_x, centre_y, emap_path=None,
                           computePropArea=True, circMCn=1e3,
                           expected_proportion=0.9, verbose=False,
                           sparse=None, parallel=False,
                           empty_cores=0.15, mem_threshold=0.85,
                           secondary_emap=None, density_rmat=True):
    """
        Compute ROI statistics with optional parallel MC sampling.

        Parameters:
            evtX, evtY: Event coordinates (np.ndarray)
            xc, yc, rc: Circle parameters (np.ndarray)
            computePropArea: Whether to compute areas/proportions (bool)
            circMCn: MC samples for density proportions (int)
            expected_proportion: Target proportion for convergence (default: 0.9)
            verbose: Print progress info (bool)
            sparse: Sparse mode (bool/None for auto)
            parallel: Enable parallel MC (bool)

        Returns:
            dict: {
                'evtbyt': Event membership matrix,
                'uroi': Unique ROI patterns,
                'broi': Transposed uroi,
                'nroi': Counts per ROI,
                'aroi': ROI and background areas,
                'ncir': Counts per circle,
                'ntot': Total events,
                'r_mat': Density proportion matrix
                'bkg_rate': background event rate (nroi[0]/aroi[0]),
                'bkg_vertices': background rectangle vertices,
                'bkg_angles': rectangle corner angles
            }
        """
    # 1. Determine sparse mode
    if sparse is None:
        sparse = should_sparse(xc, yc, rc, verbose=verbose)

    # 2. Compute event overlaps
    overlap_data = overlap_correct(n_overlap(xc, yc, rc, evtX, evtY, sparse=sparse), sparse=sparse)
    if verbose:
        print(f"Initial ROI count: {overlap_data['uroi'].shape[0]}")

    # 3. Initialize result dictionary
    result = {
        'evtbyt': overlap_data['evtbyt'],
        'uroi': overlap_data['uroi'],
        'broi': overlap_data['uroi'].T,
        'nroi': overlap_data['nroi'],
        'ncir': np.sum(overlap_data['evtbyt'], axis=0),
        'ntot': len(evtX),
        'aroi': None,
        'eroi': None,
        'r_mat': None,
        'bkg_rate': None,
        'bkg_radius': None,
        'max_theta': None,
        'tangential_pairs': [],
        'impossible_rois': np.array([], dtype=int)
    }

    # 4. Area and exposure calculations
    if computePropArea:
        area_exp_result = area_exposure_multiple(
            xc, yc, rc, emap_path, overlap_data, circMCn,
            parallel=parallel,
            empty_cores=empty_cores,
            mem_threshold=mem_threshold,
            secondary_emap=secondary_emap,
            find_rmat=density_rmat,
            expected_proportion=expected_proportion
        )

        # Directly use pre-filtered results from area_exposure_multiple
        result.update({
            'impossible_rois': area_exp_result.get('impossible_rois', np.array([], dtype=int)),
            'tangential_pairs': area_exp_result.get('tangential_pairs', []),
            'aroi': area_exp_result['roi_means'],
            'eroi': area_exp_result['roi_exposures'],
        })

        # Handle r_mat based on calculation method
        if density_rmat:
            result['r_mat'] = area_exp_result.get('r_mat', None)
        else:
            result['r_mat'] = rmat_multiple_mc(
                xc, yc, rc, overlap_data, circMCn,
                expected_proportion=expected_proportion,
                parallel=parallel,
                empty_cores=empty_cores,
                mem_threshold=mem_threshold,
                tangential_pairs=result['tangential_pairs'],
                impossible_rois=result['impossible_rois'],
                verbose=verbose
            )

        # Filter uroi, broi, nroi to match the pre-filtered aroi/eroi dimensions
        n_filtered_rois = len(result['aroi'])
        if n_filtered_rois < len(result['uroi']):
            if verbose:
                print(f"Filtering {len(result['uroi']) - n_filtered_rois} impossible ROIs from pattern matrices")

            # Create mask excluding impossible ROIs
            valid_mask = np.ones(len(result['uroi']), dtype=bool)
            valid_mask[result['impossible_rois']] = False

            result.update({
                'uroi': result['uroi'][valid_mask],
                'broi': result['uroi'][valid_mask].T,
                'nroi': result['nroi'][valid_mask]
            })

        # Compute background on filtered ROIs
        if len(result['aroi']) > 1 and len(result['nroi']) > 0:
            bg_result = background_area(xc, yc, rc, result['aroi'].copy(), theta, centre_x, centre_y, verbose)
            if bg_result is not None:
                result['aroi'], result['bkg_radius'], result['max_theta'] = bg_result
                result['nroi'] = background_count(result['bkg_radius'], centre_x, centre_y, evtX, evtY, result['nroi'])
                if result['aroi'][0] > 0:
                    result['bkg_rate'] = result['nroi'][0] / result['aroi'][0]

    # 5. Convert to sparse if requested
    if sparse:
        from scipy.sparse import csr_matrix
        result['evtbyt'] = csr_matrix(result['evtbyt'])
        if result['r_mat'] is not None:
            result['r_mat'] = csr_matrix(result['r_mat'])

    # 6. Final validation
    n_roi = len(result['aroi']) if result['aroi'] is not None else 0
    if n_roi > 0:
        assert len(result['nroi']) == n_roi, "Dimension mismatch in nroi"
        assert len(result['uroi']) == n_roi, "Dimension mismatch in uroi"
        if result['r_mat'] is not None:
            assert result['r_mat'].shape[0] == n_roi, "Dimension mismatch in r_mat"

    if verbose:
        print("\nFinal Statistics:")
        print(f"- Total ROIs: {n_roi}")
        print(f"- Excluded impossible ROIs: {len(result['impossible_rois'])}")
        if len(result['tangential_pairs']) > 0:
            print(f"- Nearly-tangential circle pairs: {len(result['tangential_pairs'])}")

    return result


def get_exposure_with_jitter(x, y, emap_path, secondary_emap=None, max_tries=5, jitter_arcsec=0.5):
    """
    Helper function to get exposure with jittering fallback.
    """
    from astropy.io import fits
    from astropy.wcs import WCS

    for attempt in range(max_tries):
        if attempt == 0:
            current_x, current_y = x, y
        else:
            angle = np.random.uniform(0, 2 * np.pi)
            offset = jitter_arcsec / 0.1318  # Convert arcsec to pixels
            current_x = x + offset * np.cos(angle)
            current_y = y + offset * np.sin(angle)

        try:
            with fits.open(emap_path) as hdul:
                wcs = WCS(hdul[0].header)
                ra, dec = wcs.all_pix2world([[current_x, current_y]], 0)[0]
            region = f"circle({ra}d,{dec}d,0)"
            exposure = _get_exposure(emap_path, region, secondary_emap)

            if exposure > 0 or not np.isnan(exposure):
                return exposure
        except Exception as e:
            print(f"Jitter attempt {attempt + 1} failed: {str(e)}")
            continue

    print(f"WARNING: All jitter attempts failed for position ({x},{y})")
    return 0.0

def which_miss(filter_result):
    """
    Identify missed indices by checking for zero column sums in r_mat.

    Parameters:
        filter_result (dict): Result dictionary from filter_evt_by_roigroup()

    Returns:
        numpy.ndarray: Array of missed indices (empty if none found)
    """
    if filter_result['r_mat'] is None:
        print("No r_mat present in results")
        return np.array([], dtype=int)

    # Calculate column sums and find zeros
    col_sums = np.sum(filter_result['r_mat'], axis=0)
    zero_cols = np.where(col_sums == 0)[0]

    if len(zero_cols) == 0:
        print("no missed indices")
        return np.array([], dtype=int)
    else:
        print(f"Found missed indices: {zero_cols}")
        return zero_cols


def missed_area(xc, yc, rc, missed_indices, total_circles, emap_path, secondary_emap=None,
                find_rmat=False, expected_proportion=0.9, validate_rmat=True):
    """
    Simplified area/exposure calculation for non-overlapping missed circles.

    Parameters:
        xc, yc, rc: Arrays for ALL circles (not just missed ones)
        missed_indices: Indices of the missed circles in xc/yc/rc
        total_circles: Total number of circles (for r_mat dimensions)
        emap_path: Primary exposure map path
        secondary_emap: Backup exposure map path
        find_rmat: Whether to compute r_mat
        expected_proportion: Target proportion for r_mat
        validate_rmat: Whether to validate r_mat sums

    Returns:
        Dictionary with areas, exposures, and properly constructed r_mat
    """
    n_missed = len(missed_indices)

    # Calculate exact areas (πr²) for missed circles
    areas = np.array([np.pi * rc[i] ** 2 for i in missed_indices])

    # Initialize exposure array
    exposures = np.zeros(n_missed)

    # Get exposures with jittering fallback for missed circles
    for idx, i in enumerate(missed_indices):
        exposures[idx] = get_exposure_with_jitter(xc[i], yc[i], emap_path, secondary_emap)

    # Create results dictionary
    result = {
        'roi_means': areas,
        'roi_exposures': exposures,
        'zero_exposure_indices': np.where(exposures == 0)[0],
        'impossible_rois': np.array([], dtype=int),
        'tangential_pairs': []
    }

    # Handle r_mat construction if requested
    if find_rmat:
        # Initialize r_mat with zeros (n_missed rows × total_circles columns)
        r_mat = np.zeros((n_missed, total_circles))

        # Set expected_proportion at the missed_indices columns
        r_mat[np.arange(n_missed), missed_indices] = expected_proportion

        result['r_mat'] = r_mat

        # Validate column sums if requested
        if validate_rmat:
            col_sums = np.sum(r_mat, axis=0)
            deviations = np.abs(col_sums[missed_indices] - expected_proportion)
            problematic = np.where(deviations > 0.01)[0]  # Small tolerance

            if len(problematic) > 0:
                print(f"Warning: {len(problematic)} circles have incorrect r_mat sums")

    return result


def missed_roi(missed_indices, evtX, evtY, xc, yc, rc, emap_path, secondary_emap,
               filter_result, find_rmat=False, expected_proportion=0.9, validate_rmat=True):
    """
    Updated version with proper type handling for ncir.
    """
    n_missed = len(missed_indices)
    total_circles = len(xc)

    # 1. Verify no events in these circles
    for i in missed_indices:
        dist_sq = (evtX - xc[i]) ** 2 + (evtY - yc[i]) ** 2
        if np.any(dist_sq <= rc[i] ** 2):
            print(f"Warning: Found events in missed circle {i}")

    # 2. Create ROI patterns (non-overlapping)
    uroi_missed = np.zeros((n_missed, total_circles), dtype=bool)
    uroi_missed[np.arange(n_missed), missed_indices] = True
    broi_missed = uroi_missed.T

    # Initialize counts with proper dtype (match filter_result['ncir'])
    nroi_missed = np.zeros(n_missed, dtype=filter_result['ncir'].dtype)
    ncir_missed = np.zeros(total_circles, dtype=filter_result['ncir'].dtype)  # Match original dtype

    # 3. Compute areas and exposures
    area_exp_result = missed_area(
        xc, yc, rc, missed_indices, total_circles,
        emap_path, secondary_emap,
        find_rmat=find_rmat,
        expected_proportion=expected_proportion,
        validate_rmat=validate_rmat
    )

    # 4. Get r_mat
    r_mat_missed = area_exp_result.get('r_mat', None)

    # 5. Update original filter_result
    filter_result['uroi'] = np.vstack([filter_result['uroi'], uroi_missed])
    filter_result['broi'] = np.hstack([filter_result['broi'], broi_missed])
    filter_result['nroi'] = np.concatenate([filter_result['nroi'], nroi_missed])
    filter_result['aroi'] = np.concatenate([filter_result['aroi'], area_exp_result['roi_means']])
    filter_result['eroi'] = np.concatenate([filter_result['eroi'], area_exp_result['roi_exposures']])

    # Safe addition for ncir (ensure matching types)
    filter_result['ncir'] = filter_result['ncir'].astype(np.int64) + ncir_missed.astype(np.int64)

    # Update r_mat if it exists
    if r_mat_missed is not None:
        if filter_result['r_mat'] is not None:
            filter_result['r_mat'] = np.vstack([filter_result['r_mat'], r_mat_missed])
        else:
            new_r_mat = np.zeros((n_missed, total_circles))
            new_r_mat[:, missed_indices] = r_mat_missed
            filter_result['r_mat'] = new_r_mat

    # 6. Update background rate
    if filter_result['aroi'][0] > 0:
        filter_result['bkg_rate'] = filter_result['nroi'][0] / filter_result['aroi'][0]

    # 7. Validate r_mat columns
    if filter_result['r_mat'] is not None:
        col_sums = np.sum(filter_result['r_mat'], axis=0)
        zero_cols = np.where(col_sums == 0)[0]

        if len(zero_cols) == 0:
            print('correction ok')
        else:
            print(f'Columns summing to 0: {zero_cols}')

    return filter_result

###########################
### Functions for design-matrix construction
### Part 6: response concatenation and design matrix construction
##########################
def evt_response(result):
    """
    Reorders the 'nroi' vector in the result dictionary by moving the first entry to the end.

    Parameters:
        result: Dictionary containing an 'nroi' vector

    Returns:
        dict: The modified result dictionary with reordered 'nroi' vector
    """
    # Make a copy of the input dictionary to avoid modifying the original
    result = result.copy()

    # Get the nroi vector
    nroi = result.get('nroi')

    if nroi is not None:
        # Convert to numpy array if it isn't already
        nroi = np.array(nroi) if not isinstance(nroi, np.ndarray) else nroi.copy()

        # Reorder: [1, 2, ..., n, 0]
        if len(nroi) > 0:
            reordered_nroi = np.concatenate([nroi[1:], [nroi[0]]])
            result['nroi'] = reordered_nroi

    return result

def make_evt_design(result, sparse=False):
    """
    Compute design matrix from ROI analysis results.
    Removes first row of r_mat and first entries of eroi/aroi,
    while preserving original aroi[0] for background area.

    Parameters:
        result: Dictionary containing:
            - r_mat: Density matrix
            - eroi: Exposure vector
            - aroi: Area vector
            - m_mat: Mask matrix (optional)
            - Abk: Background area matrix (optional)
        sparse: Whether to use sparse matrices (default: False)

    Returns:
        dict: Contains 'design' matrix and original fields

    Raises:
        ValueError: If dimensions of eroi and r_mat are incompatible after modification.
    """
    # Extract components
    r_mat = result.get('r_mat')
    eroi = result.get('eroi')
    aroi = result.get('aroi')
    m_mat = result.get('m_mat')
    Abk = result.get('Abk')

    # Check for dimension mismatch between eroi and r_mat before processing
    if r_mat is not None and eroi is not None:
        # Expected: len(eroi) - 1 should match r_mat.shape[0] - 1 (after removing first elements)
        if len(eroi) - 1 != r_mat.shape[0] - 1:
            raise ValueError(
                "Missing exposure entries for some segments. "
                f"eroi has length {len(eroi)} (expected {r_mat.shape[0]}), "
                "leading to incompatible dimensions after removing first elements."
            )
            print("aroi shape: ", len(aroi))

    # Store original aroi[0] before modification
    original_aroi0 = aroi[0] if (aroi is not None and len(aroi) > 0) else None

    # Convert to dense if sparse=False
    def ensure_correct_format(x):
        if x is None:
            return None
        if not sparse and issparse(x):
            return x.toarray()
        if sparse and not issparse(x):
            return csr_matrix(x)
        return x

    # Remove first elements unconditionally
    if r_mat is not None:
        r_mat = ensure_correct_format(r_mat)
        r_mat = r_mat[1:] if not issparse(r_mat) else r_mat[1:, :]
    if eroi is not None:
        eroi = ensure_correct_format(eroi)
        eroi = eroi[1:]
    if aroi is not None:
        aroi = ensure_correct_format(aroi)
        aroi = aroi[1:]
    m_mat = ensure_correct_format(m_mat)

    # Create default m_mat if not provided
    if m_mat is None:
        n_segments = r_mat.shape[0] if r_mat is not None else (len(aroi) if aroi is not None else None)
        if n_segments is not None:
            m_mat = csr_matrix(np.ones((n_segments, 1))) if sparse else np.ones((n_segments, 1))

    # Create diagonal matrices
    def create_diag(vec, sparse):
        if vec is None:
            return None
        if sparse:
            return diags(vec.flatten(), 0)
        return np.diag(vec.flatten())

    eroi_mat = create_diag(eroi, sparse)
    aroi_mat = create_diag(aroi, sparse)

    # Compute matrix products
    eroi_r_mat = None
    if eroi_mat is not None and r_mat is not None:
        eroi_r_mat = eroi_mat.dot(r_mat) if sparse else np.matmul(eroi_mat, r_mat)

    aroi_m_mat = None
    if aroi_mat is not None and m_mat is not None:
        aroi_m_mat = aroi_mat.dot(m_mat) if sparse else np.matmul(aroi_mat, m_mat)

    # Handle background area using original aroi[0]
    bkgd_area_mat = None
    if Abk is not None:
        if sparse:
            bkgd_area_mat = diags(Abk.diagonal() if issparse(Abk) else np.diag(Abk), 0)
        else:
            bkgd_area_mat = np.diag(Abk.diagonal() if issparse(Abk) else np.diag(Abk))
    elif original_aroi0 is not None:
        if sparse:
            bkgd_area_mat = csr_matrix([[original_aroi0]])
        else:
            bkgd_area_mat = np.array([[original_aroi0]])

    # Construct design matrix with dimension checks
    design = None
    if all(x is not None for x in [eroi_r_mat, aroi_m_mat, bkgd_area_mat]):
        if sparse:
            # For sparse matrices
            zero_rows = bkgd_area_mat.shape[0]
            zero_cols = eroi_r_mat.shape[1]
            zero_block = csr_matrix((zero_rows, zero_cols))

            top_row = hstack([eroi_r_mat, aroi_m_mat])
            bottom_row = hstack([zero_block, bkgd_area_mat])
            design = vstack([top_row, bottom_row])
        else:
            # For dense matrices
            zero_rows = bkgd_area_mat.shape[0]
            zero_cols = eroi_r_mat.shape[1]
            zero_block = np.zeros((zero_rows, zero_cols))

            top_row = np.hstack([eroi_r_mat, aroi_m_mat])
            bottom_row = np.hstack([zero_block, bkgd_area_mat])
            design = np.vstack([top_row, bottom_row])

    return {
        'design': design,
        **{k: v for k, v in result.items() if k != 'design'}
    }

###########################
### Functions for design-matrix construction
### Part 7: saving all outputs
##########################

def save_roi_to_txt(result, nroi_filename, aroi_filename, r_mat_filename,
                    eroi_filename, xc, yc, rc, tangential_filename=None,
                    design_filename=None, sparse=False):
    """
    Save ROI results to text files according to the specified workflow.
    """
    from scipy.sparse import issparse, csr_matrix, save_npz
    import numpy as np

    # Convert sparse inputs to dense if sparse=False
    def ensure_format(x):
        if x is None:
            return None
        if not sparse and issparse(x):
            return x.toarray()
        return x

    # Extract and convert data
    nroi = ensure_format(result['nroi'])
    aroi = ensure_format(result['aroi']) if 'aroi' in result else None
    r_mat = ensure_format(result['r_mat'])
    eroi = ensure_format(result.get('eroi'))
    tangential_pairs = result.get('tangential_pairs', [])

    # Save tangential pairs if requested
    if tangential_filename and tangential_pairs:
        with open(tangential_filename, 'w') as f:
            for pair in tangential_pairs:
                if len(pair) == 3:  # (roi_idx, c1, c2) format
                    _, c1, c2 = pair
                else:  # Fallback to first two elements
                    c1, c2 = pair[0], pair[1]
                f.write(f"circle({xc[c1]:.6f},{yc[c1]:.6f},{rc[c1]:.6f})\n")
                f.write(f"circle({xc[c2]:.6f},{yc[c2]:.6f},{rc[c2]:.6f})\n")
        print(f"Saved {len(tangential_pairs)} tangential pairs to {tangential_filename}")

    # Save nroi (event counts)
    np.savetxt(nroi_filename,
              np.column_stack((np.arange(len(nroi)), nroi)),
              delimiter='\t',
              fmt=['%d', '%.6f'])
    print(f"Event counts saved to {nroi_filename}")

    # Save aroi (areas)
    if aroi is not None:
        np.savetxt(aroi_filename,
                  np.column_stack((np.arange(len(aroi)), aroi)),
                  delimiter='\t',
                  fmt=['%d', '%.6f'])
        print(f"Areas saved to {aroi_filename}")

    # Save exposure data
    if eroi is not None:
        np.savetxt(eroi_filename,
                  np.column_stack((np.arange(len(eroi)), eroi)),
                  delimiter='\t',
                  fmt=['%d', '%.6f'])
        print(f"Exposures saved to {eroi_filename}")

    # Save density matrix
    if r_mat is not None:
        if sparse:
            save_npz(r_mat_filename, csr_matrix(r_mat))
            print(f"Density matrix saved to {r_mat_filename} (sparse)")
        else:
            np.savetxt(r_mat_filename, r_mat, delimiter='\t', fmt='%.6f')
            print(f"Density matrix saved to {r_mat_filename} (dense)")

    # Save design matrix (using pre-computed design from make_evt_design)
    if design_filename is not None and 'design' in result:
        design_mat = result['design']
        if design_mat is not None:
            if sparse:
                save_npz(design_filename, csr_matrix(design_mat))
                print(f"Design matrix saved to {design_filename} (sparse)")
            else:
                np.savetxt(design_filename,
                           ensure_format(design_mat),
                           delimiter='\t',
                           fmt='%.6f')
                print(f"Design matrix saved to {design_filename} (dense)")
        else:
            print("No design matrix to save")

    print("ROI data saved successfully")

def count_events_in_closest_circle(evtX, evtY, xc, yc, rc):
    """
    For each photon event, find the closest circle, compute the distance to it,
    and count how many events land inside their closest circle.

    Parameters:
        evtX (np.ndarray): X-positions of photon events.
        evtY (np.ndarray): Y-positions of photon events.
        xc (np.ndarray): X-center locations of circles.
        yc (np.ndarray): Y-center locations of circles.
        rc (np.ndarray): Radii of circles.

    Returns:
        int: Number of events that land inside their closest circle.
    """
    # Input validation
    if not all(isinstance(arr, np.ndarray) for arr in [evtX, evtY, xc, yc, rc]):
        raise ValueError("All inputs must be numpy arrays.")
    if len(evtX) != len(evtY):
        raise ValueError("evtX and evtY must have the same size.")
    if len(xc) != len(yc) or len(xc) != len(rc):
        raise ValueError("xc, yc, and rc must have the same size.")

    ntot = len(evtX)  # Total number of events
    mc = len(xc)      # Number of circles

    # Compute pairwise distances between events and circle centers
    dx = evtX[:, np.newaxis] - xc[np.newaxis, :]  # x differences
    dy = evtY[:, np.newaxis] - yc[np.newaxis, :]  # y differences
    distances = np.sqrt(dx**2 + dy**2)  # Euclidean distances

    # Find the closest circle for each event
    closest_circle_indices = np.argmin(distances, axis=1)  # Indices of closest circles
    closest_distances = distances[np.arange(ntot), closest_circle_indices]  # Distances to closest circles

    # Compare distances with radii of closest circles
    closest_radii = rc[closest_circle_indices]  # Radii of closest circles
    inside_closest_circle = closest_distances <= closest_radii  # Boolean mask for events inside closest circle

    # Count the number of events inside their closest circle
    num_inside = np.sum(inside_closest_circle)

    return num_inside


####################
# separate background
###################

def compute_area_only(i, xc, yc, rc, overlap_data=None, verbose=False, circMCn=1e5):
    """
    Compute only the area components for a single circle, using area_exposure_circle_mc()
    for multiple overlap cases.

    Parameters:
        i (int): Index of target circle
        xc, yc, rc (arrays): Circle parameters (x, y coordinates and radii)
        overlap_data (dict): Precomputed overlap information (optional)
        verbose (bool): Print debug information
        circMCn (int): Number of MC samples for multiple overlap case

    Returns:
        tuple: (segments, areas) where:
            segments: Array of segment patterns
            areas: Corresponding areas for each segment
    """
    # Calculate overlaps if not provided
    if overlap_data is None:
        overlap_data = overlap_correct(n_overlap(xc, yc, rc))

    overlap_indices = overlap_data['overlap_indices'][i]

    # Case 1: No overlaps
    if len(overlap_indices) == 0:
        seg = np.zeros(len(xc), dtype=np.uint8)
        seg[i] = 1
        area = pi * rc[i] ** 2
        return np.array([seg]), np.array([area])

    # Case 2: Single overlap
    if len(overlap_indices) == 1:
        j = overlap_indices[0]
        d = overlap_data['dsep_matrix'][i, j]
        overlap = circolap(d, rc[i], rc[j])['overlap_area']

        segments = np.zeros((2, len(xc)), dtype=np.uint8)
        segments[0, i] = 1  # Circle i only
        segments[1, [i, j]] = 1  # Overlap region

        areas = np.array([
            pi * rc[i] ** 2 - overlap,  # Non-overlapping part
            overlap  # Overlapping area
        ])
        return segments, areas

    # Case 3: Multiple overlaps - use area_exposure_circle_mc()
    if verbose:
        print(f"Circle {i} has multiple overlaps, using area_exposure_circle_mc()")

    # Call with emap_path=None to skip exposure calculation
    segments, areas, _ = area_exposure_circle_mc(
        i=i,
        xc=xc,
        yc=yc,
        rc=rc,
        emap_path=None,  # Skip exposure calculation
        circMCn=circMCn,
        verbose=verbose,
        overlap_data=overlap_data,
        max_attempts=5,
        exposure_bin=32,
        find_rmat=False  # Skip density calculation
    )

    return segments, areas


def _process_area_wrapper(i, xc, yc, rc, overlap_data=None, circMCn=1e5, verbose=False):
    """
    Standalone processing function for parallel execution that focuses only on area computation.
    Wrapper around compute_area_only() with consistent interface for parallel processing.

    Parameters:
        i (int): Index of target circle
        xc, yc, rc (arrays): Circle parameters
        overlap_data (dict): Precomputed overlap information
        circMCn (int): Number of MC samples for multiple overlaps case
        verbose (bool): Verbose output flag

    Returns:
        tuple: (segments, areas) - same as compute_area_only()
    """
    try:
        # Call compute_area_only with all parameters
        return compute_area_only(
            i=i,
            xc=xc,
            yc=yc,
            rc=rc,
            overlap_data=overlap_data,
            verbose=verbose,
            circMCn=circMCn
        )
    except Exception as e:
        # Provide error context for parallel debugging
        error_msg = f"Error processing circle {i} at ({xc[i]}, {yc[i]}) with r={rc[i]}: {str(e)}"
        if verbose:
            print(error_msg)
        raise RuntimeError(error_msg) from e


def area_multiple(xc, yc, rc, overlap_data, circMCn=1000,
                  parallel=False, empty_cores=0.15, mem_threshold=0.8,
                  verbose=False, impossible_txt=None):
    """
    ROI area computation pipeline (without exposure or rmat calculations).

    Parameters:
        xc, yc, rc: Circle parameters (arrays)
        overlap_data: Precomputed overlap information
        circMCn: Base number of MC samples
        parallel: Enable parallel processing
        empty_cores: Fraction of CPU cores to leave unused
        mem_threshold: Memory usage threshold for fallback to serial
        verbose: Print progress information
        impossible_txt: Path to save impossible ROIs (optional)

    Returns:
        Dictionary containing:
            roi_means: Area means for each ROI
            roi_patterns: ROI patterns
            roi_counts: Counts per ROI
            impossible_rois: Indices of impossible ROIs
            tangential_pairs: List of tangential circle pairs
    """
    # Input validation
    required_keys = {'uroi', 'overlap_matrix', 'overlap_indices'}
    if not all(k in overlap_data for k in required_keys):
        missing = required_keys - overlap_data.keys()
        raise KeyError(f"Missing keys in overlap_data: {missing}")

    uroi = np.asarray(overlap_data['uroi'], dtype=np.uint8)
    if uroi.ndim != 2:
        raise ValueError("uroi must be 2D array")

    n_roi, n_circles = uroi.shape
    is_sparse = issparse(overlap_data.get('overlap_matrix', None))

    # Initialize area matrix only
    area_mat = lil_matrix((n_roi, n_circles), dtype=np.float64) if is_sparse else \
        np.zeros((n_roi, n_circles), dtype=np.float64)

    # Memory check
    mem = psutil.virtual_memory()
    if mem.used / mem.total > mem_threshold:
        print(f"Memory usage {mem.used / mem.total:.1%} > threshold {mem_threshold:.0%} - forcing serial")
        parallel = False

    # Primary processing
    try:
        if parallel:
            max_workers = max(1, int((1 - empty_cores) * os.cpu_count()))
            print(f"Parallel processing with {max_workers} workers")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _process_area_wrapper,
                        i, xc, yc, rc, overlap_data, circMCn, verbose
                    ): i for i in range(n_circles)
                }

                results = []
                for future in tqdm(as_completed(futures), total=n_circles,
                                   desc="Processing areas"):
                    mem = psutil.virtual_memory()
                    if mem.used / mem.total > mem_threshold:
                        raise MemoryError("Memory threshold exceeded")

                    i = futures[future]
                    segments, areas = future.result()
                    results.append((i, segments, areas))

                # Sort and update matrix
                results.sort()
                for i, segments, areas in results:
                    _update_area_matrix(area_mat, segments, areas, uroi, i, is_sparse)
        else:
            for i in tqdm(range(n_circles), desc="Processing areas serially"):
                segments, areas = _process_area_wrapper(
                    i, xc, yc, rc, overlap_data, circMCn, verbose
                )
                _update_area_matrix(area_mat, segments, areas, uroi, i, is_sparse)

    except (MemoryError, pickle.PicklingError) as e:
        print(f"Parallel failed ({str(e)}), falling back to serial")
        return _serial_area_only(xc, yc, rc, overlap_data, circMCn, is_sparse, verbose)
    except Exception as e:
        print(f"Critical error: {traceback.format_exc()}")
        raise

    # Calculate area means
    aroi = _calculate_adjusted_means(area_mat, is_sparse)

    # Identify impossible ROIs (zero area multi-circle ROIs)
    zero_mask = aroi == 0
    zero_indices = np.where(zero_mask)[0]
    impossible_rois = set()
    tangential_pairs = []

    for roi_idx in zero_indices:
        circle_indices = np.where(uroi[roi_idx] == 1)[0]
        if len(circle_indices) == 1:  # Skip single-circle ROIs
            continue

        impossible_rois.add(roi_idx)
        if len(circle_indices) == 2:
            tangential_pairs.append((circle_indices[0], circle_indices[1]))

    impossible_rois = np.array(sorted(impossible_rois), dtype=int)

    # Save impossible ROIs if requested
    if impossible_txt and len(impossible_rois) > 0:
        with open(impossible_txt, 'w') as f:
            f.write("roi_index\tnum_circles\tcircle_indices\tx_coords\ty_coords\tradii\tarea\n")
            for roi_idx in impossible_rois:
                circle_indices = np.where(uroi[roi_idx] == 1)[0]
                f.write(
                    f"{roi_idx}\t"
                    f"{len(circle_indices)}\t"
                    f"{','.join(map(str, circle_indices))}\t"
                    f"{','.join(f'{xc[i]:.2f}' for i in circle_indices)}\t"
                    f"{','.join(f'{yc[i]:.2f}' for i in circle_indices)}\t"
                    f"{','.join(f'{rc[i]:.2f}' for i in circle_indices)}\t"
                    f"{aroi[roi_idx]:.2f}\n"
                )

    # Create valid ROI mask
    valid_roi_mask = np.ones(n_roi, dtype=bool)
    valid_roi_mask[impossible_rois] = False

    return {
        'roi_means': aroi[valid_roi_mask],
        'roi_patterns': uroi[valid_roi_mask],
        'roi_counts': overlap_data.get('nroi', np.zeros(n_roi, dtype=int))[valid_roi_mask],
        'impossible_rois': impossible_rois,
        'tangential_pairs': tangential_pairs
    }


def _update_area_matrix(area_mat, segments, areas, uroi, circle_idx, is_sparse):
    """Helper to update area matrix for a single circle"""
    matches = np.all(segments[:, None, :] == uroi[None, :, :], axis=-1)
    for seg_idx in range(matches.shape[0]):
        matched_rois = np.where(matches[seg_idx])[0]
        for matched_roi in matched_rois:
            if is_sparse:
                area_mat[matched_roi, circle_idx] = areas[seg_idx]
            else:
                area_mat[matched_roi, circle_idx] = areas[seg_idx]


def _serial_area_only(xc, yc, rc, overlap_data, circMCn, is_sparse, verbose):
    """Fallback serial processing when parallel fails"""
    n_roi, n_circles = overlap_data['uroi'].shape
    area_mat = lil_matrix((n_roi, n_circles), dtype=np.float64) if is_sparse else \
        np.zeros((n_roi, n_circles), dtype=np.float64)

    for i in range(n_circles):
        segments, areas = _process_area_wrapper(
            i, xc, yc, rc, overlap_data, circMCn, verbose
        )
        _update_area_matrix(area_mat, segments, areas, overlap_data['uroi'], i, is_sparse)

    return {
        'roi_means': _calculate_adjusted_means(area_mat, is_sparse),
        'roi_patterns': overlap_data['uroi'],
        'roi_counts': overlap_data.get('nroi', np.zeros(n_roi, dtype=int)),
        'impossible_rois': np.array([], dtype=int),
        'tangential_pairs': []
    }


def filter_evt_for_bkgd(evtX, evtY, xc, yc, rc, theta, centre_x, centre_y, verbose=False):
    """
    Compute only background count and area from event data.

    Parameters:
        evtX, evtY: Event coordinates (np.ndarray)
        xc, yc, rc: Circle parameters (np.ndarray)
        verbose: Print progress info (bool)

    Returns:
        dict: {
            'bkg_count': Number of background events,
            'bkg_area': Background area,
            'bkg_rate': Background rate (count/area),
            'bkg_vertices': Background region vertices,
            'bkg_angles': Vertex angles
        }
    """
    # 1. Compute basic event overlaps
    overlap_data = overlap_correct(n_overlap(xc, yc, rc, evtX, evtY, sparse=True))

    # 2. Initialize minimal result structure
    result = {
        'nroi': overlap_data['nroi'],  # Event counts per ROI
        'aroi': None,  # Will be filled with areas
        'uroi': overlap_data['uroi']  # ROI patterns
    }

    # 3. Compute areas only (no exposure or rmat)
    area_result = area_multiple(
        xc, yc, rc,
        overlap_data=overlap_data,
        parallel=False,
        verbose=verbose
    )

    result['aroi'] = area_result['roi_means']
    result['uroi'] = area_result['roi_patterns']
    result['nroi'] = overlap_data['nroi']#[:len(result['aroi'])]  # Match dimensions

    # Compute background on filtered ROIs
    if len(result['aroi']) > 1 and len(result['nroi']) > 0:
        bg_result = background_area(xc, yc, rc, result['aroi'].copy(), theta, centre_x, centre_y, verbose)
        if bg_result is not None:
            bkg_area, bkg_radius, max_theta = bg_result
            result['nroi'] = background_count(bkg_radius, centre_x, centre_y, evtX, evtY, result['nroi'])
            bkg_count = result['nroi'][0]  # Background is always first ROI
            return {
                'nroi': result['nroi'],
                'bkg_count': bkg_count,
                'bkg_area': bkg_area[0],  # Background area is first element
                'bkg_rate': bkg_count / bkg_area[0] if bkg_area[0] > 0 else 0,
                'bkg_radius': bkg_radius,
                'max_theta': max_theta
            }

    # Fallback if background computation fails
    if verbose:
        print("Warning: Could not compute proper background region")
    return {
        'bkg_count': 0,
        'bkg_area': 0,
        'bkg_rate': 0,
        'bkg_radius': None,
        'max_theta': None
    }


def save_bkgd_to_txt(bkgd_info, output_filename, save_radius=True, verbose=True):
    """
    Save background information from filter_evt_for_bkgd() to a text file.

    Parameters:
        bkgd_info (dict): Output dictionary from filter_evt_for_bkgd()
        output_filename (str): Path to output text file
        save_vertices (bool): Whether to save vertex coordinates
        verbose (bool): Whether to print status messages
    """
    try:
        with open(output_filename, 'w') as f:
            # Write header
            f.write("# Background region information\n")
            f.write(f"# Count: {bkgd_info['bkg_count']}\n")
            f.write(f"# Area: {bkgd_info['bkg_area']:.6f}\n")
            f.write(f"# Rate: {bkgd_info['bkg_rate']:.6f}\n")

            # Save vertex information if available
            if save_radius and bkgd_info['bkg_radius'] is not None and bkgd_info['max_theta'] is not None:
                f.write("\n# Radius of background region:\n")
                f.write(f"{bkgd_info['bkg_radius']}\n")
                f.write("\n# Max theta:\n")
                f.write(f"{bkgd_info['max_theta']}\n")

        if verbose:
            print(f"Background information saved to {output_filename}")
            print(f"- Count: {bkgd_info['bkg_count']}")
            print(f"- Area: {bkgd_info['bkg_area']:.6f}")
            print(f"- Rate: {bkgd_info['bkg_rate']:.6f}")

    except Exception as e:
        print(f"Error saving background information: {str(e)}")
        raise