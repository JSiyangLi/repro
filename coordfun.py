import numpy as np
import ciao_contrib.runtool
from ciao_contrib.runtool import *
from ciao_contrib.runtool import dmcoords
from astropy.wcs import WCS
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u

def dm_xyoffx_to_file(ra, dec, evtfile, writefile):
    """Convert RA/Dec to physical coordinates using dmcoords and write to file"""
    with open(writefile, 'w') as f:
        for i in range(len(ra)):
            # Run dmcoords
            dmcoords(infile=evtfile, asolfile=None, opt="cel", ra=ra[i], dec=dec[i])

            # Get coordinates
            xpos = dmcoords.x
            ypos = dmcoords.y
            theta = dmcoords.theta
            phi = dmcoords.phi

            # Write results to the output file
            f.write(f"{xpos}\t{ypos}\t{theta}\t{phi}\n")


def dm_xyoffx_filter(ra, dec, writefile, theta_thresh=np.nan):
    """
    Wrapper for dm_xyoffx_to_file and filter_xyoffx_file with optional theta filtering.

    Parameters:
        ra (list or array): List of RA values.
        dec (list or array): List of DEC values.
        writefile (str): Path to the output file.
        theta_thresh (float): Threshold for filtering theta values. If nan, no filtering is applied.
    """
    # Run dm_xyoffx_to_file to generate coordinates
    dm_xyoffx_to_file(ra, dec, "temp_coords.xyoffx")

    # Load the temporary coordinates file
    coords = np.loadtxt("temp_coords.xyoffx")

    # Apply theta filtering if theta_thresh is not nan
    if not np.isnan(theta_thresh):
        # Filter rows where theta <= theta_thresh
        coords = coords[coords[:, 2] <= theta_thresh]

    # Write the filtered coordinates to the output file
    np.savetxt(writefile, coords, delimiter="\t", fmt="%.6f")

def dm_xyreg_to_file(ra, dec, psfsiz, writefile, evtfile, pix_scale=0.1318):
    """
    Save circle regions in the format 'circle(xpos, ypos, psfsiz_pix)' to a file.

    Parameters:
        ra (list or np.ndarray): RA coordinates of the circles in degrees.
        dec (list or np.ndarray): DEC coordinates of the circles in degrees.
        writefile (str): Path to the output file.
        psfsiz (list or np.ndarray): PSF sizes in arcseconds.
        pix_scale (float): Pixel scale in arcseconds per pixel (default: 0.1318).
    """
    # Ensure psfsiz is a NumPy array of type float
    psfsiz = np.array(psfsiz, dtype=float)

    # Convert psfsiz from arcseconds to pixels
    psfsiz_pix = psfsiz / pix_scale

    # Open the output file for writing
    with open(writefile, 'w') as f:
        for i in range(len(ra)):
            # Run dmcoords to get xpos and ypos
            dmcoords(infile=evtfile, asolfile=None, opt="cel", ra=ra[i], dec=dec[i])

            # Get coordinates
            xpos = dmcoords.x
            ypos = dmcoords.y

            # Write the circle region to the file
            f.write(f"circle({xpos}, {ypos}, {psfsiz_pix[i]})\n")

def check_circle_overlaps_file(filename):
    """
    Check if there are any overlapping circles in the given file using vectorization.

    Parameters:
        filename (str): Path to the file containing circle definitions.

    Returns:
        list: A list of tuples containing the indices of overlapping circles.
    """
    # Parse the file to extract circle parameters
    circles = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('circle'):
                # Extract xpos, ypos, and radius
                parts = line.strip().replace('circle(', '').replace(')', '').split(',')
                xpos = float(parts[0])
                ypos = float(parts[1])
                radius = float(parts[2])
                circles.append((xpos, ypos, radius))

    # Convert to a NumPy array for vectorized operations
    circles = np.array(circles)
    x = circles[:, 0]  # x positions
    y = circles[:, 1]  # y positions
    r = circles[:, 2]  # radii

    # Compute pairwise distances between all circles
    # Using broadcasting to avoid explicit loops
    dx = x[:, np.newaxis] - x[np.newaxis, :]  # x differences
    dy = y[:, np.newaxis] - y[np.newaxis, :]  # y differences
    distances = np.sqrt(dx**2 + dy**2)  # Euclidean distances

    # Compute pairwise sum of radii
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]

    # Find overlapping pairs (excluding self-comparisons)
    overlaps = np.where((distances < r_sum) & (distances > 0))

    # Convert to a list of tuples
    overlap_pairs = list(zip(overlaps[0], overlaps[1]))

    return overlap_pairs

# Function to check if the center of one circle is inside another circle
def center_inside(circle1, circle2):
    # Extract RA, Dec, and radius from the dictionaries
    ra1, dec1, r1_arcsec = circle1["ra"], circle1["dec"], circle1["r_arcsec"]
    ra2, dec2, r2_arcsec = circle2["ra"], circle2["dec"], circle2["r_arcsec"]
    # Create SkyCoord objects for the centers of the circles
    coord1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
    coord2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
    # Calculate the angular separation between the centers
    separation = coord1.separation(coord2).arcsecond
    # Check if the center of circle1 is inside circle2
    return separation <= r2_arcsec

# Function to check if two circles overlap
def circles_overlap(circle1, circle2):
    # Extract RA, Dec, and radius from the dictionaries
    ra1, dec1, r1_arcsec = circle1["ra"], circle1["dec"], circle1["r_arcsec"]
    ra2, dec2, r2_arcsec = circle2["ra"], circle2["dec"], circle2["r_arcsec"]
    # Create SkyCoord objects for the centers of the circles
    coord1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg, frame='icrs')
    coord2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg, frame='icrs')
    # Calculate the angular separation between the centers
    separation = coord1.separation(coord2).arcsecond
    # Check if the circles overlap
    return separation <= (r1_arcsec + r2_arcsec)

# Function to check if a circle should be removed based on the diameter condition
def should_remove_circle(circle, bin_size, pixel_scale=0.1318):
    # Calculate the diameter of the circle in arcseconds
    diameter_arcsec = 2 * circle["r_arcsec"]
    # Check if the diameter is smaller than the bin size in pixels multiplied by the pixel scale
    return diameter_arcsec < (bin_size * pixel_scale * np.sqrt(2))
