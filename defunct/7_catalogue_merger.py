import numpy as np
from catfun import parse_circle_file, match_xray_optical_sources, iterate_prop_values, plot_density_results, is_mean_significantly_different, adjust_optical_catalogues, filter_xyoffx_file, compare_row_counts, combine_txt_files
from coordfun import dm_xyoffx_to_file

def get_prop_from_user():
    """
    Prompt the user to input 'prop=...' and extract the prop value.

    Returns:
        float: The extracted prop value.
    """
    while True:
        # Prompt the user for input
        user_input = input("Please enter 'prop=...': ").strip()

        # Check if the input starts with 'prop='
        if user_input.startswith("prop="):
            try:
                # Extract the numeric value after 'prop='
                prop_value = float(user_input.split("=")[1])

                # Check if the value is between 0 and 1
                if 0 <= prop_value <= 1:
                    return prop_value
                else:
                    print("Error: prop must be between 0 and 1. Please try again.")
            except ValueError:
                print("Error: Invalid input. Please enter a numeric value after 'prop='.")
        else:
            print("Error: Input must start with 'prop='. Please try again.")
prop_value = get_prop_from_user()

# Read X-ray detections from unique_stars.txt
xray_detections = parse_circle_file('unique_stars.txt')

if not compare_row_counts('unique_stars.txt', 'unique_xray.xyoffx'):
    raise ValueError("Adjusted and original optical sources differ in number")

# Read optical catalogues from J_A+A_375_863_sub.reg
optical_catalogues = parse_circle_file('J_A+A_375_863_sub.reg')

# Define constants
distance_threshold = 0.1318 * 8 * np.sqrt(2)  # Distance threshold in arcsec

# Iterate over prop values and compute results
results = match_xray_optical_sources(xray_detections, optical_catalogues, distance_threshold, prop_value, filtering=True, verbose=True)

# Plot empirical densities
plot_density_results(results)

# Conduct tests
data = np.load('plotting_data.npz')
need_adjust = is_mean_significantly_different(prop_value, data)

# Check if adjustments are needed
if need_adjust:
    print("Adjustments on RA, DEC, and radii will be done.")

    # Step 1: Adjust RA, DEC, and radii in the optical catalogues
    ra_list, dec_list, adj_optcat = adjust_optical_catalogues(
        results['filtered_catalogues'], results['expected_ra_diff'], results['expected_dec_diff'], "27_b32_psf.fits"
    )

    # Step 2: Re-run dm_xyoffx_to_file to generate adjusted coordinates
    dm_xyoffx_to_file(ra_list, dec_list, 'J_A+A_375_863_adj.xyoffx')

    # Step 3: Merge xray_detections and adj_optcat
    merged_catalogues = xray_detections + adj_optcat

    # Step 4: Filter the adjusted optical coordinates and combine with X-ray coordinates
    filter_xyoffx_file("unique_xray.xyoffx", 'J_A+A_375_863_adj.xyoffx', results['optical_removals'], "cbind_coords.xyoffx")

else:
    # No adjustments needed
    adj_optcat = optical_catalogues

    # Merge xray_detections and adj_optcat
    merged_catalogues = xray_detections + adj_optcat

    # Filter the original optical coordinates and combine with X-ray coordinates
    filter_xyoffx_file("unique_xray.xyoffx", 'J_A+A_375_863_sub.xyoffx', results['optical_removals'], "cbind_coords.xyoffx")

# Check if the number of rows in "cbind_coords.xyoffx" matches the length of merged_catalogues
if not compare_row_counts("cbind_coords.xyoffx", len(merged_catalogues)):
    raise ValueError("radec and xyoffx differ in number of rows")

# Create a label array: 0 for X-ray sources, 1 for optical sources
source_labels = np.array([0] * len(xray_detections) + [1] * len(adj_optcat))

# Count the number of X-ray and filtered optical sources
num_xray_sources = len(xray_detections)
num_filtered_optical_sources = len(adj_optcat)

# Print the results
print(f"Number of X-ray Sources: {num_xray_sources}")
print(f"Number of filtered Optical Sources: {num_filtered_optical_sources}")

# Save the counts to a .txt file
with open('source_counts.txt', 'w') as f:
    f.write(f"Number of X-ray Sources: {num_xray_sources}\n")
    f.write(f"Number of filtered Optical Sources: {num_filtered_optical_sources}\n")
    f.write(f"Source labels array: {source_labels}\n")

# Write merged catalogues to a single .txt file
with open('merged_catalogues.txt', 'w') as f:
    for entry in merged_catalogues:
        ra, dec, radius = entry  # Unpack the entry (RA, Dec, radius)
        # Write the entry in the desired format
        f.write(f"circle({ra}d, {dec}d, {radius}\")\n")