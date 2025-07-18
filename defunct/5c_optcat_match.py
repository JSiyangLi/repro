import numpy as np
from catfun import parse_circle_file, match_xray_optical_sources, iterate_prop_values, plot_results, plot_density_results, conduct_t_tests, adjust_optical_catalogues
import matplotlib.pyplot as plt

# Read X-ray detections from unique_stars.txt
xray_detections = parse_circle_file('unique_stars.txt')

# Read optical catalogues from J_A+A_375_863_sub.reg
optical_catalogues = parse_circle_file('J_A+A_375_863_sub.reg')

# Define constants
distance_threshold = 0.1318 * 8 * np.sqrt(2)  # Distance threshold in arcsec

# Check for overlaps and apply filtering rules
print('start expectation plotting')
# Iterate over prop values and compute results
results = iterate_prop_values(xray_detections, optical_catalogues, distance_threshold, prop_values = np.arange(0.01, 1, 0.01))

# Plot the results, expectations and variances vs. proportion threshold
plot_results(results)