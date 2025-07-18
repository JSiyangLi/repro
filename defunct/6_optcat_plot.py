import numpy as np
import matplotlib.pyplot as plt

# Step 11: Load the saved outputs
data = np.load('plotting_data.npz')
prop_values = data['prop_values']
mean_expected_angsep = data['mean_expected_angsep']
mean_num_matches = data['mean_num_matches']
mean_var_angsep = data['mean_var_angsep']
var_expected_angsep = data['var_expected_angsep']
var_num_matches = data['var_num_matches']
num_nans_expected_angsep = data['num_nans_expected_angsep']
num_nans_num_matches = data['num_nans_num_matches']
optresults_expected_angsep = data['optresults_expected_angsep']
optresults_num_matches = data['optresults_num_matches']
closest_expected_angsep_prop = data['closest_expected_angsep_prop']
closest_expected_angsep_value = data['closest_expected_angsep_value']
intersections_expected_angsep = data['intersections_expected_angsep']
intersections_num_matches = data['intersections_num_matches']
null_siz = data['null_siz']

# Step 12: Calculate standard errors
std_error_expected_angsep = np.sqrt(mean_var_angsep) / np.sqrt(optresults_num_matches) # mean SE of individual catalogue
std_error_num_matches = np.sqrt(var_num_matches) / np.sqrt(null_siz - num_nans_num_matches)
std_error_expected_angsep_var = np.sqrt(var_expected_angsep) / np.sqrt(null_siz - num_nans_expected_angsep) # SE of avg-separation-per-catalogue across the whole null sample

# Step 13: Create the first multi-facet plot with two facets
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot mean expected_angsep vs. prop_values with both sets of standard error bars
ax1.errorbar(prop_values, optresults_expected_angsep, yerr=std_error_expected_angsep, color='g', linestyle='-', label='Expected Angular Separation ± mean SE(separation per catalogue)')
ax1.errorbar(prop_values, mean_expected_angsep, yerr=std_error_expected_angsep_var, color='c', linestyle='-', label='Mean Expected Angular Separation ± SE(avg-separation per catalogue)')
ax1.scatter(closest_expected_angsep_prop, closest_expected_angsep_value, color='r', zorder=5)
ax1.annotate(f'Most CorrInvariant:\nArea: {closest_expected_angsep_prop:.2f}\nSeparation: {closest_expected_angsep_value:.2f}',
             xy=(closest_expected_angsep_prop, closest_expected_angsep_value),
             xytext=(10, 10), textcoords='offset points', color='r')

# Plot intersection points for expected_angsep
if len(intersections_expected_angsep) > 0:
    for x, y in intersections_expected_angsep:
        ax1.scatter(x, y, color='orange', zorder=5)
        ax1.annotate(f'Intersection: ({x:.2f}, {y:.2f})',
                     xy=(x, y), xytext=(10, -20), textcoords='offset points', color='orange')
else:
    # Add a sentence if there are no intersections
    ax1.text(0, 0, 'No intersection between angular separation curves',
             horizontalalignment='left', verticalalignment='bottom',
             transform=ax1.transAxes, fontsize=12, color='red')

ax1.set_xlabel('Proportion Threshold')
ax1.set_ylabel('Mean Expected Angular Separation (arcsec)')
ax1.set_title('Mean Expected Angular Separation vs. Proportion Threshold')
ax1.legend()

# Plot mean num_matches vs. prop_values with standard error bars
ax2.errorbar(prop_values, mean_num_matches, yerr=std_error_num_matches, color='r', linestyle='-', label='Mean Number of Matches ± SE')
ax2.plot(prop_values, optresults_num_matches, color='purple', linestyle='--', label='Original Number of Matches')

# Plot the minimum mean number of matches
min_num_matches_idx = np.nanargmin(mean_num_matches)
min_num_matches_prop = prop_values[min_num_matches_idx]
min_num_matches_value = mean_num_matches[min_num_matches_idx]
ax2.scatter(min_num_matches_prop, min_num_matches_value, color='b', zorder=5)
ax2.annotate(f'Area: {min_num_matches_prop:.2f}\nMatches: {min_num_matches_value:.2f}',
             xy=(min_num_matches_prop, min_num_matches_value),
             xytext=(10, 10), textcoords='offset points', color='b')

# Plot intersection points for num_matches
for x, y in intersections_num_matches:
    ax2.scatter(x, y, color='orange', zorder=5)
    ax2.annotate(f'Intersection: ({x:.2f}, {y:.2f})',
                 xy=(x, y), xytext=(10, -20), textcoords='offset points', color='orange')

ax2.set_xlabel('Proportion Threshold')
ax2.set_ylabel('Mean Number of Matches')
ax2.set_title('Mean Number of Matches vs. Proportion Threshold')
ax2.legend()

# Add null_siz information at the bottom of the first plot
plt.figtext(0.5, 0.01, f'Null Sample Size = {null_siz}', ha='center', fontsize=12)

# Adjust layout for the first plot
plt.tight_layout()
plt.show()

# Step 14: Create the second standalone plot with two facets
fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 12))

# Facet 1: Plot false match proportion (mean_num_matches / optresults_num_matches)
false_match_proportion = mean_num_matches / optresults_num_matches
min_false_match_idx = np.nanargmin(false_match_proportion)
min_false_match_prop = prop_values[min_false_match_idx]
min_false_match_value = false_match_proportion[min_false_match_idx]

ax3.plot(prop_values, false_match_proportion, color='orange', linestyle='-', label='False Match Proportion')
ax3.scatter(min_false_match_prop, min_false_match_value, color='b', zorder=5)
ax3.annotate(f'Area: {min_false_match_prop:.2f}\nFalse Match: {min_false_match_value:.2f}',
             xy=(min_false_match_prop, min_false_match_value),
             xytext=(10, 10), textcoords='offset points', color='b')
ax3.set_xlabel('Proportion Threshold')
ax3.set_ylabel('False Match Proportion')
ax3.set_title('False Match Proportion vs. Proportion Threshold')
ax3.legend()

# Facet 2: Plot optresults_num_matches - mean_num_matches
difference_num_matches = optresults_num_matches - mean_num_matches
max_difference_idx = np.nanargmax(difference_num_matches)
max_difference_prop = prop_values[max_difference_idx]
max_difference_value = difference_num_matches[max_difference_idx]

ax4.plot(prop_values, difference_num_matches, color='purple', linestyle='-', label='Difference in Number of Matches')
ax4.scatter(max_difference_prop, max_difference_value, color='r', zorder=5)
ax4.annotate(f'Area: {max_difference_prop:.2f}\nMatches: {max_difference_value:.2f}',
             xy=(max_difference_prop, max_difference_value),
             xytext=(10, -20), textcoords='offset points', color='r')
ax4.set_xlabel('Proportion Threshold')
ax4.set_ylabel('Expected True Matches per Catalogue')
ax4.set_title('Expected True Matches vs. Proportion Threshold')
ax4.legend()

# Add null_siz information at the bottom of the second plot
plt.figtext(0.5, 0.01, f'Null Sample Size = {null_siz}', ha='center', fontsize=12)

# Adjust layout for the second plot
plt.tight_layout()
plt.show()

# Step 14: Print the identified prop_values, global variables, NaN counts, and intersection points
print(f"Proportion threshold for closest mean expected_angsep to original: {closest_expected_angsep_prop}")
print(f"Closest mean expected angular separation: {closest_expected_angsep_value:.4f} arcsec")
print(f"Proportion threshold for minimum mean number of matches: {min_num_matches_prop}")
print(f"Minimum mean number of matches: {min_num_matches_value:.4f}")
print(f"Proportion threshold for minimum false match proportion: {min_false_match_prop}")
print(f"Minimum false match proportion: {min_false_match_value:.4f}")
print(f"Number of NaNs in mean expected angular separation: {num_nans_expected_angsep}")
print(f"Number of NaNs in mean number of matches: {num_nans_num_matches}")

# Print intersection points for expected_angsep
if len(intersections_expected_angsep) > 0:
    print("\nIntersection points for expected angular separation:")
    for i, (x, y) in enumerate(intersections_expected_angsep, start=1):
        print(f"Intersection {i}: Proportion Threshold = {x:.4f}, Expected Angular Separation = {y:.4f} arcsec")
else:
    print("\nNo intersection points found for expected angular separation.")

# Print intersection points for num_matches
if len(intersections_num_matches) > 0:
    print("\nIntersection points for number of matches:")
    for i, (x, y) in enumerate(intersections_num_matches, start=1):
        print(f"Intersection {i}: Proportion Threshold = {x:.4f}, Number of Matches = {y:.4f}")
else:
    print("\nNo intersection points found for number of matches.")