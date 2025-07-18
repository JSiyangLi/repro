def read_specific_lines(file_path, line_numbers):
    """Read specific lines from a text file (0-based indexing)."""
    lines_to_read = set(line_numbers)  # Use a set for O(1) lookups
    lines_found = {}

    with open(file_path, 'r') as file:
        for current_line_num, line in enumerate(file):  # Starts at 0 by default
            if current_line_num in lines_to_read:
                lines_found[current_line_num] = line.strip()
                # Early exit if all lines are found
                if len(lines_found) == len(lines_to_read):
                    break
    return lines_found


# Define the file path and line numbers (0-based)
file_path = "cbind_sub_cleared.xyreg"
line_numbers = [639, 640, 719, 720, 723, 724, 804, 805, 885, 886, 946, 947]  # Now treated as 0-based

# Read the lines
found_lines = read_specific_lines(file_path, line_numbers)

# Print the results (showing 0-based indices)
for line_num, line_content in found_lines.items():
    print(f"Line (0-based) {line_num}: {line_content}")

# Check for missing lines
missing_lines = set(line_numbers) - set(found_lines.keys())
if missing_lines:
    print(f"\nWarning: The following 0-based lines were not found: {sorted(missing_lines)}")


