import os
os.system('source /Users/jl1023/anaconda3/envs/ciao-4.17/bin/ciao.sh')
os.environ['PARAM_FILE'] = '/Users/jl1023/cxcds_param4/dmstat.par'
import ciao_contrib.runtool
from ciao_contrib.runtool import *
from ciao_contrib.runtool import dmstat

# Define the input and output file names
input_files = {
    "b32_coord.txt": 1.5,  # a = 1.5 for b32_coord.txt
    "b16_coord.txt": 1.25  # a = 1.25 for b16_coord.txt
}
output_files = {
    "b32_coord.txt": "b32_coord_clear.txt",
    "b16_coord.txt": "b16_coord_clear.txt"
}

# Map input files to their corresponding emap files
emap_files = {
    "b32_coord.txt": "27_b32.emap",
    "b16_coord.txt": "27_b16.emap"
}

# Function to process a file
def process_file(input_file, output_file, a, emap_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Strip any leading/trailing whitespace
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            # Construct the dmstat command
            dmstat_command = f"{emap_file}[sky={line}]"
            dmstat(dmstat_command, centroid=False, sigma=False)

            # Check the condition
            out_min = float(dmstat.out_min)
            out_max = float(dmstat.out_max)
            if a * out_min < out_max:
                print(f"Removing line from {input_file}: {line}")
            else:
                # Write the line to the output file
                outfile.write(line + "\n")

# Process each file
for input_file, a in input_files.items():
    output_file = output_files[input_file]
    emap_file = emap_files[input_file]  # Get the corresponding emap file
    process_file(input_file, output_file, a, emap_file)

print("Processing complete. Cleaned files saved.")