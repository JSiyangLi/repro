import os
from catfun import parse_circle_file
from coordfun import dm_xyreg_to_file, dm_xyoffx_to_file

# Debugging - verify file access
print("\n=== Debugging Information ===")
print(f"Current directory: {os.getcwd()}")
evtfile = "hrcf00027_repro_evt2.fits"
print(f"File exists: {os.path.exists(evtfile)}")
print(f"File readable: {os.access(evtfile, os.R_OK)}")

# Try with absolute path if relative fails
evtfile = os.path.abspath(evtfile)
print(f"Trying with absolute path: {evtfile}")
print(f"Absolute path exists: {os.path.exists(evtfile)}")
print("============================\n")

# Input and output file names
input_file = "b1_coord.txt"
xyregfile = "b1_xray.xyreg"
xyoffxfile = "b1_xray.xyoffx"

# Parse the input file
circles = parse_circle_file(input_file)
ra = [circle[0] for circle in circles]
dec = [circle[1] for circle in circles]
psfsiz = [circle[2] for circle in circles]

try:
    dm_xyreg_to_file(ra, dec, psfsiz, xyregfile, evtfile)
    dm_xyoffx_to_file(ra, dec, evtfile, xyoffxfile)
    print(f"Successfully created {xyregfile} and {xyoffxfile}")
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nAdditional troubleshooting steps:")
    print("1. Verify the FITS file isn't corrupted:")
    print("   > dmstat hrcf00027_repro_evt2.fits")
    print("2. Check CIAO environment is properly initialized")
    print("3. Try running dmcoords manually:")
    print(f"   > dmcoords {evtfile} option=sky ra=0 dec=0")