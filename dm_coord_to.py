import subprocess
from ciao_contrib.runtool import dmcoords
import os
os.environ['PARAM_FILE'] = '/Users/jl1023/cxcds_param4/dmcoords.par'

def dm_radec_to(evtfile, input_file, output_file, asolfil=None, tdet=False, chip=False, det=False):
    """
    Wrapper to run dmcoords on input RA, Dec and compute physical and detector coordinates
    and off-axis angles for a specified event file.

    Parameters:
        evtfile (str): Path to the event file.
        asolfil (str): Path to the aspect solution file.
        input_file (str): Path to the input file containing RA and Dec values.
        output_file (str): Path to the output file for storing results.
        tdet (bool): If True, include TDETX and TDETY in the output.
        chip (bool): If True, include CHIPX and CHIPY in the output.
        det (bool): If True, include DETX and DETY in the output.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Read RA and Dec
            rr, dd = line.strip().split()

            # Run dmcoords
            dmcoords(infile=evtfile, asolfile=asolfil, opt="cel", ra=rr, dec=dd)

            # Get coordinates
            xx = dmcoords.x
            yy = dmcoords.y
            theta = dmcoords.theta
            phi = dmcoords.phi
            tdetx = dmcoords.tdetx
            tdety = dmcoords.tdety
            chipx = dmcoords.chipx
            chipy = dmcoords.chipy
            detx = dmcoords.detx
            dety = dmcoords.dety

            # Write results to output file
            result = f"{xx}\t{yy}\t{theta}\t{phi}"
            if tdet:
                result += f"\t{tdetx}\t{tdety}"
            if chip:
                result += f"\t{chipx}\t{chipy}"
            if det:
                result += f"\t{detx}\t{dety}"
            outfile.write(result + "\n")


def dm_xy_to(evtfile, input_file, output_file, tdet=False, chip=False, det=False):
    """
    Wrapper to run dmcoords on input sky pixel coordinates and compute celestial
    and detector coordinates and off-axis angles for a specified event file.

    Parameters:
        evtfile (str): Path to the event file.
        input_file (str): Path to the input file containing X and Y values.
        output_file (str): Path to the output file for storing results.
        tdet (bool): If True, include TDETX and TDETY in the output.
        chip (bool): If True, include CHIPX and CHIPY in the output.
        det (bool): If True, include DETX and DETY in the output.
    """
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Read X and Y
            xxyy = line.strip()
            if not xxyy:
                continue  # Skip empty lines

            # Split X and Y
            xx, yy = xxyy.split()
            xx = float(xx)
            yy = float(yy)

            # Skip if X or Y is <= 0
            if xx <= 0 or yy <= 0:
                continue

            # Run dmcoords
            dmcoords(infile=evtfile, opt="sky", x=xx, y=yy)

            # Get coordinates
            ra = dmcoords.ra
            dec = dmcoords.dec
            theta = dmcoords.theta
            phi = dmcoords.phi
            tdetx = dmcoords.tdetx
            tdety = dmcoords.tdety
            chipx = dmcoords.chipx
            chipy = dmcoords.chipy
            detx = dmcoords.detx
            dety = dmcoords.dety

            # Write results to output file
            result = f"{ra}\t{dec}\t{theta}\t{phi}"
            if tdet:
                result += f"\t{tdetx}\t{tdety}"
            if chip:
                result += f"\t{chipx}\t{chipy}"
            if det:
                result += f"\t{detx}\t{dety}"
            outfile.write(result + "\n")
