from bilinear_interpolation import bi_interpolation_psfsize
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import ascii, fits
radec_names = [
    "27_b1.wsrc.radec"
]
fits_names = [
    "27_b1_psf.fits"
]

coord_files = [
    "b1_coord.txt"
]

for (radec_name, coord_file, fits_name) in zip(radec_names, coord_files, fits_names):
    radec = ascii.read(radec_name, names = ["ra", "dec"])
    N = len(radec['ra'])
    print('N = ', N)
    sc = SkyCoord(ra=radec['ra'] * u.degree, dec=radec['dec'] * u.degree)
    # Open the fits file and extract WCS
    f = fits.open(fits_name)
    w = wcs.WCS(f[0].header)
    # Compute coordinates using WCS
    coord = wcs.utils.skycoord_to_pixel(sc, w)
    psf_size = np.array([bi_interpolation_psfsize(f[0].data, coord[0][j], coord[1][j]) for j in range(N)])
    with open(coord_file, 'w') as file:
        for i in range(N):
            result = 'circle(' + str(radec['ra'][i]) + 'd,' + str(radec['dec'][i]) + 'd,' + str(psf_size[i]) + '\'\')'
            file.write(result + '\n')
