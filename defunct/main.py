import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import ascii, fits

b32_event = fits.open('../repro/27_b32.fits')
b16_event = fits.open('../repro/27_b16.fits')
b08_event = fits.open('../repro/27_b08.fits')
b08_odegap_event = fits.open('../repro/27_b08_odegap.fits')

b32_psf = fits.open('../repro/27_b32_psf.fits')
b16_psf = fits.open('../repro/27_b16_psf.fits')
b08_psf = fits.open('../repro/27_b08_psf.fits')
b08_odegap_psf = fits.open('../repro/27_b08_odegap_psf.fits')




b32_event[0]