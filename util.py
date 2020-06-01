import numpy as np

__all__ = ['aperture']

def aperture(npix=256, cent_obs=0.0, spider=0, overfill=1.0):
    """
    Compute the aperture image of a telescope
  
    Args:
        npix (int, optional): number of pixels of the aperture image
        cent_obs (float, optional): central obscuration fraction
        spider (int, optional): spider size in pixels
    
    Returns:
        real: returns the aperture of the telescope
    """
    illum = np.ones((npix,npix),dtype='d')
    x = np.arange(-npix/2,npix/2,dtype='d')
    y = np.arange(-npix/2,npix/2,dtype='d')

    xarr = np.outer(np.ones(npix,dtype='d'),x)
    yarr = np.outer(y,np.ones(npix,dtype='d'))

    rarr = np.sqrt(np.power(xarr,2) + np.power(yarr,2))/(npix/2)
    outside = np.where(rarr > 1.0/overfill)
    inside = np.where(rarr < cent_obs)

    illum[outside] = 0.0
    if np.any(inside[0]):
        illum[inside] = 0.0

    if (spider > 0):
        start = int(npix/2 - int(spider)/2)
        illum[start:start+int(spider),:] = 0.0
        illum[:,start:start+int(spider)] = 0.0

    return illum

def psf_scale(wavelength, telescope_diameter, simulation_pixel_size):
        """
        Return the PSF scale appropriate for the required pixel size, wavelength and telescope diameter
        The aperture is padded by this amount; resultant pix scale is lambda/D/psf_scale, so for instance full frame 256 pix
        for 3.5 m at 532 nm is 256*5.32e-7/3.5/3 = 2.67 arcsec for psf_scale = 3

        https://www.strollswithmydog.com/wavefront-to-psf-to-mtf-physical-units/#iv
                
        """
        return 206265.0 * wavelength * 1e-8 / (telescope_diameter * simulation_pixel_size)