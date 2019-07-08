import numpy as np
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose


@u.quantity_input(counts=u.electron,
                  npix=u.pixel,
                  n_background=u.pixel,
                  background=u.electron / u.pixel,
                  darkcurrent=u.electron / u.pixel,
                  readnoise=u.electron / u.pixel,
                  gain=u.electron / u.adu,
                  ad_err=u.adu / u.pixel)
def snr(counts,
        npix=1 * u.pixel,
        n_background=np.inf * u.pixel,
        background=0 * u.electron / u.pixel,
        darkcurrent=0 * u.electron / u.pixel,
        readnoise=0 * u.electron / u.pixel,
        gain=1 * u.electron / u.adu,
        ad_err=0.289 * u.adu / u.pixel):
    """
    A function to calculate the signal to noise ratio (SNR) of an
    astronomical observation with a given number of counts (from
    "Handbook of CCD Astronomy", Steve Howell, 2000, pg 55).

    Parameters
    ----------
    counts : `~astropy.units.Quantity`
        Total number of counts in an arbitrary exposure time with units of
        electrons.
    npix : `~astropy.units.Quantity`, optional
        Number of pixels under consideration for the signal with units of
        pixels. Default is 1 * astropy.units.pixel.
    n_background : `~astropy.units.Quantity`, optional
        Number of pixels used in the background estimation with units of
        pixels. Default is set to np.inf * astropy.units.pixel
        such that there is no contribution of error due to background
        estimation. This assumes that n_background will be >> npix.
    background : `~astropy.units.Quantity`, optional
        Total photons per pixel due to the background/sky with units of
        electrons/pixel.
        Default is 0 * astropy.units.electron / astropy.units.pixel.
    darkcurrent : `~astropy.units.Quantity`, optional
        Total electrons per pixel due to the dark current with units
        of electrons/pixel.
        Default is 0 * astropy.units.electron / astropy.units.pixel.
    readnoise : `~astropy.units.Quantity`, optional
        Electrons per pixel from the read noise with units of electrons/pixel.
        Default is 0 * astropy.units.electron / astropy.units.pixel.
    gain : `~astropy.units.Quantity`, optional
        Gain of the CCD with units of electrons/ADU. Default is
        1 * astropy.units.electron / astropy.units.adu such that the
        contribution to the error due to the gain is assumed to be small.
    ad_err : `~astropy.units.Quantity`, optional
        An estimate of the 1 sigma error within the A/D converter with units of
        adu/pixel. Default is set to
        0.289 * astropy.units.adu / astropy.units.pixel
        (Merline & Howell, 1995).

    Returns
    -------
    sn : `~astropy.units.Quantity`
        The signal to noise ratio of the given observation in untis of
        sqrt(electrons).
    """
    readnoise = get_shotnoise(readnoise)
    gain_err = get_shotnoise(gain * ad_err)

    pixel_terms = npix * (1 + npix / n_background)
    detector_noise = (background + darkcurrent +
                      readnoise ** 2 + gain_err ** 2)

    sn = counts / np.sqrt(counts + pixel_terms * detector_noise)

    return sn


def get_shotnoise(detector_property):
    """
    Returns the shot noise (i.e. non-Poissonion noise) in the correct
    units.
    """
    return detector_property.value ** 2 * np.sqrt(1 * u.electron / u.pixel)


def exposure_time_from_snr(snr,
                           npix=1 * u.pixel,
                           n_background=np.inf * u.pixel,
                           background=0 * u.electron / u.pixel,
                           darkcurrent=0 * u.electron / u.pixel,
                           readnoise=0 * u.electron / u.pixel,
                           gain=1 * u.electron / u.adu,
                           ad_err=0.289 * u.adu / u.pixel):
    """
    Returns the exposure time needed (in seconds) to achieve the desired
    signal to noise ratio (from "Handbook of CCD Astronomy", Steve Howell,
    2000, pg 55).

    Parameters
    ----------
    snr : `~astropy.units.Quantity`
        The signal to noise ratio of the given observation in untis of
        sqrt(electrons).
    npix : `~astropy.units.Quantity`, optional
        Number of pixels under consideration for the signal with units of
        pixels. Default is 1 * astropy.units.pixel.
    n_background : `~astropy.units.Quantity`, optional
        Number of pixels used in the background estimation with units of
        pixels. Default is set to np.inf * astropy.units.pixel
        such that there is no contribution of error due to background
        estimation. This assumes that n_background will be >> npix.
    background : `~astropy.units.Quantity`, optional
        Total photons per pixel due to the backround/sky with units of
        electrons/pixel.
        Default is 0 * astropy.units.electron / astropy.units.pixel.
    darkcurrent : `~astropy.units.Quantity`, optional
        Total electrons per pixel due to the dark current with units
        of electrons/pixel.
        Default is 0 * astropy.units.electron / astropy.units.pixel.
    readnoise : `~astropy.units.Quantity`, optional
        Electrons per pixel from the read noise with units of electrons/pixel.
        Default is 0 * astropy.units.electron / astropy.units.pixel.
    gain : `~astropy.units.Quantity`, optional
        Gain of the CCD with units of electrons/ADU. Default is
        1 * astropy.units.electron / astropy.units.adu such that the
        contribution to the error due to the gain is assumed to be small.
    ad_err : `~astropy.units.Quantity`, optional
        An estimate of the 1 sigma error within the A/D converter with units of
        adu/pixel. Default is set to
        0.289 * astropy.units.adu / astropy.units.pixel
        (Merline & Howell, 1995).

    Returns
    -------
    t : `~astropy.units.Quantity`
        The exposure time needed (in seconds) to achieve the given signal
        to noise ratio.
    """


def test_snr_calc():
    """
    A test to check that the math in snr() is done correctly.
    Based on the worked example in "A Handbook to CCD Astronomy",
    Steven Howell, 2000, pg. 56-57
    """
    t = 300 * u.s
    readnoise = 5 * u.electron / u.pixel
    darkcurrent = 22 * u.electron / u.pixel / u.hr
    darkcurrent = (darkcurrent * t).to(u.electron / u.pixel)
    gain = 5 * u.electron / u.adu
    background = 620 * u.adu / u.pixel * gain
    n_background = 200 * u.pixel
    npix = 1 * u.pixel
    counts = 24013 * u.adu * gain

    result = snr(counts, npix=npix, background=background,
                 darkcurrent=darkcurrent, readnoise=readnoise,
                 gain=gain, n_background=n_background)

    # the value of the answer given in the text
    answer = 342 * np.sqrt(1 * u.electron)

    # allow error to be +/- 1 for this test
    assert_quantity_allclose(result, answer, atol=1)


def test_snr_bright_object():
    """
    Test that snr() returns sqrt(counts), the expected value
    for a bright target.
    """
    counts = 25e5 * u.electron
    result = snr(counts)
    answer = np.sqrt(counts)

    assert_quantity_allclose(result, answer, rtol=1e-7)
