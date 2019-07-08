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
        ad_err=np.sqrt(0.289) * u.adu / u.pixel):
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
    return detector_property.value * np.sqrt(1 * u.electron / u.pixel)


@u.quantity_input(snr=np.sqrt(1 * u.electron),
                  countrate=u.electron / u.s,
                  npix=u.pixel,
                  n_background=u.pixel,
                  background=u.electron / u.pixel / u.s,
                  darkcurrent=u.electron / u.pixel / u.s,
                  readnoise=u.electron / u.pixel,
                  gain=u.electron / u.adu,
                  ad_err=u.adu / u.pixel)
def exposure_time_from_snr(snr, countrate,
                           npix=1 * u.pixel,
                           n_background=np.inf * u.pixel,
                           background=0 * u.electron / u.pixel / u.s,
                           darkcurrent=0 * u.electron / u.pixel / u.s,
                           readnoise=0 * u.electron / u.pixel,
                           gain=1 * u.electron / u.adu,
                           ad_err=np.sqrt(0.289) * u.adu / u.pixel):
    """
    Returns the exposure time needed (in seconds) to achieve the desired
    signal to noise ratio (from "Handbook of CCD Astronomy", Steve Howell,
    2000, pg 55).

    Parameters
    ----------
    snr : `~astropy.units.Quantity`
        The signal to noise ratio of the given observation in untis of
        sqrt(electrons).
    countrate : `~astropy.units.Quantity`
        The counts per second with units of electron/second.
    npix : `~astropy.units.Quantity`, optional
        Number of pixels under consideration for the signal with units of
        pixels. Default is 1 * astropy.units.pixel.
    n_background : `~astropy.units.Quantity`, optional
        Number of pixels used in the background estimation with units of
        pixels. Default is set to np.inf * astropy.units.pixel
        such that there is no contribution of error due to background
        estimation. This assumes that n_background will be >> npix.
    background : `~astropy.units.Quantity`, optional
        Photons per pixel per second due to the backround/sky with units of
        electrons/second/pixel.
        Default is 0 * (astropy.units.electron /
                        astropy.units.second / astropy.units.pixel)
    darkcurrent : `~astropy.units.Quantity`, optional
        Electrons per pixel per second due to the dark current with units
        of electrons/second/pixel.
        Default is 0 * (astropy.units.electron /
                        astropy.units.second / astropy.units.pixel)
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
    readnoise = get_shotnoise(readnoise)
    gain_err = get_shotnoise(gain * ad_err)

    # how to define the moment when the smaller error terms become
    # non-negligible?
    if gain_err.value > 1 or n_background.value != np.inf:
        t = solve_t_numerically()

    else:
        # solve t with the quadratic equation (pg. 57 of Howell 2000)
        A = countrate ** 2
        B = (-1) * snr ** 2 * (countrate + npix * (background + darkcurrent))
        C = (-1) * snr ** 2 * npix * readnoise ** 2

        t = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    return t


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
    assert_quantity_allclose(result, answer, atol=1 * np.sqrt(1 * u.electron))


def test_snr_bright_object():
    """
    Test that snr() returns sqrt(counts), the expected value
    for a bright target.
    """
    counts = 25e5 * u.electron
    result = snr(counts)
    answer = np.sqrt(counts)

    assert_quantity_allclose(result, answer, rtol=1e-7)


def test_t_exp_numeric():
    """
    A test to check that the numerical method in exposure_time_from_snr()
    (i.e. when the error in background noise or gain is non-negligible) is
    done correctly. Based on the worked example in "A Handbook to
    CCD Astronomy", Steven Howell, 2000, pg. 56-57
    """
    t = 300 * u.s  # answer from Howell 2000
    snr = 342 * np.sqrt(1 * u.electron)
    gain = 5 * u.electron / u.adu
    countrate = 24013 * u.adu * gain / t
    npix = 1 * u.pixel
    n_background = 200 * u.pixel
    background = 620 * u.adu / u.pixel * gain / t
    darkcurrent = 22 * u.electron / u.pixel / u.hr
    darkcurrent = darkcurrent.to(u.electron / u.pixel / u.s)
    readnoise = 5 * u.electron / u.pixel

    result = exposure_time_from_snr(snr, countrate, npix=npix,
                                    n_background=n_background,
                                    background=background,
                                    darkcurrent=darkcurrent,
                                    readnoise=readnoise, gain=gain)
    answer = t

    assert_quantity_allclose(result, answer, rtol=1e-7)


def test_t_exp_analytic():
    """
    A test to check that the analytic method in exposure_time_from_snr() is
    done correctly. Based on the worked example in "A Handbook to
    CCD Astronomy", Steven Howell, 2000, pg. 56-57
    """
    snr_set = 50 * np.sqrt(1 * u.electron)
    countrate = 1000 * u.electron / u.s
    npix = 1 * u.pixel
    background_rate = 100 * u.electron / u.pixel / u.s
    darkcurrent_rate = 5 * u.electron / u.pixel / u.s
    darkcurrent_rate = darkcurrent_rate.to(u.electron / u.pixel / u.s)
    readnoise = 1 * u.electron / u.pixel

    t = exposure_time_from_snr(snr_set, countrate, npix=npix,
                               background=background_rate,
                               darkcurrent=darkcurrent_rate,
                               readnoise=readnoise)
    snr_calc = snr(countrate * t,
        npix=npix,
        background=background_rate * t,
        darkcurrent=darkcurrent_rate * t,
        readnoise=readnoise)

    assert_quantity_allclose(snr_calc, snr_set,
                             atol=0.5 * np.sqrt(1 * u.electron))
