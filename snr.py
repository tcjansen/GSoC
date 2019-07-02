import numpy as np
import astropy.units as u


def snr(counts, npix=1, n_background=np.inf, background=0, darkcurrent=0,
        readnoise=0, gain=0, ad_err=0.289):
    """
    A function to calculate the signal to noise ratio (SNR) of an
    astronomical observation (from "Handbook of CCD Astronomy",
    Steve Howell, 2000, pg 55).

    Parameters
    ----------
    counts : float or int
        Total number of counts in an arbitrary exposure time.
    npix : float or int, optional
        Number of pixels under consideration for the signal. Default is 1.
    n_background : float or int, optional
        Number of pixels used in the background estimation. Default
        is set to infinite such that there is no contribution of
        error due to background estimation. This assumes that the
        n_background will be >> npix.
    background : float or int, optional
        Total photons per pixel due to the backround/sky. Default is 0.
    darkcurrent : float or int, optional
        Total electrons per pixel due to the dark current. Default is 0.
    readnoise : float or int, optional
        Electrons per pixel from the read noise. Default is 0.
    gain : float or int, optional
        Gain of the CCD in electrions/ADU. Default is 0 such that the
        contribution to the error due to the gain is assumed to be small.
    ad_err : float or int, optional
        An estimate of the 1 sigma error within the A/D converter.
        Default is set to 0.289 (Merline & Howell, 1995).

    Returns
    -------
    sn : float
        The signal to noise ratio of the given observation in sqrt(e-).
    """
    args = locals()  # maps arguments into a dictionary

    # convert arguments to quantities if not already of that type
    for arg in args:
        if type(args[arg]) != u.quantity.Quantity:
            args[arg] = args[arg] * get_unit(arg)

    readnoise = get_shotnoise(args['readnoise'])
    gain_err = get_shotnoise((args['gain'] * args['ad_err']))

    pixel_terms = args['npix'] * (1 + args['npix'] / args['n_background'])
    detector_noise = args['background'] + args['darkcurrent'] + \
        readnoise ** 2 + gain_err ** 2

    sn = args['counts'] / np.sqrt(args['counts'] +
                                  pixel_terms * detector_noise)

    return sn


def get_unit(arg):
    """ Converts int/float arguments into quantities with associated units.
    """
    if arg == 'counts':
        return u.electron
    elif arg in ['npix', 'n_background']:
        return u.pixel
    elif arg in ['background', 'darkcurrent', 'readnoise']:
        return u.electron / u.pixel
    elif arg == 'gain':
        return u.electron / u.adu
    elif arg == 'ad_err':
        return u.adu / u.pixel


def get_shotnoise(arg):
    """ Returns the shot noise (i.e. non-Poissonion noise) in the correct
    units.
    """
    return arg.value ** 2 * np.sqrt(1 * u.electron / u.pixel)


def test_units():
    """ A test to check that snr() returns a quantity given
    non-quantity arguments.
    Based on the worked example in "A Handbook to CCD Astronomy",
    Steven Howell, 2000, pg. 56-57
    """
    t = 300
    readnoise = 5
    darkcurrent = 22 * t / 60 / 60
    gain = 5
    background = 620 * gain
    n_background = 200
    npix = 1
    counts = 24013 * gain

    result = snr(counts, npix=npix, background=background,
                 darkcurrent=darkcurrent, readnoise=readnoise,
                 gain=gain, n_background=n_background)

    # the value of the answer given in the text
    answer = 342 * np.sqrt(1 * u.electron)

    # allow error to be within 1 sigma for this test
    assert abs(result - answer) < np.sqrt(1 * u.electron)


def test_math():
    """ A test to check that the math in snr() is done correctly.
    Based on the worked example in "A Handbook to CCD Astronomy",
    Steven Howell, 2000, pg. 56-57
    """
    t = 300 * u.s
    readnoise = 5 * u.electron
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

    # allow error to be within 1 sigma for this test
    assert abs(result - answer) < np.sqrt(1 * u.electron)


def test_bright():
    """ Test that snr() returns sqrt(counts), the expected value
    for a bright target.
    """
    counts = 25e5
    answer = np.sqrt(counts * u.electron)
    result = snr(counts)
    difference_threshold = 1e-3 * np.sqrt(1 * u.electron)
    assert abs(result - answer) < difference_threshold
