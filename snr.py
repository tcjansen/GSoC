import numpy as np
import astropy.units as u


def snr(count, t=None, npix=1, nb=np.inf, background=0, darkcurrent=0,
        readnoise=0, gain=0, ADerr=0.289):
    """ A function to calculate the signal to noise ratio (SNR) of an
    astronomical observation.

    Parameters
    ----------
    count (float or int)
        Total number of counts in the observation. If t is given,
        count is assumed to be the countrate.
    t (float)
        Exposure time of the observation. Default is None.
    npix (int)
        Number of pixels under consideration for the signal. Default is 1.
    nb (int)
        Number of pixels used in the background estimation. Default
        is set to infinite such that there is no contribution of
        error due to background estimation. This assumes that the
        nb will be >> npix.
    background (int)
        Photons per pixel due to the backround/sky. Default is 0.
    darkcurrent (int)
        Electrons per pixel due to the dark current. Default is 0.
    readnoise (int)
        Electrons per pixel from the read noise. Default is 0.
    gain (int)
        Gain of the CCD in electrions/ADU. Default is 0 such that the
        contribution to the error due to the gain is assumed to be small.
    ADerr (float)
        An estimate of the 1 sigma error within the A/D converter.
        Default is set to 0.289 (Merline & Howell, 1995).

    Returns
    -------
    float
    The signal to noise ratio of the given observation in sqrt(e-).
    """
    args = locals()  # maps arguments into a dictionary

    # convert arguments to quantities if not already of that type
    for arg in args:
        if type(args[arg]) != u.quantity.Quantity \
           and args[arg] is not None:
            args[arg] = args[arg] * get_unit(arg)

    # calculate the SNR in a given exposure time t
    if t:
        for arg in ['count', 'background', 'darkcurrent']:
            args[arg] = args[arg] * args['t']

    readnoise = get_shotnoise(args['readnoise'])
    gain_err = get_shotnoise((args['gain'] * args['ADerr']))

    pixel_terms = args['npix'] * (1 + args['npix'] / args['nb'])
    detector_noise = args['background'] + args['darkcurrent'] + \
        readnoise ** 2 + gain_err ** 2

    return args['count'] / np.sqrt(args['count'] +
                                   pixel_terms * detector_noise)


def get_unit(arg):
    """ Converts int/float arguments into quantities with associated units.
    """
    if arg == 'count':
        return u.electron
    elif arg == 't':
        return u.s
    elif arg in ['npix', 'nb']:
        return u.pixel
    elif arg in ['background', 'darkcurrent', 'readnoise']:
        return u.electron / u.pixel
    elif arg == 'gain':
        return u.electron / u.adu
    elif arg == 'ADerr':
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
    nb = 200
    background = 620 * gain
    npix = 1
    count = 24013 * gain

    result = snr(count, npix=npix, background=background,
                 darkcurrent=darkcurrent, readnoise=readnoise,
                 gain=gain, nb=nb)

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
    nb = 200 * u.pixel
    background = 620 * u.adu / u.pixel * gain
    npix = 1 * u.pixel
    count = 24013 * u.adu * gain

    result = snr(count, npix=npix, background=background,
                 darkcurrent=darkcurrent, readnoise=readnoise,
                 gain=gain, nb=nb)

    # the value of the answer given in the text
    answer = 342 * np.sqrt(1 * u.electron)

    # allow error to be within 1 sigma for this test
    assert abs(result - answer) < np.sqrt(1 * u.electron)
