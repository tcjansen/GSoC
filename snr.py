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
    # convert arguments to quantities if not already of that type
    args = locals()  # maps arguments into a dictionary

    for arg in args:
        if type(args[arg]) != u.quantity.Quantity \
           and args[arg] is not None:
            args[arg] = args[arg] * get_unit(arg)

    if t:
        # calculate the SNR in a given exposure time t
        args['count'] = args['count'] * args['t']
        args['background'] *= args['background'] * args['t']
        args['darkcurrent'] *= args['darkcurrent'] * args['t']

    readnoise = args['readnoise'].value ** 2 * \
        np.sqrt(1 * u.electron / u.pixel)
    gain_err = (args['gain'] * args['ADerr']).value ** 2 * \
        np.sqrt(1 * u.electron / u.pixel)

    return args['count'] / np.sqrt(args['count'] + args['npix'] *
                                   (1 + args['npix'] / args['nb']) *
                                   (args['background'] + args['darkcurrent'] +
                                    readnoise ** 2 +
                                    gain_err ** 2)
                                   )


def get_unit(arg):
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


def howell_test():
    """ A test based on the worked example in "A Handbook to CCD Astronomy",
    Steven Howell, 2000, pg. 56
    """
    t = 300
    readnoise = 5
    darkcurrent = 22
    gain = 5
    nb = 200
    background = 620
    npix = 1
    count = 24013

    result = snr(count * gain, npix=npix, background=background * gain,
                 darkcurrent=darkcurrent * t / 60 / 60, readnoise=readnoise,
                 gain=gain, nb=nb)

    # the value of the answer given in the text
    answer = 342 * np.sqrt(1 * u.electron)

    # allow error to be within 1 sigma for this test
    assert abs(result - answer) < np.sqrt(1 * u.electron)


def units_test():
    """ A test based on the worked example in "A Handbook to CCD Astronomy",
    Steven Howell, 2000, pg. 56
    """
    t = 300 * u.s
    readnoise = 5 * u.electron
    darkcurrent = 22 * u.electron / u.pixel / u.hr
    darkcurrent = (darkcurrent * t).to(u.electron / u.pixel)
    gain = 5 * u.electron / u.adu
    nb = 200 * u.pixel
    background = 620 * u.adu / u.pixel
    npix = 1 * u.pixel
    count = 24013 * u.adu

    # the value of the answer given in the text
    result = snr(count * gain, npix=npix, background=background * gain,
                 darkcurrent=darkcurrent, readnoise=readnoise,
                 gain=gain, nb=nb)
    answer = 342 * np.sqrt(1 * u.electron)

    # allow error to be within 1 sigma for this test
    assert abs(result - answer) < np.sqrt(1 * u.electron)
