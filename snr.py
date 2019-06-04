import numpy as np


def snr(count, t=None, npix=1, sky=0, darkcurrent=0, readnoise=0,
        gain=0, nb=np.inf, ADerr=0.289):
    """ A function to calculate the signal to noise ratio (SNR) of an
    astronomical observation.

    Parameters
    ----------
    count (float or int)
        Total number of counts in the observation. If t is given,
        count is assumed to be the countrate.
    t (float)
        Exposure time of the observation. Default is None.
    countrate (float or int)
        Photons per second of the observation. Default is None.
    npix (int)
        Number of pixels under consideration for the signal. Default is 1.
    sky (int)
        Photons per pixel due to the backround/sky. Default is 0.
    darkcurrent (int)
        Electrons per pixel due to the dark current. Default is 0.
    readnoise (int)
        Electrons per pixel from the read noise. Default is 0.
    nb (int)
        Number of pixels used in the background estimation. Default
        is set to infinite such that there is no contribution of
        error due to background estimation. This assumes that the
        nb will be >> npix.
    gain (int)
        Gain of the CCD in electrions/ADU. Default is 0 such that the
        contribution to the error due to the gain is assumed to be small.
    ADerr (float)
        An estimate of the 1 sigma error within the A/D converter.
        Default is set to 0.289 (Merline & Howell, 1995).

    Returns
    -------
    float
    The signal to noise ratio of the given observation.
    """
    signal = count

    if t:
        # calculate the SNR in a given exposure time t
        signal *= t
        sky *= t
        darkcurrent *= t

    return signal / np.sqrt(signal + npix *
                            (1 + npix / nb) *
                            (sky + darkcurrent + readnoise ** 2 +
                             (gain * ADerr) ** 2)
                            )
