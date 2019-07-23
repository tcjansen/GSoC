from synphot.observation import Observation
import astropy.units as u
import numpy as np


def from_R(source_spectrum, bandpass, R, waverange, force='none'):
    """
    Returns a spectrum with bins defined by a spectral resolution
    R and a given range of wavelengths. The bin width varies over the
    range of wavelengths such that the ratio of the wavelength to the
    bin width at that wavelength is equal to the given R.
    """
    first_wl = waverange[0].to(u.angstrom).value
    binset = np.array([first_wl])
    binwidth = first_wl / R

    while binset[-1] <= waverange[-1].to(u.angstrom).value:
        next_wl = binset[-1] + binwidth / 2
        binset = np.append(binset, next_wl)
        binwidth = next_wl / R

    return Observation(source_spectrum, bandpass, binset=binset * u.angstrom,
                       force=force)


def from_wave_binwidth(source_spectrum, bandpass, binwidth, waverange=None,
                       force='none'):
    """
    Returns a spectrum with bins defined by a constant bin width specified
    by the user.
    """
    if waverange is None:
        waverange = bandpass.waveset

    # make sure units are the same before stripping them for np.arange()
    waverange = waverange.to(u.angstrom).value
    binwidth = binwidth.to(u.angstrom).value

    binset = (np.arange(waverange[0], waverange[-1] + binwidth, binwidth) *
              u.angstrom)

    return Observation(source_spectrum, bandpass, binset=binset, force=force)


def from_wave_array(source_spectrum, bandpass, waveset, force='none'):
    """
    Returns a spectrum with bins defined by the wavelength array provided
    by the user.
    """
    return Observation(source_spectrum, bandpass, binset=waveset, force=force)
