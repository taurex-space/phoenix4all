import numpy as np
from astropy import constants as const
from astropy import units as u


def planck(spectrum: u.Quantity, temperature: u.Quantity) -> u.Quantity:
    """Calculate the Planck function for a given wavelength and temperature.


    Calculation will use wavelength form:

    $$
    B_\\lambda(\\lambda, T) = \\frac{2hc^2}{\\lambda^5} \\frac{1}{e^{\\frac{hc}{\\lambda k_B T}} - 1}
    $$


    Parameters:
        wavelength (u.Quantity): Wavelength(s) at which to calculate the Planck function. Must have units of length.
        temperature (u.Quantity): Temperature(s) at which to calculate the Planck function. Must have units of temperature.

    Returns:
        u.Quantity: The Planck function values at the specified wavelengths and temperatures, in arbitrary units of spectral radiance (e.g., erg / (s cm^2 A sr)).
    """

    wavelength = spectrum.to(u.m, equivalencies=u.spectral())
    temperature = temperature.to(u.K)

    factor1 = (2 * const.h * const.c**2) / wavelength**5
    exponent = (const.h * const.c) / (wavelength * const.k_B * temperature)
    factor2 = 1 / (np.exp(exponent) - 1)

    B_lambda = np.pi * factor1 * factor2
    return B_lambda.to(u.erg / (u.s * u.cm**2 * u.AA), equivalencies=u.spectral_density(wavelength))
