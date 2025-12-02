import pathlib
from typing import Optional

from astropy import units as u

from .sources import HiResFitsSource, InterpolationMode, SVOSource, SynphotSource, find_source


def get_spectrum(
    *,
    teff: float,
    logg: float,
    feh: float,
    alpha: float,
    source: str = "synphot",
    interpolation_mode: InterpolationMode = InterpolationMode.LINEAR,
    use_planck: bool = False,
    bounds_error: bool = False,
    path: Optional[pathlib.Path] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs,
) -> tuple[u.Quantity, u.Quantity]:
    """All in one loader for Phoenix spectra.

    Will load the specified source, download files if necessary, and return the spectrum for
    the specified parameters.

    For example:

    ```python
    from phoenix4all import get_spectrum

    wave, flux = get_spectrum(teff=5800, logg=4.5, feh=0.0, alpha=0.0)
    ```

    Args:
        teff: Effective temperature in Kelvin.
        logg: Surface gravity in cgs.
        feh: Metallicity [Fe/H].
        alpha: Alpha element enhancement [alpha/Fe].
        source: The source to use for the Phoenix models. Default is "synphot".
        interpolation_mode: The interpolation mode to use. Default is linear.
        use_planck: If True, use Planck function when out of temperature bounds.
        bounds_error: If True, raise an error if the requested parameters are out of bounds.
        path: Optional path to the local directory containing the model files.
        base_url: Optional base URL for downloading model files if not found locally.
        model_name: Optional model name to use when downloading files.

    Returns:
        A tuple of (wavelength, flux) as astropy Quantities.
    """
    source_klass = find_source(source)
    phoenix_source = source_klass(
        path=path, base_url=base_url, model_name=model_name, interpolation_mode=interpolation_mode, **kwargs
    )

    return phoenix_source.spectrum(
        teff=teff, logg=logg, feh=feh, alpha=alpha, use_planck=use_planck, bounds_error=bounds_error
    )


__all__ = ["HiResFitsSource", "InterpolationMode", "SVOSource", "SynphotSource", "find_source", "get_spectrum"]
