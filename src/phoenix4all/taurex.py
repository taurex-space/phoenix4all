import typing as t

import numpy as np
import numpy.typing as npt
from astropy import units as u
from taurex.stellar import Star

from phoenix4all import get_spectrum


class Phoenix4AllStar(Star):
    def __init__(
        self,
        temperature: float = 5000,
        radius: float = 1.0,
        distance: float = 1,
        magnitudeK: float = 10.0,
        mass: float = 1.0,
        metallicity: float = 1.0,
        alpha: float = 0.0,
        source: str = "svo",
        interpolation_mode: t.Optional[str] = "linear",
        use_planck: bool = True,
        bounds_error: bool = False,
        path: t.Optional[str] = None,
        model_name: t.Optional[str] = "bt-settl-cifist",
        logg: t.Optional[float] = None,
    ) -> None:
        super().__init__(
            temperature=temperature,
            radius=radius,
            distance=distance,
            magnitudeK=magnitudeK,
            mass=mass,
            metallicity=metallicity,
        )
        self.alpha = alpha
        self.logg_value = None if logg is None else logg

        self.get_spectrum_params = {
            "source": source,
            "interpolation_mode": interpolation_mode,
            "use_planck": use_planck,
            "bounds_error": bounds_error,
            "path": path,
            "model_name": model_name,
        }

    @property
    def logg(self):
        if self.logg_value is None:
            import math

            import astropy.units as u
            from astropy.constants import G

            mass = self._mass * u.kg
            radius = self._radius * u.m

            small_g = (G * mass) / (radius**2)

            small_g = small_g.to(u.cm / u.s**2)

            return math.log10(small_g.value)
        else:
            return self.logg_value

    @property
    def metallicity(self):
        return self._metallicity

    def initialize(self, wngrid: npt.NDArray[np.float64]) -> None:
        """Initializes the blackbody spectrum on the given wavenumber grid

        Parameters
        ----------
        wngrid: :obj:`array`
            Wavenumber grid cm-1 to compute black body spectrum

        """

        wlgrid, sed = get_spectrum(
            teff=self.temperature, logg=self.logg, feh=self.metallicity, alpha=self.alpha, **self.get_spectrum_params
        )
        current_wngrid = wlgrid.to(u.k, equivalencies=u.spectral()).value

        sort_wngrid_indices = np.argsort(current_wngrid)
        current_wngrid = current_wngrid[sort_wngrid_indices]
        sed = sed[sort_wngrid_indices]

        sed_taurex = sed.to(u.W / u.m**2 / u.um)

        sed_interp = np.interp(wngrid, current_wngrid, sed_taurex.value, left=0.0, right=0.0)

        self.sed = sed_interp

    @classmethod
    def input_keywords(self) -> tuple[str, ...]:
        return ("phoenix4all",)
