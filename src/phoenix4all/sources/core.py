import abc
import enum
import pathlib
from dataclasses import asdict, dataclass
from typing import Callable, Literal, Optional

import numpy as np
import pandas as pd
from astropy import units as u

from ..log import debug_function
from ..radiative import planck


class InterpolationMode(str, enum.Enum):
    NEAREST = "nearest"
    LINEAR = "linear"


class UnknownInterpolationModeError(ValueError):
    def __init__(self, mode: str):
        super().__init__(f"Unknown interpolation mode: {mode}")


class InterpolationBoundaryError(ValueError):
    def __init__(self, param: str, value: float, bounds: tuple[float, float]):
        super().__init__(f"Parameter {param} with value {value} is out of bounds {bounds}")


class NoAvailableDataError(ValueError):
    def __init__(self):
        super().__init__("No available data found for the specified parameters.")


class ModelNotFoundError(ValueError):
    def __init__(self, model_name: str):
        super().__init__(f"Model '{model_name}' not found in the available models.")


@dataclass
class PhoenixDataFile:
    """Class representing a Phoenix model data file with its parameters."""

    teff: int
    logg: float
    feh: float
    filename: str
    alpha: float


@dataclass
class WeightedPhoenixDataFile(PhoenixDataFile):
    """Class representing a Phoenix model data file with its parameters and interpolation weight."""

    weight: float


DataFileLoader = Callable[[PhoenixDataFile], tuple[u.Quantity, u.Quantity]]

FilterType = Optional[tuple[float, float] | float | u.Quantity | Literal["all"]]


@debug_function
def construct_phoenix_dataframe(datafile_list: list[PhoenixDataFile]) -> pd.DataFrame:
    """Construct a DataFrame from a list of PhoenixDataFile instances."""

    serialised_data = [asdict(datafile) for datafile in datafile_list]
    df = pd.DataFrame(serialised_data)
    # df.set_index(["teff", "logg", "feh", "alpha"], inplace=True)
    return df


@debug_function
def find_nearest_points(df: pd.DataFrame, teff: int, logg: float, feh: float, alpha: float = 0.0) -> pd.DataFrame:
    """Find the nearest grid points in the DataFrame to the specified parameters.

    Args:
        df: DataFrame with MultiIndex (teff, logg, feh, alpha) and a 'filename' column.
        teff: Effective temperature to match.
        logg: Surface gravity to match.
        feh: Metallicity to match.
        alpha: Alpha element enhancement to match (default is 0.0).
    Returns:
        DataFrame with the nearest grid points and their filenames.

    """
    # Since there will be multiple temperatures, we will progressively filter the DataFrame

    tvalues = df["teff"].unique()
    tclosest = tvalues[np.abs(tvalues - teff).argsort()[:2]]  # Get two closest temperatures
    if np.any(tclosest == teff):
        tclosest = np.array([teff])  # If exact match, only keep that
    df_t = df.loc[df["teff"].isin(tclosest)]

    gvalues = df_t["logg"].unique()
    gclosest = gvalues[np.abs(gvalues - logg).argsort()[:2]]  # Get two closest logg values
    if np.any(gclosest == logg):
        gclosest = np.array([logg])  # If exact match, only keep that
    df_g = df_t.loc[df_t["logg"].isin(gclosest)]
    # Now filter by feh
    fvalues = df_g["feh"].unique()
    fclosest = fvalues[np.abs(fvalues - feh).argsort()[:2]]  # Get two closest feh values
    if np.any(fclosest == feh):
        fclosest = np.array([feh])  # If exact match, only keep that
    df_f = df_g.loc[df_g["feh"].isin(fclosest)]
    # Finally filter by alpha
    avals = df_f["alpha"].unique()
    aclosest = avals[np.abs(avals - alpha).argsort()[:2]]  # Get two closest alpha values
    if np.any(aclosest == alpha):
        aclosest = np.array([alpha])  # If exact match, only keep that
    df_a = df_f.loc[df_f["alpha"].isin(aclosest)]
    return df_a

    # Now filter by logg
    # gvalues = df_t.index.get_level_values("logg").unique()
    # gclosest = gvalues[np.abs(gvalues - logg).argsort()[:2]]  # Get two closest logg values
    # if np.any(gclosest == logg):
    #     gclosest = np.array([logg])  # If exact match, only keep that
    # df_g = df_t.loc[df_t.index.get_level_values("logg").isin(gclosest)]
    # # Now filter by feh
    # fvalues = df_g.index.get_level_values("feh").unique()
    # fclosest = fvalues[np.abs(fvalues - feh).argsort()[:2]]  # Get two closest feh values
    # if np.any(fclosest == feh):
    #     fclosest = np.array([feh])  # If exact match, only keep that
    # df_f = df_g.loc[df_g.index.get_level_values("feh").isin(fclosest)]
    # # Finally filter by alpha
    # avals = df_f.index.get_level_values("alpha").unique()
    # aclosest = avals[np.abs(avals - alpha).argsort()[:2]]  # Get two closest alpha values
    # if np.any(aclosest == alpha):
    #     aclosest = np.array([alpha])  # If exact match, only keep that
    # df_a = df_f.loc[df_f.index.get_level_values("alpha").isin(aclosest)]
    # return df_a


@debug_function
def compute_weights(
    nearest_df: pd.DataFrame,
    teff: int,
    logg: float,
    feh: float,
    alpha: float = 0.0,
) -> list[WeightedPhoenixDataFile]:
    """Compute interpolation weights for the nearest grid points and attach to the DataFrame.

    Args:
        nearest_df: DataFrame with the nearest grid points and their filenames.
        teff: Effective temperature to match.
        logg: Surface gravity to match.
        feh: Metallicity to match.
        alpha: Alpha element enhancement to match (default is 0.0).
    Returns:
        Dataframe with an additional 'weight' column for interpolation.
    """
    # Extract the unique values for each parameter
    teff_vals = sorted(nearest_df["teff"].unique())
    logg_vals = sorted(nearest_df["logg"].unique())
    feh_vals = sorted(nearest_df["feh"].unique())
    alpha_vals = sorted(nearest_df["alpha"].unique())

    target_point = [teff, logg, feh, alpha]

    t = []
    for param_val, param_values in zip(target_point, [teff_vals, logg_vals, feh_vals, alpha_vals]):
        if len(param_values) == 1:
            t.append(0.0)
        else:
            lower, upper = min(param_values), max(param_values)
            t_val = (param_val - lower) / (upper - lower)
            t.append(t_val)
    t_teff, t_logg, t_feh, t_alpha = t

    weights = {}
    for idx, row in nearest_df.iterrows():
        idx_i = teff_vals.index(row["teff"])
        idx_j = logg_vals.index(row["logg"])
        idx_k = feh_vals.index(row["feh"])
        idx_l = alpha_vals.index(row["alpha"])

        weight = (
            ((1 - t_teff) if idx_i == 0 else t_teff if len(teff_vals) > 1 else 1)
            * ((1 - t_logg) if idx_j == 0 else t_logg if len(logg_vals) > 1 else 1)
            * ((1 - t_feh) if idx_k == 0 else t_feh if len(feh_vals) > 1 else 1)
            * ((1 - t_alpha) if idx_l == 0 else t_alpha if len(alpha_vals) > 1 else 1)
        )
        weights[idx] = weight

    nearest_df = nearest_df.copy()
    weighted_datafile = []
    nearest_df["weight"] = pd.Series(weights)
    for _, row in nearest_df.iterrows():
        if row["weight"] > 0:
            weighted_datafile.append(
                WeightedPhoenixDataFile(
                    teff=row["teff"],
                    logg=row["logg"],
                    feh=row["feh"],
                    alpha=row["alpha"],
                    filename=row["filename"],
                    weight=row["weight"],
                )
            )
    return weighted_datafile


@debug_function
def find_nearest_datafile(df: pd.DataFrame, teff: int, logg: float, feh: float, alpha: float = 0.0) -> PhoenixDataFile:
    """Find the single nearest data file in the DataFrame to the specified parameters.

    Args:
        df: DataFrame with MultiIndex (teff, logg, feh, alpha) and a 'filename' column.
        teff: Effective temperature to match.
        logg: Surface gravity to match.
        feh: Metallicity to match.
        alpha: Alpha element enhancement to match (default is 0.0).
    Returns:
        PhoenixDataFile instance with the nearest grid point and its filename.
    """

    # Compute the distance to each point in the DataFrame
    distances = np.sqrt(
        (df["teff"] - teff) ** 2 + (df["logg"] - logg) ** 2 + (df["feh"] - feh) ** 2 + (df["alpha"] - alpha) ** 2
    )
    min_idx = distances.idxmin()
    row = df.loc[min_idx]
    return PhoenixDataFile(
        teff=row["teff"], logg=row["logg"], feh=row["feh"], alpha=row["alpha"], filename=row["filename"]
    )


@debug_function
def compute_weighted_flux(
    weighted_data: list[WeightedPhoenixDataFile],
    file_loader: DataFileLoader,
    *,
    wavelength_grid: Optional[u.Quantity] = None,
) -> tuple[u.Quantity, u.Quantity]:
    """Compute the weighted flux from a list of WeightedPhoenixDataFile instances.

    Args:
        weighted_data: List of WeightedPhoenixDataFile instances with weights and filenames.
    Returns:
        A tuple of (wavelengths, weighted_flux) as astropy Quantities.
    """
    fluxes = []
    wls = []

    for data in weighted_data:
        wl, flux = file_loader(data)

        wls.append(wl)
        fluxes.append(flux * data.weight)

    perform_interpolation = wavelength_grid is not None
    # Check that all wavelength arrays are the same
    if not perform_interpolation:
        for wl in wls[1:]:
            if not np.array_equal(wl, wls[0]):
                smallest_wl = np.argmin([len(w) for w in wls])
                wavelength_grid = wls[smallest_wl]
                perform_interpolation = True
                break
    if perform_interpolation:
        # Interpolate all fluxes to the common wavelength grid
        new_fluxes = []

        for wl, flux in zip(wls, fluxes):
            interp_flux = np.interp(wavelength_grid.value, wl.value, flux.value, left=0.0, right=0.0) << flux.unit
            new_fluxes.append(interp_flux)
        total_flux = sum(new_fluxes)
        return wavelength_grid, total_flux
    else:
        total_flux = sum(fluxes)
        return wls[0], total_flux


@debug_function
def filter_parameter(df: pd.DataFrame, param: str, value: FilterType) -> pd.DataFrame:
    """Filter the DataFrame based on the given parameter and value.

    Args:
        df: DataFrame with MultiIndex (teff, logg, feh, alpha) and a 'filename' column.
        param: Parameter to filter by ('teff', 'logg', 'feh', 'alpha').
        value: Value to filter by. Can be a single value, a tuple specifying a range, or "all".
    Returns:
        Filtered DataFrame.
    """
    if value == "all" or value is None:
        return df
    elif isinstance(value, (tuple, list, np.ndarray)) and len(value) == 2:
        return df[df[param].between(value[0], value[1])]
    else:
        return df.loc[df[param] == value]


def test_boundaries(param: str, value: float, bounds: tuple[float, float]):
    if value < bounds[0] or value > bounds[1]:
        raise InterpolationBoundaryError(param, value, bounds)


class PhoenixSource(abc.ABC):
    """Abstract base class for Phoenix model sources."""

    KEY: str = "ignore"

    @classmethod
    def available_models(cls) -> list[str]:
        """Return a list of available model names for this source."""
        return []

    def __init__(
        self,
        path: Optional[pathlib.Path] = None,
        interpolation_mode: InterpolationMode = "linear",
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize the Phoenix source.

        Args:
            path: Local path to the model files. If None, models will be downloaded as needed.
            interpolation_mode: Interpolation mode to use ("nearest" or "linear").
            base_url: Optional base URL to download the models from. Defaults to the standard Phoenix STSCI model url.
            model_name: Optional model name to use in the URL if needed.
        Raises:
            ValueError: If an unknown interpolation mode is provided.

        """
        self.interpolation_mode = interpolation_mode
        if interpolation_mode not in InterpolationMode:
            raise UnknownInterpolationModeError(interpolation_mode)

        self.base_url = base_url
        self.model_name = model_name
        self.path = pathlib.Path(path) if path else None

        self.boundaries_cache = None

    @abc.abstractmethod
    def spectral_grid(self) -> Optional[u.Quantity]:
        """Return the wavelength grid if available."""
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        from .registry import register_source as register

        super().__init_subclass__(**kwargs)
        if cls.KEY == "ignore":
            return  # Do not register the base class

        register(cls.KEY, cls)

    @classmethod
    def validate_datafile(cls, directory: pathlib.Path) -> list[bool]:
        """Validate if the file format is correct for this source.

        Given a directory, will check if the files in the directory match the expected format for this source.

        Args:
            directory: Path to the directory containing the model files.

        Returns:
            A list of booleans indicating whether each file is valid for this source.
        """
        raise NotImplementedError

    def metadata(self) -> dict:
        """Return metadata about the Phoenix source."""
        return {}

    def boundaries(self):
        teff = (9999999, -9999999)
        logg = (9999999, -9999999)
        feh = (9999999, -9999999)
        alpha = (9999999, -9999999)
        for datafile in self.list_available_files():
            teff = (min(teff[0], datafile.teff), max(teff[1], datafile.teff))
            logg = (min(logg[0], datafile.logg), max(logg[1], datafile.logg))
            feh = (min(feh[0], datafile.feh), max(feh[1], datafile.feh))
            alpha = (min(alpha[0], datafile.alpha), max(alpha[1], datafile.alpha))
        return {"teff": teff, "logg": logg, "feh": feh, "alpha": alpha}

    @classmethod
    def download_model(
        cls,
        output_dir: pathlib.Path,
        *,
        teff: FilterType,
        logg: FilterType,
        feh: FilterType,
        alpha: FilterType = 0.0,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        mkdir: bool = True,
        progress: bool = True,
    ) -> pathlib.Path:
        """Download a Phoenix model given the specified parameters.

        Parameters can be either a single value, a tuple specifying a range, or "all" to download all available files for a model.

        Args:
            output_dir: Path to save the downloaded file.
            teff: Effective temperature range/value of the desired model.
            logg: Surface gravity range/value of the desired model.
            feh: Metallicity range/value of the desired model.
            alpha: Alpha element enhancement range/value of the desired model (default is 0.0).
            base_url: Optional base URL to download the model from. Defaults to the standard Phoenix STSCI model url.
            model_name: Optional model name to use in the URL if needed. (Some sources may require this)
            mkdir: Whether to create the output directory if it does not exist.
        Returns:
            Path to the downloaded file.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def list_available_files(self) -> list[PhoenixDataFile]:
        """List available Phoenix model files."""
        pass

    @abc.abstractmethod
    def load_file(self, dataset: PhoenixDataFile) -> tuple[u.Quantity, u.Quantity]:
        """Load the content of a Phoenix model file.

        Args:
            dataset: A PhoenixDataFile instance representing the model to load.
        Returns:
            The content of the model file as a string.
        """
        pass

    def spectrum(
        self,
        teff: int,
        logg: Optional[float] = 0.0,
        feh: Optional[float] = 0.0,
        alpha: float = 0.0,
        bounds_error: bool = True,
        use_planck: bool = False,
    ):
        """Get the spectrum for the specified parameters.

        Args:
            teff: Effective temperature.
            logg: Surface gravity.
            feh: Metallicity.
            alpha: Alpha element enhancement (default is 0.0).
            bounds_error: If True, raise an error if the parameters are out of bounds. If False, clip to the nearest available model.
            use_planck: If True, use a blackbody spectrum if the temperature is out of bounds. Overrides bounds_error for temperature.
        Returns:
            A tuple of (wavelengths, flux) as astropy Quantities.
        """
        boundaries = self.boundaries() if self.boundaries_cache is None else self.boundaries_cache
        self.boundaries_cache = boundaries

        # Temeperature check first
        if teff < boundaries["teff"][0] or (teff > boundaries["teff"][1] and use_planck):
            # We need to get a wavelength grid to return
            # Replace this with a more appropriate wavelength grid if needed
            # possibly based on the nearest model or loading it beforehand
            wav = self.spectral_grid()

            flux = planck(wav, teff << u.K)
            return wav, flux

        if bounds_error:
            test_boundaries("teff", teff, boundaries["teff"])
            test_boundaries("logg", logg, boundaries["logg"])
            test_boundaries("feh", feh, boundaries["feh"])
            test_boundaries("alpha", alpha, boundaries["alpha"])
        else:
            teff = np.clip(teff, boundaries["teff"][0], boundaries["teff"][1])
            logg = np.clip(logg, boundaries["logg"][0], boundaries["logg"][1])
            feh = np.clip(feh, boundaries["feh"][0], boundaries["feh"][1])
            alpha = np.clip(alpha, boundaries["alpha"][0], boundaries["alpha"][1])

        df = construct_phoenix_dataframe(self.list_available_files())
        if self.interpolation_mode == InterpolationMode.NEAREST:
            nearest = find_nearest_points(df, teff=teff, logg=logg, feh=feh, alpha=alpha)
            if nearest.shape[0] == 0:
                raise NoAvailableDataError
            nearest_datafile = find_nearest_datafile(nearest, teff=teff, logg=logg, feh=feh, alpha=alpha)
            wl, flux = self.load_file(nearest_datafile)
            return wl, flux
        elif self.interpolation_mode == InterpolationMode.LINEAR:
            nearest = find_nearest_points(df, teff=teff, logg=logg, feh=feh, alpha=alpha)
            if nearest.shape[0] == 0:
                raise NoAvailableDataError
            weighted_datafiles = compute_weights(nearest, teff=teff, logg=logg, feh=feh, alpha=alpha)
            wl, flux = compute_weighted_flux(weighted_datafiles, self.load_file)
            return wl, flux
        else:
            raise UnknownInterpolationModeError(self.interpolation_mode)
