import functools
import pathlib
import re
import urllib.parse
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.utils.data import download_file

from ..log import debug_function, module_logger
from ..net.http import fetch_listing
from .core import FilterType, InterpolationMode, NoAvailableDataError, PhoenixDataFile, PhoenixSource

_log = module_logger(__name__)

BASE_URL = "https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/"

BASE_MODEL = "PHOENIX-ACES-AGSS-COND-2011"

# Ignore anything before lte and after .fits
PATTERN_WITH_ALPHA = re.compile(
    r".*lte(\d{5})([+-][0-9]+\.[0-9]+)([+-][0-9]+\.[0-9]+)\.Alpha=([+-][0-9]+\.[0-9]+)\..*\.fits$"
)
PATTERN_WITHOUT_ALPHA = re.compile(r".*lte(\d{5})([+-][0-9]+\.[0-9]+)([+-][0-9]+\.[0-9]+)\..*\.fits$")


def load_directory_from_cache():
    import importlib.resources as ires
    import json

    from ..io import json_unzip

    with ires.open_text("phoenix4all.cache.hiresfits", "hiresfit_cache.jsonz") as f:
        result = json_unzip(json.load(f))

    return [PhoenixDataFile(**r) for r in result]


@debug_function
@functools.lru_cache(maxsize=100)
def recursive_list(url: str) -> list:
    """Recursively list all files in a given URL directory.

    Args:
        url: The base URL to start listing from.
    Returns:
        A list of all file URLs found recursively.

    """
    files = []
    _, listing = fetch_listing(url)
    for item in listing:
        if item.name in ("./", "../"):
            continue
        if item.name.endswith("/"):
            # It's a directory, recurse into it
            files.extend(recursive_list(urllib.parse.urljoin(url, item.name)))
        elif item.name.endswith(".fits"):
            # It's a file, add to the list
            files.append(urllib.parse.urljoin(url, item.name))
    return files


@debug_function
def parse_filename(filename: str) -> Optional[PhoenixDataFile]:
    """Parse a Phoenix model filename to extract its parameters.

    Args:
        filename: The Phoenix model filename.
    Returns:
        A PhoenixDataFile instance with extracted parameters, or None if parsing fails.
    """
    # There are two forms of the files names
    # ALpha enhancement:
    # lte02300-0.00+0.5.Alpha=+0.60.[SOME MODEL NAME WE DONT CARE ABOUT].fits
    # No alpha enhancement:
    # lte02300-0.00-0.0.[SOME MODEL NAME WE DONT CARE ABOUT].fits

    # Format is
    # lte<temperature>-<logg>+/-<feh>.[SOME MODEL NAME WE DONT CARE ABOUT].fits
    # or
    # lte<temperature>-<logg>+/-<feh>.Alpha=+/-<alpha>.[SOME MODEL NAME WE DONT CARE ABOUT].fits

    match = PATTERN_WITH_ALPHA.match(filename)
    if match:
        teff = int(match.group(1))
        logg = float(match.group(2))
        feh = float(match.group(3))
        alpha = float(match.group(4))
        return PhoenixDataFile(teff=teff, logg=-logg, feh=feh, alpha=alpha, filename=filename)
    match = PATTERN_WITHOUT_ALPHA.match(filename)
    if match:
        teff = int(match.group(1))
        logg = float(match.group(2))
        feh = float(match.group(3))
        alpha = 0.0
        return PhoenixDataFile(teff=teff, logg=-logg, feh=feh, alpha=alpha, filename=filename)
    return None


@debug_function
def load_wavelength_grid(path: Optional[str] = None, url: Optional[str] = None) -> u.Quantity:
    """Load the wavelength grid

    Args:
        path: Optional local path to the wavelength grid file. If None, downloads to astropy cache.
        url: Optional URL to download the wavelength grid from. Defaults to the standard Gottingen model url.
    Returns:
        The wavelength grid as an astropy Quantity.
    """

    local_path = pathlib.Path(path) if path else None
    local_path = local_path / "WAVE_{BASE_MODEL}.fits" if local_path else None
    url = url or urllib.parse.urljoin(BASE_URL, f"WAVE_{BASE_MODEL}.fits")
    local_path = download_file(url, pkgname="phoenix4all", cache=True) if local_path is None else local_path
    from astropy.io import fits

    with fits.open(local_path) as hdul:
        data = hdul[0].data.copy()  # Assuming the data is in the first extension

    return np.asarray(data) << u.AA


@debug_function
def list_available_files(
    path: Optional[str] = None,
    base_url: Optional[str] = None,
    url_model: str = "PHOENIX-ACES-AGSS-COND-2011",
) -> list:
    """List available Phoenix model files from the catalogue.

    Args:
        path: Optional local path where the catalogue file is stored. If None, downloads to a temporary location.
        base_url: Optional base URL to download the catalogue from. Defaults to the standard Phoenix STSCI model url.

    Returns:
        A list of filenames available in the Phoenix model catalogue.
    """
    # if path is given walk through the directory and list all fits files
    if path:
        path = pathlib.Path(path)
        data_files = path.rglob("*.fits")
        data_files = [str(f) for f in data_files]

    else:
        base_url = base_url or BASE_URL
        # We recuresively list all files in the base_url
        # Make sure model_url ends with /
        if base_url == BASE_URL and url_model == BASE_MODEL:
            _log.info("Loading from cache")
            return load_directory_from_cache()

        model_url = urllib.parse.urljoin(base_url, url_model)

        if not model_url.endswith("/"):
            model_url += "/"
        data_files = recursive_list(str(model_url))

    data_files = [parse_filename(f) for f in data_files]
    data_files = [f for f in data_files if f is not None]

    return data_files


@debug_function
def load_file(dataset: PhoenixDataFile, wavelength_grid: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
    """Load the content of a Phoenix model file.

    Args:
        dataset: A PhoenixDataFile instance representing the model to load.
    Returns:
        The content of the model file as a string.
    """
    from astropy.io import fits

    local_path = dataset.filename
    if not pathlib.Path(local_path).exists():
        local_path = download_file(dataset.filename, pkgname="phoenix4all", cache=True)

    wav = wavelength_grid

    with fits.open(local_path) as hdul:
        flux = hdul[0].data.copy()  # Assuming the data is in the first extension

    # erg/s/cm^2/cm
    flux = np.array(flux) << (u.erg / (u.s * u.cm**2) / u.cm)
    return wav, flux


@debug_function
def download_model(
    output_dir: pathlib.Path,
    *,
    teff: FilterType,
    logg: FilterType,
    feh: FilterType,
    alpha: FilterType = 0.0,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    progress: bool = True,
) -> list[pathlib.Path]:
    """Download a specific Phoenix model file based on the given parameters.

    Will mirror the directory structure of the Synphot models.

    Args:
        output_dir: Local path to save the downloaded model file.
        teff: Effective temperature of the desired model.
        logg: Surface gravity of the desired model.
        feh: Metallicity of the desired model.
        alpha: Alpha element enhancement of the desired model (default is 0.0).
        base_url: Optional base URL to download the catalogue from. Defaults to the standard Phoenix STSCI model url.
    Returns:
        The local file path to the downloaded model file.
    """
    from ..net.http import download_to_directory
    from .core import construct_phoenix_dataframe, filter_parameter

    output_dir = pathlib.Path(output_dir)
    base_url = base_url or BASE_URL
    model_name = model_name or BASE_MODEL
    _log.info(f"Listing available files from {base_url}. This may take a while...")
    files = list_available_files(base_url=base_url, url_model=model_name)
    df = construct_phoenix_dataframe(files)
    # Filter the DataFrame based on the provided parameters
    df = filter_parameter(df, "teff", teff)
    df = filter_parameter(df, "logg", logg)
    df = filter_parameter(df, "feh", feh)
    df = filter_parameter(df, "alpha", alpha)

    files_to_download = []
    output_path_for_file = []

    # Add catalogue file
    files_to_download.append(urllib.parse.urljoin(base_url, f"WAVE_{model_name}.fits"))
    output_path_for_file.append(output_dir)

    if df.shape[0] == 0:
        raise NoAvailableDataError
    for _, row in df.iterrows():
        dataset = PhoenixDataFile(
            teff=row["teff"], logg=row["logg"], feh=row["feh"], alpha=row["alpha"], filename=row["filename"]
        )
        # Local path to save the file
        # Remove base_url from filename to get relative path
        relative_path = dataset.filename
        if relative_path.startswith(base_url):
            relative_path = relative_path[len(base_url) :]
        local_path = output_dir / relative_path
        # Now we dont need the filename att the end so just the directory to put it in
        # Remove the filename from the path
        local_dir = local_path.parent

        files_to_download.append(dataset.filename)
        output_path_for_file.append(local_dir)

    _log.info("Downloading files: %s", files_to_download)

    return download_to_directory(files_to_download, output_path_for_file, progress=progress)


class HiResFitsSource(PhoenixSource):
    """Concrete implementation of PhoenixSource for Synphot models."""

    KEY: str = "hiresfits"

    def __init__(
        self,
        *,
        path: Optional[pathlib.Path] = None,
        interpolation_mode: InterpolationMode = "linear",
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Initialize the SynphotSource.

        Will

        Args:
            path: Optional local path where the catalogue file is stored. If None, downloads to a temporary location.
            base_url: Optional base URL to download the catalogue from. Defaults to the standard Phoenix STSCI model url.
            model_name: Optional model name to filter the catalogue files. If None, uses all available models.
            interpolation_mode: Interpolation mode to use. Options are "linear", "nearest"

        Raises:
            ValueError: If an unknown interpolation mode is provided.
        """
        super().__init__(
            path=path,
            interpolation_mode=interpolation_mode,
            base_url=base_url or BASE_URL,
            model_name=model_name or BASE_MODEL,
        )

        self.data_files = list_available_files(path=self.path, base_url=self.base_url, url_model=self.model_name)

        self.wavelength_grid = load_wavelength_grid(
            path=self.path, url=urllib.parse.urljoin(self.base_url, f"WAVE_{self.model_name}.fits")
        )

    def metadata(self) -> dict:
        """Return metadata about the Phoenix source."""
        return {
            "source": "HiResFit",
            "url_source": self.base_url,
            "reference": "arXiv:1303.5632",
        }

    @classmethod
    def available_models(cls) -> list[str]:
        """Return a list of available model names for this source."""
        return ["PHOENIX-ACES-AGSS-COND-2011"]

    def spectral_grid(self):
        return self.wavelength_grid

    def list_available_files(self) -> list[PhoenixDataFile]:
        """List available Phoenix model files."""
        return self.data_files

    def load_file(self, dataset: PhoenixDataFile) -> tuple[u.Quantity, u.Quantity]:
        return load_file(dataset, self.wavelength_grid)

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
    ) -> list[pathlib.Path]:
        """Download specific Phoenix model files based on the given parameters.

        Will mirror the directory structure of the Synphot models.

        Args:
            output_dir: Local path to save the downloaded model files.
            teff: Effective temperature(s) of the desired model(s).
            logg: Surface gravity(ies) of the desired model(s).
            feh: Metallicity(ies) of the desired model(s).
            alpha: Alpha element enhancement(s) of the desired model(s) (default is 0.0).
            base_url: Optional base URL to download the catalogue from. Defaults to the standard Phoenix STSCI model url.
            mkdir: Whether to create the output directory if it does not exist (default is True).

        """
        output_dir = pathlib.Path(output_dir)
        if mkdir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        return download_model(
            output_dir,
            teff=teff,
            logg=logg,
            feh=feh,
            alpha=alpha,
            base_url=base_url,
            model_name=model_name,
            progress=progress,
        )
