import pathlib
import urllib.parse
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.utils.data import download_file

from phoenix4all.log import debug_function, module_logger

from .core import FilterType, InterpolationMode, NoAvailableDataError, PhoenixDataFile, PhoenixSource

BASE_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/"

_log = module_logger(__name__)


@debug_function
def get_catalogue(*, path: Optional[pathlib.Path] = None, base_url: Optional[str] = None) -> pathlib.Path:
    """Download the Phoenix model catalogue file if not already present locally.

    Args:
        path: Optional local path to save the catalogue file. If None, downloads to a temporary location.
        base_url: Optional base URL to download the catalogue from. Defaults to the standard Phoenix STSCI model url.
    Returns:
        The local file path to the downloaded catalogue file.
    """
    _log.debug("get_catalogue called with path=%s, base_url=%s", path, base_url)
    if path:
        path = pathlib.Path(path)
        return path / "catalog.fits"
    url = urllib.parse.urljoin(base_url or BASE_URL, "catalog.fits")
    local_path = download_file(url, pkgname="phoenix4all", cache=True) if path is None else path
    return pathlib.Path(local_path)


@debug_function
def load_catalogue_paths(catalogue_path: pathlib.Path) -> list[str]:
    """Load the catalogue file and return a list of file paths.

    Args:
        catalogue_path: Local path to the catalogue file.
    Returns:
        A list of file paths from the catalogue.
    """
    from astropy.io import fits

    with fits.open(catalogue_path) as hdul:
        data = hdul[1].data  # Assuming the data is in the first extension
        file_paths = [d[1][:-5] for d in data]  # Remove [g] extension
    return file_paths


@debug_function
def list_available_files(path: Optional[str] = None, base_url: Optional[str] = None) -> list:
    """List available Phoenix model files from the catalogue.

    Args:
        path: Optional local path where the catalogue file is stored. If None, downloads to a temporary location.
        base_url: Optional base URL to download the catalogue from. Defaults to the standard Phoenix STSCI model url.

    Returns:
        A list of filenames available in the Phoenix model catalogue.
    """
    _log.debug("list_available_files called with path=%s, base_url=%s", path, base_url)
    path = pathlib.Path(path) if path else None
    base_url = base_url or BASE_URL
    catalog_path = get_catalogue(path=path, base_url=base_url)
    _log.debug("Using catalogue at %s", catalog_path)
    from astropy.io import fits

    data_files = []
    with fits.open(catalog_path) as hdul:
        data = hdul[1].data  # Assuming the data is in the first extension
        for d in data:
            properties = d[0]
            temperature, feh, logg = [float(p) for p in properties.split(",")]
            _log.debug("Found model: Teff=%s, logg=%s, [Fe/H]=%s", temperature, logg, feh)
            filename = d[1][:-5]
            _log.debug("Filename: %s", filename)
            filename = str(path / filename) if path else urllib.parse.urljoin(base_url, filename)
            data_files.append(PhoenixDataFile(teff=int(temperature), logg=logg, feh=feh, alpha=0.0, filename=filename))
    ## Now if path is given we need to filter the files to only those that exist
    if path:
        data_files = [df for df in data_files if pathlib.Path(df.filename).exists()]
    return data_files


def load_file(dataset: PhoenixDataFile) -> tuple[u.Quantity, u.Quantity]:
    """Load the content of a Phoenix model file.

    Args:
        dataset: A PhoenixDataFile instance representing the model to load.
    Returns:
        The content of the model file as a string.
    """
    from astropy.io import fits

    from phoenix4all.net import is_remote_url

    local_path = dataset.filename
    # check if this is a local file or a URL

    if is_remote_url(dataset.filename):
        local_path = download_file(dataset.filename, pkgname="phoenix4all", cache=True)

    local_path = pathlib.Path(local_path)

    if not local_path.exists():
        raise FileNotFoundError(local_path)

    with fits.open(local_path) as hdul:
        columns = hdul[1].columns
        # logg has format g10 = log = 1.0 so convert to int
        logg = int(dataset.logg * 10)
        # No alpha enhancement in this version

        logg_index = columns.names.index(f"g{logg:02d}")

        data = np.array(hdul[1].data.copy())  # Assuming the data is in the first extension

        data_shape = data.shape[0]
        data_view = data.view(">f8").reshape(data_shape, -1)
        wav = data_view[:, 0]
        flux = data_view[:, logg_index]

    wav = np.array(wav) << u.AA
    flux = np.array(flux) << (u.erg / (u.s * u.cm**2 * u.AA))
    return wav, flux


def download_model(
    output_dir: pathlib.Path,
    *,
    teff: FilterType,
    logg: FilterType,
    feh: FilterType,
    alpha: FilterType = 0.0,
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
    files = list_available_files(base_url=base_url)
    df = construct_phoenix_dataframe(files)
    # Filter the DataFrame based on the provided parameters
    df = filter_parameter(df, "teff", teff)
    df = filter_parameter(df, "logg", logg)
    df = filter_parameter(df, "feh", feh)
    df = filter_parameter(df, "alpha", alpha)

    files_to_download = []
    output_path_for_file = []

    # Add catalogue file
    files_to_download.append(urllib.parse.urljoin(base_url, "catalog.fits"))
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

    _log.info("Downloading files:", files_to_download)
    print(files_to_download)
    print(output_path_for_file)
    return download_to_directory(files_to_download, output_path_for_file, progress=progress)

    # phoenix.download_model(output_path=output_path, teff=dataset.teff, logg=dataset.logg,
    #                        feh=dataset.feh, alpha=dataset.alpha, base_url=base_url)


class SynphotSource(PhoenixSource):
    """Concrete implementation of PhoenixSource for Synphot models."""

    KEY: str = "synphot"

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
        super().__init__(path=path, interpolation_mode=interpolation_mode, base_url=base_url, model_name=model_name)

        self.data_files = list_available_files(path=self.path, base_url=self.base_url)

        self.boundaries_cache = self.boundaries()

        # Use minimum temperature model to get wavelength grid
        # Not sure if theres a more elegant way of doing it
        self.wavelength_grid = self.spectrum(
            teff=self.boundaries_cache["teff"][0], logg=0.0, feh=0.0, alpha=0.0, bounds_error=True
        )[0]

    @classmethod
    def available_models(cls) -> list[str]:
        """Return a list of available model names for this source."""
        return ["agss2009"]

    def metadata(self) -> dict:
        """Return metadata about the Phoenix source."""
        return {
            "source": "Synphot",
            "url_source": self.base_url,
            "reference": "Allard et al. 03, Allard et al. 07, Allard et al. 09",
        }

    def spectral_grid(self):
        return self.wavelength_grid

    def list_available_files(self) -> list[PhoenixDataFile]:
        """List available Phoenix model files."""
        return self.data_files

    def load_file(self, dataset: PhoenixDataFile) -> tuple[u.Quantity, u.Quantity]:
        return load_file(dataset)

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
            output_dir, teff=teff, logg=logg, feh=feh, alpha=alpha, base_url=base_url, progress=progress
        )
