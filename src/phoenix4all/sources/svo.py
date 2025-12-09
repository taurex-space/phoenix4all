import pathlib
import urllib.parse as urlparse
from dataclasses import dataclass
from typing import Optional

import bs4
import requests
from astropy import units as u

from ..log import debug_function, module_logger
from .core import FilterType, InterpolationMode, NoAvailableDataError, PhoenixDataFile, PhoenixSource

_log = module_logger(__name__)

BASE_URL = "https://svo2.cab.inta-csic.es/theory/newov2/"

valid_models = [
    "cond00",
    "dusty00",
    "atmo2020_ceq",
    "atmo2020_neq_strong",
    "atmo2020_neq_weak",
    "bt-cond",
    "bt-dusty",
    "bt-nextgen-agss2009",
    "bt-nextgen-gns93",
    "bt-settl",
    "bt-settl-agss",
    "bt-settl-cifist",
    "bt-settl-gns93",
    "bt-settl-2014",
    "coelho_highres",
    "coelho_sed",
    "drift",
    "koester2",
    "Kurucz2003all",
    "Kurucz2003",
    "Kurucz2003alp",
    "levenhagen17",
    "NextGen2",
    "NextGen",
]


@dataclass
class SVOModel:
    id: str
    long_name: str


def create_filename(model_id: str, datafile: PhoenixDataFile) -> str:
    """Create a filename for a Phoenix model based on its parameters.

    Args:
        model_id: The SVO model ID.
        teff: Effective temperature.
        logg: Surface gravity.
        feh: Metallicity [Fe/H].
        alpha: Alpha element enhancement.
    Returns:
        The constructed filename as a string.
    """
    filename = (
        f"svo_{model_id}_T{datafile.teff:05d}_g{datafile.logg:.2f}_m{datafile.feh:+.2f}_a{datafile.alpha:.2f}.txt"
    )
    return filename


def convert_filename_to_datafile(filename: str, model_id: str) -> Optional[PhoenixDataFile]:
    """Convert a Phoenix model filename to a PhoenixDataFile object.

    Args:
        filename: The filename to convert.
    Returns:
        A PhoenixDataFile object or None if parsing fails.
    """
    import re

    pattern = rf"svo_{re.escape(model_id)}_T(?P<teff>\d{{5}})_g(?P<logg>-?\d+\.\d{{2}})_m(?P<feh>[+-]?\d+\.\d{{2}})_a(?P<alpha>[+-]?\d+\.\d{{2}})\.txt"
    match = re.match(pattern, pathlib.Path(filename).name)
    if not match:
        return None

    teff = int(match.group("teff"))
    logg = float(match.group("logg"))
    feh = float(match.group("feh"))
    alpha = float(match.group("alpha"))

    return PhoenixDataFile(teff=teff, logg=logg, feh=feh, alpha=alpha, filename=filename)


def find_datasets_in_path(path: pathlib.Path, model_id: str) -> list[PhoenixDataFile]:
    """Find Phoenix model files in a given local path."""
    data_files = []
    for file in path.glob("svo_*.txt"):
        data_file = convert_filename_to_datafile(file.name, model_id)
        if data_file:
            data_files.append(data_file)
    return data_files


def load_available_data_from_cache(model_id: str) -> list[PhoenixDataFile]:
    import importlib.resources as ires
    import json

    from ..io import json_unzip
    from .core import ModelNotFoundError

    with ires.open_text("phoenix4all.cache.svo", "svo_dataset.jsonz") as f:
        result = json_unzip(json.load(f))
    if model_id in result:
        result = result[model_id]
    else:
        raise ModelNotFoundError(model_id)

    return [PhoenixDataFile(**r) for r in result]


def list_available_models(no_cache: bool = True, base_url: str = BASE_URL) -> list[SVOModel]:
    index_path = urlparse.urljoin(base_url, "index.php")
    # Add to query models=cond00
    index_path = index_path + "?" + urlparse.urlencode({"models": "cond00"})

    response = requests.get(index_path, timeout=100)
    response.raise_for_status()

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(response.text, "html.parser")

    selector = soup.find("select", {"name": "reqmodels[]"})
    options = selector.find_all("option")

    return [SVOModel(option["value"], option.text) for option in options if option["value"] in valid_models]


def _determine_property_indicies(soup: bs4.BeautifulSoup) -> dict[str, int]:
    """Determine the indices of the properties in the table header."""
    trs = soup.find_all("td", class_="tabcab")

    indices = {}

    for idx, tr in enumerate(trs):
        text = tr.text.strip()
        indices[text] = idx

    return indices


def _parse_data_row(
    tr: bs4.BeautifulSoup, indices: dict[str, int], base_url: str = BASE_URL
) -> Optional[PhoenixDataFile]:
    """Parse a data row from the SVO table."""

    tds = tr.find_all("td", class_="tabfld")
    if len(tds) == 0:
        return None
    teff = int(tds[indices["Teff"]].text.strip()) if "Teff" in indices else None
    logg = float(tds[indices["Logg"]].text.strip()) if "Logg" in indices else 0.0
    meta = float(tds[indices["Metallicity"]].text.strip()) if "Metallicity" in indices else 0.0
    alpha = float(tds[indices["Alpha"]].text.strip()) if "Alpha" in indices else 0.0
    link = tds[-3].find("a")["href"]
    return PhoenixDataFile(teff=teff, logg=logg, feh=meta, alpha=alpha, filename=urlparse.urljoin(base_url, link))


def list_datasets_from_url(model_id: str, base_url: str = BASE_URL) -> list[PhoenixDataFile]:
    """List available Phoenix model files from the catalogue.

    Args:
        model_id: The SVO model ID to query.
        base_url: Optional base URL to download the catalogue from. Defaults to the standard SVO library.
    """
    data = requests.post(
        urlparse.urljoin(base_url, "index.php"),
        timeout=100,
        data={
            "models": model_id,
            "oby": "",
            "odesc": "",
            "sbut": "",
            "params[bt-settl][teff][min]": "0",
            "params[bt-settl][teff][max]": "700000",
            "params[bt-settl][logg][min]": "-90",
            "params[bt-settl][logg][max]": "90",
            "params[bt-settl][meta][min]": "-90",
            "params[bt-settl][meta][max]": "90",
            "params[bt-settl][alpha][min]": "-90",
            "params[bt-settl][alpha][max]": "90",
            "nres": "all",
            "boton": "Search",
            "reqmodels[]": model_id,
        },
    )
    data.raise_for_status()
    soup = bs4.BeautifulSoup(data.text, "html.parser")
    indices = _determine_property_indicies(soup)
    trs = soup.find_all("tr")
    datasets = []
    for tr in trs:
        dataset = _parse_data_row(tr, indices)
        if dataset:
            datasets.append(dataset)
    return datasets


@debug_function
def list_available_dataset(model_id: str, path: Optional[pathlib.Path] = None, base_url: Optional[str] = None) -> list:
    """List available Phoenix model files from the catalogue.

    Args:
        path: Optional local path where the catalogue file is stored. If None, downloads to a temporary location.
        base_url: Optional base URL to download the catalogue from. Defaults to the standard Phoenix STSCI model url.

    Returns:
        A list of filenames available in the Phoenix model catalogue.
    """
    base_url = base_url or BASE_URL
    if path:
        path = pathlib.Path(path)
    elif base_url == BASE_URL:
        return load_available_data_from_cache(model_id)

    if path is not None and path.exists() and path.is_dir():
        return find_datasets_in_path(path, model_id)
    else:
        return list_datasets_from_url(model_id, base_url)


@debug_function
def load_file(data_file: PhoenixDataFile) -> str:
    """Load a Phoenix model file from a given URL or local path.

    Args:
        data_file: The PhoenixDataFile object representing the model file to load.
    Returns:
        The content of the model file as a string.
    """
    import pandas as pd
    from astropy.utils.data import download_file

    filename = data_file.filename

    if filename.startswith("http"):
        filename = download_file(data_file.filename, cache=True, pkgname="phoenix4all")

    filename = pathlib.Path(filename)
    if not pathlib.Path(filename).exists():
        _log.error("File %s does not exist", data_file.filename)
        raise FileNotFoundError(filename)

    df = pd.read_csv(filename, comment="#", names=["wavelength", "flux"], sep=r"\s+")
    spectral = df["wavelength"].values << u.AA
    flux = df["flux"].values << u.erg / (u.s * u.cm**2 * u.AA)
    return spectral, flux


def download_model(
    output_dir: pathlib.Path,
    model_id: str,
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
    files = list_available_dataset(model_id=model_id, base_url=base_url)
    df = construct_phoenix_dataframe(files)
    # Filter the DataFrame based on the provided parameters
    df = filter_parameter(df, "teff", teff)
    df = filter_parameter(df, "logg", logg)
    df = filter_parameter(df, "feh", feh)
    df = filter_parameter(df, "alpha", alpha)

    files_to_download = []
    output_path_for_file = []

    if df.shape[0] == 0:
        raise NoAvailableDataError
    for _, row in df.iterrows():
        dataset = PhoenixDataFile(
            teff=row["teff"], logg=row["logg"], feh=row["feh"], alpha=row["alpha"], filename=row["filename"]
        )
        data_filename = create_filename(model_id, dataset)
        # Local path to save the file
        # Remove base_url from filename to get relative path

        local_path = output_dir / data_filename
        # Now we dont need the filename att the end so just the directory to put it in
        # Remove the filename from the path

        files_to_download.append(dataset.filename)
        output_path_for_file.append(local_path)

    _log.info("Downloading %s files to %s outputs", len(files_to_download), len(output_path_for_file))

    return download_to_directory(files_to_download, output_path_for_file, progress=progress, includes_filename=True)

    # _log.debug("list_available_files called with path=%s, base_url=%s", path, base_url)
    # path = pathlib.Path(path) if path else None
    # base_url = base_url or BASE_URL
    # catalog_path = get_catalogue(path=path, base_url=base_url)
    # _log.debug("Using catalogue at %s", catalog_path)
    # from astropy.io import fits

    # data_files = []
    # with fits.open(catalog_path) as hdul:
    #     data = hdul[1].data  # Assuming the data is in the first extension
    #     for d in data:
    #         properties = d[0]
    #         temperature, feh, logg = [float(p) for p in properties.split(",")]
    #         _log.debug("Found model: Teff=%s, logg=%s, [Fe/H]=%s", temperature, logg, feh)
    #         filename = d[1][:-5]
    #         _log.debug("Filename: %s", filename)
    #         filename = str(path / filename) if path else urllib.parse.urljoin(base_url, filename)
    #         data_files.append(PhoenixDataFile(teff=int(temperature), logg=logg, feh=feh, alpha=0.0, filename=filename))
    # ## Now if path is given we need to filter the files to only those that exist
    # if path:
    #     data_files = [df for df in data_files if pathlib.Path(df.filename).exists()]
    # return data_files


class SVOSource(PhoenixSource):
    """Concrete implementation of PhoenixSource for Synphot models."""

    KEY: str = "svo"

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
        base_url = base_url or BASE_URL
        super().__init__(path=path, interpolation_mode=interpolation_mode, base_url=base_url, model_name=model_name)

        self.allowed_models = [model.id for model in list_available_models(no_cache=True, base_url=self.base_url)]
        if model_name is None:
            model_name = self.allowed_models[0]

        self.data_files = list_available_dataset(model_id=model_name, path=path, base_url=base_url)
        self.boundaries_cache = self.boundaries()

        # Use minimum temperature model to get wavelength grid
        # Not sure if theres a more elegant way of doing it
        self.wavelength_grid = self.spectrum(
            teff=self.boundaries_cache["teff"][0], logg=0.0, feh=0.0, alpha=0.0, bounds_error=False
        )[0]

    @classmethod
    def available_models(cls) -> list[str]:
        """Return a list of available model names for this source."""
        return valid_models

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
    def list_available_models(
        cls,
        base_url: Optional[str] = None,
        no_cache: bool = True,
    ) -> list[SVOModel]:
        """List available Phoenix models in the Synphot catalogue.

        Args:
            base_url: Optional base URL to download the catalogue from. Defaults to the standard Phoenix STSCI model url.
            no_cache: Whether to bypass any caching mechanisms (default is True).
        Returns:
            A list of available SVOModel objects.
        """
        base_url = base_url or BASE_URL
        return list_available_models(no_cache=no_cache, base_url=base_url)

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
            model_id=model_name,
            teff=teff,
            logg=logg,
            feh=feh,
            alpha=alpha,
            base_url=base_url,
            progress=progress,
        )
