# Phoenix4All - The Phoenix library for the lazy astronomer

[![Release](https://img.shields.io/github/v/release/taurex-space/phoenix4all)](https://img.shields.io/github/v/release/taurex-space/phoenix4all)
[![Build status](https://img.shields.io/github/actions/workflow/status/taurex-space/phoenix4all/main.yml?branch=main)](https://github.com/taurex-space/phoenix4all/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/taurex-space/phoenix4all/branch/main/graph/badge.svg)](https://codecov.io/gh/taurex-space/phoenix4all)
[![Commit activity](https://img.shields.io/github/commit-activity/m/taurex-space/phoenix4all)](https://img.shields.io/github/commit-activity/m/taurex-space/phoenix4all)
[![License](https://img.shields.io/github/license/taurex-space/phoenix4all)](https://img.shields.io/github/license/taurex-space/phoenix4all)

All in one library for loading and using phoenix spectra

- **Github repository**: <https://github.com/taurex-space/phoenix4all/>
- **Documentation** <https://taurex-space.github.io/phoenix4all/>


## What is it?

Phoenix4All is a Python library that provides an easy-to-use interface for accessing and utilizing the PHOENIX stellar atmosphere models and synthetic spectra. It attempts to solve the infuriatingly complex process of downloading, managing, and interpolating PHOENIX models by providing a simple and efficient API that does it for you. It has been especially designed for astronomers and astrophysicists who need to work with stellar spectra in their research. But its really designed to make your life easier if you need to use PHOENIX models.

## Features

- **Lazy loading**: Download and cache models on-the-fly as needed.
- **Downloader**: A downloader to grab the models if you want to manage them yourself.
- **Interpolation**: Linearly interpolate between models to get spectra for arbitrary stellar parameters.
- **Multiple sources**: Support for different PHOENIX model sources, including high-resolution FITS files and lower-resolution ASCII files.
- **Minimal memory usage**: Only needed spectral files are downloaded then loaded into memory.
- **Command-line interface**: A CLI tool to quickly fetch and manage models from the terminal.


## Installation
You can install Phoenix4All using pip:

```bash
pip install phoenix4all
```

## Usage
Here's a simple example of how to use Phoenix4All to fetch and interpolate a PHOENIX model:

```python
from phoenix4all import get_spectrum

wavelength, flux = get_spectrum(
    teff=5778,  # Effective temperature in Kelvin
    logg=4.44,  # Surface gravity in cgs
    feh=0.0,    # Metallicity [Fe/H]
    alpha=0.0,  # Alpha element enhancement [alpha/Fe] if you're too cool for school.
    source='synphot',  # Source of the models
)

print(wavelength, flux)
```

That's it! Phoenix4All will handle the rest, downloading and caching the necessary models, and interpolating to get the desired spectrum.
For multi model sources like `svo`, you can specify the model name as well:

```python
from phoenix4all import get_spectrum
wavelength, flux = get_spectrum(
    teff=5778,
    logg=4.44,
    feh=0.0,
    alpha=0.0,
    source='svo',
    model='bt-dusty',  # Specify the model name
)
print(wavelength, flux)
```

Now if you have already downloaded the models, you can specify the directory where they are stored:

```python
from phoenix4all import get_spectrum
wavelength, flux = get_spectrum(
    teff=5778,
    logg=4.44,
    feh=0.0,
    alpha=0.0,
    source='synphot',
    path='/path/to/your/phoenix/models'  # Specify your local directory
)
print(wavelength, flux)
```

If you want to see what models are available from a source, you can do this:

```python
from phoenix4all import SVOSource
print(SVOSource.available_models())
```

## Being less lazy

You can work directly with the sources if you want more control:

```python
from phoenix4all import SynphotSource

source = SynphotSource(path='/path/to/your/phoenix/models')
spectrum = source.spectrum(teff=5778, logg=4.44, feh=0.0, alpha=0.0)
wavelength, flux = spectrum
print(wavelength, flux)
```

Its faster since theres less overhead in figuring out which files are available.

## Downloading

Ohhhh fancy! You can also use the command-line interface to fetch a model. We can see what models are available from a source like this:

```bash
python -m phoenix4all.downloader --help
```
Which will give you:
```bash
Usage: python -m phoenix4all.downloader [OPTIONS] COMMAND [ARGS]...

  Download Phoenix model files from various sources.

Options:
  --help  Show this message and exit.

Commands:
  hiresfits  Download Phoenix model files to PATH from source 'hiresfits'.
  svo        Download Phoenix model files to PATH from source 'svo'.
  synphot    Download Phoenix model files to PATH from source 'synphot'.
```

We can select a source and see what models are available:

```bash
python -m phoenix4all.downloader svo --help
```
Which will give you:
```bash
  --mkdir / --no-mkdir            Create the output directory if it does not
                                  exist.
  --progress / --no-progress      Show download progress bar.
  --teff FLOAT
  --logg FLOAT
  --feh FLOAT
  --alpha FLOAT
  --teff-range <FLOAT FLOAT>...   Range of Teff values to download (min max).
  --logg-range <FLOAT FLOAT>...   Range of logg values to download (min max).
  --feh-range <FLOAT FLOAT>...    Range of [Fe/H] values to download (min
                                  max).
  --alpha-range <FLOAT FLOAT>...  Range of [alpha/Fe] values to download (min
                                  max).
  --model [cond00|dusty00|atmo2020_ceq|atmo2020_neq_strong|atmo2020_neq_weak|bt-cond|bt-dusty|bt-nextgen-agss2009|bt-nextgen-gns93|bt-settl|bt-settl-agss|bt-settl-cifist|bt-settl-gns93|bt-settl-2014|coelho_highres|coelho_sed|drift|koester2|Kurucz2003all|Kurucz2003|Kurucz2003alp|levenhagen17|NextGen2|NextGen]
                                  Optional model name to download (if
                                  supported by source).
  --base-url TEXT                 Optional base URL to download from (if
                                  supported by source).
  --help                          Show this message and exit.
```

Lets choose the classic **BT-Settl** model from the SVO source and download all models with effective temperatures between 5000K and 6000K, surface gravities between 0 and 2, metallicity [Fe/H] of 0.0, and alpha element enhancement [alpha/Fe] of 0.0:

```bash
python -m phoenix4all.downloader svo /path/to/download/models --model bt-settl --teff-range 5000 6000 --logg-range 0 2 --feh 0.0 --alpha 0.0 --mkdir --progress
```

This command will download the specified models to the given path, creating the directory if it doesn't exist, and showing a progress bar during the download.

Done!


## Supported Sources and Models

Right now, Phoenix4All supports the following PHOENIX models from sources:

- **HiResFITS**: High-resolution FITS files from the [PHOENIX project](https://phoenix.astro.physik.uni-goettingen.de/data/v2.0/HiResFITS/).
    - PHOENIX-ACES-AGSS-COND-2011
- **Synphot**: FITS files from [STSCI](https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/phoenix/)
    - AGSS2009
- **SVO**: ASCII files from the [Spanish Virtual Observatory](http://svo2.cab.inta-csic.es/theory/newov2/index.php)
    - AMES-COND 2000
    - AMES-DUSTY 2000
    - ATMO 2020, CEQ
    - ATMO 2020, NEQ strong
    - ATMO 2020, NEQ weak
    - BT-COND
    - BT-DUSTY
    - BT-NextGen (AGSS2009)
    - BT-NextGen (GNS93)
    - BT-Settl
    - BT-Settl (AGSS2009)
    - BT-Settl (CIFIST)
    - BT-Settl (GNS93)
    - BT-Settl 2014
    - Coelho HiRes
    - Coelho SED
    - DRIFT-PHOENIX
    - Koester WD Models
    - Kurucz 2003
    - Levenhagen 2017
    - NextGen
    - NextGen (Solar)


## Testing Regime

We use the **YOLO** testing regime which is *I wrote this to solve an issue I had but I haven't have time to write tests so use at your own risk*. So there are very few tests right now. But I will add more tests as I go along.

## TODO

- Add support for Pollux source.
- Make an auto guessing function to figure out which source to use based on available files.


## TauREx 3 plugin

Phoenix4All comes with a TauREx 3 plugin to use Phoenix4All stars as stellar sources in TauREx 3. Just import the `Phoenix4AllStar` class from `phoenix4all.taurex` and use it like any other TauREx 3 stellar source.

For Input file usage, just set the `star_type` to `phoenix4all` and provide the necessary stellar parameters:

```
[Star]
star_type = phoenix4all
temperature= 5000
radius= 1.0

# All optional but these are the defaults
distance= 1
mass= 1.0
metallicity= 1.0
alpha= 0.0
source="svo"
interpolation_mode = linear
use_planck = certainly
bounds_error = nope
# path = /path/to/phoenix/models Provide this if you've predownloaded the models
model_name = bt-settl-cifist
# logg = 4.5 # Optional, if not provided, will be calculated from mass and radius
```


## Citation

Please cite both the sources and models as well. We will be working on a proper citation guide soon and a built in citation facility in future releases.

## Contributing

For the love of God, please do. Open an issue or a pull request on GitHub. Really just adding more sources would be amazing.

## License

Phoenix4All is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
