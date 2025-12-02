# Phoenix4All

[![Release](https://img.shields.io/github/v/release/taurex-space/phoenix4all)](https://img.shields.io/github/v/release/taurex-space/phoenix4all)
[![Build status](https://img.shields.io/github/actions/workflow/status/taurex-space/phoenix4all/main.yml?branch=main)](https://github.com/taurex-space/phoenix4all/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/taurex-space/phoenix4all)](https://img.shields.io/github/commit-activity/m/taurex-space/phoenix4all)
[![License](https://img.shields.io/github/license/taurex-space/phoenix4all)](https://img.shields.io/github/license/taurex-space/phoenix4all)

## What is it?

Phoenix4All is a Python library that provides an easy-to-use interface for accessing and utilizing the PHOENIX stellar atmosphere models and synthetic spectra. It attempts to solve the infuriatingly complex process of downloading, managing, and interpolating PHOENIX models by providing a simple and efficient API that does it for you. It has been especially designed for astronomers and astrophysicists who need to work with stellar spectra in their research. But its really designed to make your life easier if you need to use PHOENIX models.

## Features

- **Lazy loading**: Download and cache models on-the-fly as needed.
- **Downloader**: A downloader to grab the models if you want to manage them yourself.
- **Interpolation**: Linearly interpolate between models to get spectra for arbitrary stellar parameters.
- **Multiple sources**: Support for different PHOENIX model sources, including high-resolution FITS files and lower-resolution ASCII files.
- **Minimal memory usage**: Only needed spectral files are downloaded then loaded into memory.
- **Command-line interface**: A CLI tool to quickly fetch and manage models from the terminal.
