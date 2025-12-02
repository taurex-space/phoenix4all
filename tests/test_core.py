import pytest
import pandas as pd
import numpy as np
from astropy import units as u

from phoenix4all.sources.core import (
    PhoenixDataFile,
    WeightedPhoenixDataFile,
    construct_phoenix_dataframe,
    find_nearest_points,
    compute_weights,
    find_nearest_datafile,
    filter_parameter,
    InterpolationBoundaryError,
    NoAvailableDataError,
)

@pytest.fixture
def sample_data():
    return [
        PhoenixDataFile(teff=5000, logg=4.5, feh=0.0, alpha=0.0, filename="file1"),
        PhoenixDataFile(teff=5500, logg=4.0, feh=-0.5, alpha=0.2, filename="file2"),
        PhoenixDataFile(teff=6000, logg=4.5, feh=0.5, alpha=0.0, filename="file3"),
        PhoenixDataFile(teff=6500, logg=5.0, feh=0.0, alpha=0.1, filename="file4"),
    ]

@pytest.fixture
def sample_dataframe(sample_data):
    return construct_phoenix_dataframe(sample_data)

def test_construct_phoenix_dataframe(sample_data):
    df = construct_phoenix_dataframe(sample_data)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ["teff", "logg", "feh", "filename" , "alpha",]
    assert "filename" in df.columns
    assert len(df) == len(sample_data)

def test_find_nearest_points(sample_dataframe):
    nearest = find_nearest_points(sample_dataframe, teff=5500, logg=4.5, feh=0.0, alpha=0.0)
    assert not nearest.empty
    assert len(nearest) <= 16  # Maximum 2^4 combinations of nearest points

def test_compute_weights(sample_dataframe):
    nearest = find_nearest_points(sample_dataframe, teff=5500, logg=4.5, feh=0.0, alpha=0.0)
    weighted_files = compute_weights(nearest, teff=5500, logg=4.5, feh=0.0, alpha=0.0)
    assert len(weighted_files) > 0
    assert all(isinstance(wf, WeightedPhoenixDataFile) for wf in weighted_files)
    assert all(wf.weight > 0 for wf in weighted_files)

def test_find_nearest_datafile(sample_dataframe):
    nearest_file = find_nearest_datafile(sample_dataframe, teff=5500, logg=4.5, feh=0.0, alpha=0.0)
    assert isinstance(nearest_file, PhoenixDataFile)
    assert nearest_file.filename == "file2"  # Closest match based on the sample data

def test_filter_parameter(sample_dataframe):
    filtered = filter_parameter(sample_dataframe, "teff", (5000, 6000))
    assert not filtered.empty
    assert ((5000 <= filtered["teff"]) & (filtered["teff"] <= 6000)).all()

def test_interpolation_boundary_error():
    with pytest.raises(InterpolationBoundaryError):
        raise InterpolationBoundaryError("teff", 7000, (5000, 6500))

def test_no_available_data_error():
    with pytest.raises(NoAvailableDataError):
        raise NoAvailableDataError