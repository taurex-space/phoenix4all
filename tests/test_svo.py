import pathlib
from unittest.mock import MagicMock, patch

import pytest

from phoenix4all.sources.svo import (
    PhoenixDataFile,
    SVOModel,
    convert_filename_to_datafile,
    create_filename,
    find_datasets_in_path,
    list_available_models,
    list_datasets_from_url,
)


@pytest.fixture
def mock_phoenix_datafile():
    return PhoenixDataFile(teff=5000, logg=4.5, feh=0.0, alpha=0.0, filename="mock_file.txt")


def test_create_filename(mock_phoenix_datafile):
    model_id = "bt-settl"
    expected_filename = "svo_bt-settl_T05000_g4.50_m+0.00_a0.00.txt"
    assert create_filename(model_id, mock_phoenix_datafile) == expected_filename


def test_convert_filename_to_datafile():
    filename = "svo_bt-settl_T05000_g4.50_m+0.00_a0.00.txt"
    model_id = "bt-settl"
    datafile = convert_filename_to_datafile(filename, model_id)
    assert datafile.teff == 5000
    assert datafile.logg == 4.5
    assert datafile.feh == 0.0
    assert datafile.alpha == 0.0
    assert datafile.filename == filename


def test_convert_filename_to_datafile_invalid():
    filename = "invalid_filename.txt"
    model_id = "bt-settl"
    assert convert_filename_to_datafile(filename, model_id) is None


@patch("pathlib.Path.glob")
def test_find_datasets_in_path(mock_glob, mock_phoenix_datafile):
    mock_glob.return_value = [pathlib.Path("svo_bt-settl_T05000_g4.50_m+0.00_a0.00.txt")]
    path = pathlib.Path("/mock/path")
    model_id = "bt-settl"
    datasets = find_datasets_in_path(path, model_id)
    assert len(datasets) == 1
    assert datasets[0].teff == 5000
    assert datasets[0].logg == 4.5
    assert datasets[0].feh == 0.0
    assert datasets[0].alpha == 0.0


@patch("requests.get")
def test_list_available_models(mock_get):
    mock_response = MagicMock()
    mock_response.text = """
    <html>
        <body>
            <select name="reqmodels[]">
                <option value="bt-settl">BT-Settl</option>
                <option value="cond00">Cond00</option>
            </select>
        </body>
    </html>
    """
    mock_get.return_value = mock_response

    models = list_available_models()
    assert len(models) == 2
    assert models[0] == SVOModel(id="bt-settl", long_name="BT-Settl")
    assert models[1] == SVOModel(id="cond00", long_name="Cond00")


@patch("requests.post")
@patch("phoenix4all.sources.svo._determine_property_indicies")
@patch("phoenix4all.sources.svo._parse_data_row")
def test_list_datasets_from_url(mock_parse_data_row, mock_determine_indices, mock_post):
    mock_response = MagicMock()
    mock_response.text = "<html><table><tr></tr></table></html>"
    mock_post.return_value = mock_response
    mock_determine_indices.return_value = {"Teff": 0, "Logg": 1, "Metallicity": 2, "Alpha": 3}
    mock_parse_data_row.return_value = PhoenixDataFile(
        teff=5000, logg=4.5, feh=0.0, alpha=0.0, filename="mock_file.txt"
    )

    datasets = list_datasets_from_url("bt-settl")
    assert len(datasets) == 1
    assert datasets[0].teff == 5000
    assert datasets[0].logg == 4.5
    assert datasets[0].feh == 0.0
    assert datasets[0].alpha == 0.0
