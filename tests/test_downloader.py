import pytest
from click.testing import CliRunner

from phoenix4all.downloader import initialize_source_commands, main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_sources(mocker):
    # Mock the list_sources function to return a fixed list of sources
    mocker.patch("phoenix4all.sources.list_sources", return_value=["mock_source"])
    # Mock the find_source function to return a mock source class
    mock_source_class = mocker.Mock()
    mock_source_class.available_models.return_value = ["model1", "model2"]
    mock_source_class.download_model = mocker.Mock()

    mocker.patch("phoenix4all.sources.find_source", return_value=mock_source_class)
    initialize_source_commands()
    return mock_source_class


def test_main_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Download Phoenix model files from various sources." in result.output


def test_source_command_help(runner, mock_sources):
    # Re-initialize commands with mocked sources
    result = runner.invoke(main, ["mock_source", "--help"])
    assert result.exit_code == 0
    assert "Download Phoenix model files to PATH from source 'mock_source'." in result.output


def test_source_command_download(runner, mock_sources, tmp_path):
    output_dir = tmp_path / "output"
    result = runner.invoke(
        main,
        [
            "mock_source",
            str(output_dir),
            "--mkdir",
            "--teff",
            "5000",
            "--logg",
            "4.0",
            "--feh",
            "0.0",
            "--alpha",
            "0.0",
            "--model",
            "model1",
            "--progress",
        ],
    )

    mock_sources.download_model.assert_called_once_with(
        output_dir=output_dir,
        teff=5000,
        logg=4.0,
        feh=0.0,
        alpha=0.0,
        mkdir=True,
        model_name="model1",
        progress=True,
        base_url=None,
    )
    assert result.exit_code == 0


def test_source_command_download_all_flags(runner, mock_sources, tmp_path):
    output_dir = tmp_path / "output"
    result = runner.invoke(
        main,
        [
            "mock_source",
            str(output_dir),
            "--mkdir",
            "--all-teff",
            "--all-logg",
            "--all-feh",
            "--all-alpha",
            "--model",
            "model2",
        ],
    )
    assert result.exit_code == 0
    mock_sources.download_model.assert_called_once_with(
        output_dir=output_dir,
        teff="all",
        logg="all",
        feh="all",
        alpha="all",
        mkdir=True,
        model_name="model2",
        progress=False,
        base_url=None,
    )


def test_source_command_invalid_model(runner, mock_sources, tmp_path):
    output_dir = tmp_path / "output"
    result = runner.invoke(
        main,
        [
            "mock_source",
            str(output_dir),
            "--model",
            "invalid_model",
        ],
    )
    assert result.exit_code != 0
    assert "Invalid value for '--model'" in result.output
