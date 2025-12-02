import pathlib
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

from phoenix4all.net.http import (
    ColumnDetectionError,
    FetchListingError,
    aherf2filename,
    check_file_and_length,
    download_to_directory,
    fetch_listing,
    human2bytes,
    parse,
)


# Test human2bytes
@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("1B", 1),
        ("1K", 1024),
        ("1M", 1048576),
        ("1G", 1073741824),
        ("1.5K", 1536),
        (None, None),
        ("1024", 1024),
    ],
)
def test_human2bytes(input_str, expected):
    assert human2bytes(input_str) == expected


# Test aherf2filename
@pytest.mark.parametrize(
    "a_href, expected",
    [
        ("/path/to/file/", "file/"),
        ("/path/to/file", "file"),
        ("file%20name", "file name"),
    ],
)
def test_aherf2filename(a_href, expected):
    assert aherf2filename(a_href) == expected


# Test parse
def test_parse():
    html = """
    <html>
        <head><title>Index of /test</title></head>
        <body>
            <pre>
                <a href="file1.txt">file1.txt</a> 2023-01-01 12:00 1K
                <a href="file2.txt">file2.txt</a> 2023-01-02 12:00 2K
            </pre>
        </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")
    cwd, listing = parse(soup)
    assert cwd == "/test"
    assert listing[0].name == "file1.txt"
    assert listing[0].size == 1024
    assert listing[1].name == "file2.txt"
    assert listing[1].size == 2048


# Test fetch_listing
@patch("requests.get")
def test_fetch_listing(mock_get):
    mock_response = MagicMock()
    mock_response.content = b"""
    <html>
        <head><title>Index of /test</title></head>
        <body>
            <pre>
                <a href="file1.txt">file1.txt</a> 2023-01-01 12:00 1K
            </pre>
        </body>
    </html>
    """
    mock_get.return_value = mock_response

    url = "http://example.com/test"
    cwd, listing = fetch_listing(url)
    assert cwd == "/test"
    assert listing[0].name == "file1.txt"
    assert listing[0].size == 1024


# Test check_file_and_length
def test_check_file_and_length(tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("a" * 1024)
    assert check_file_and_length(file_path, 1024) is True
    assert check_file_and_length(file_path, 2048) is False
    assert check_file_and_length(pathlib.Path("nonexistent_file.txt"), 1024) is False


# Test download_to_directory
@patch("requests.get")
def test_download_to_directory(mock_get, tmp_path):
    mock_response = MagicMock()
    mock_response.iter_content = lambda chunk_size: [b"data"]
    mock_response.headers = {"Content-Length": "4"}
    mock_response.__enter__.return_value = mock_response
    mock_get.return_value = mock_response

    files = ["http://example.com/file1.txt"]
    output_paths = [tmp_path / "file1.txt"]
    downloaded_files = download_to_directory(files, output_paths, progress=False)

    assert len(downloaded_files) == 1
    assert downloaded_files[0].exists()
    assert downloaded_files[0].read_text() == "data"


# Test exceptions
def test_column_detection_error():
    with pytest.raises(ColumnDetectionError):
        raise ColumnDetectionError()


def test_fetch_listing_error():
    with pytest.raises(FetchListingError):
        raise FetchListingError()
