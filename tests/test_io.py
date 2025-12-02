import pytest
import base64
import json
import zlib
from pathlib import Path
from phoenix4all.io import json_zip, json_unzip, get_package_download_cache_dir, ZIPJSON_KEY, ZIPJSON_TYPE

def test_json_zip_and_unzip():
    # Test data
    original_data = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}
    
    # Test json_zip
    compressed_data = json_zip(original_data)
    assert isinstance(compressed_data, dict)
    assert ZIPJSON_KEY in compressed_data
    assert isinstance(compressed_data[ZIPJSON_KEY], str)
    
    # Test json_unzip
    decompressed_data = json_unzip(compressed_data)
    assert decompressed_data == original_data


def test_json_zip_empty_dict():
    # Test with an empty dictionary
    original_data = {}
    compressed_data = json_zip(original_data)
    decompressed_data = json_unzip(compressed_data)
    assert decompressed_data == original_data


def test_json_zip_invalid_input():
    # Test with invalid input for json_unzip
    invalid_data = {"invalid_key": "invalid_value"}
    with pytest.raises(KeyError):
        json_unzip(invalid_data)  # Should raise KeyError due to missing ZIPJSON_KEY


def test_get_package_download_cache_dir(monkeypatch):
    # Mock the astropy.config.paths.get_cache_dir_path function
    mock_cache_dir = Path("/mock/cache/dir")
    
    def mock_get_cache_dir_path(package_name):
        assert package_name == "phoenix4all"
        return mock_cache_dir
    
    monkeypatch.setattr("astropy.config.paths.get_cache_dir_path", mock_get_cache_dir_path)
    
    # Test get_package_download_cache_dir
    cache_dir = get_package_download_cache_dir()
    assert cache_dir == mock_cache_dir