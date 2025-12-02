import pytest

from phoenix4all.io import ZIPJSON_KEY, get_package_download_cache_dir, json_unzip, json_zip


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


def test_get_package_download_cache_dir():
    # Mock the astropy.config.paths.get_cache_dir_path function
    # Test get_package_download_cache_dir
    cache_dir = get_package_download_cache_dir()
    assert ".phoenix4all" in str(cache_dir)
