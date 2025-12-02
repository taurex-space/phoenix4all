import logging

import pytest

from phoenix4all.log import create_logger, debug_function, module_logger


def test_module_logger():
    logger_name = "test.module"
    log = module_logger(logger_name)
    assert isinstance(log, logging.Logger)
    assert log.name == logger_name


def test_create_logger():
    subname = "test_submodule"
    log = create_logger(subname)
    assert isinstance(log, logging.Logger)
    assert log.name == f"phoenix4all.{subname}"


def test_debug_function_decorator(caplog):
    @debug_function
    def sample_function(a, b):
        return a + b

    with caplog.at_level(logging.DEBUG):
        result = sample_function(2, 3)

    assert result == 5
    assert "Entering with args=(2, 3), kwargs={}" in caplog.text
    assert "Exiting with result=5" in caplog.text


def test_debug_function_decorator_exception(caplog):
    @debug_function
    def sample_function_with_exception():
        raise ValueError("Test exception")  # noqa: TRY003

    with caplog.at_level(logging.DEBUG), pytest.raises(ValueError, match="Test exception"):
        sample_function_with_exception()

    assert "Entering with args=(), kwargs={}" in caplog.text
    assert "Exception in sample_function_with_exception" in caplog.text
