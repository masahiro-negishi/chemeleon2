"""Contract test for pyright configuration.

This test validates that pyproject.toml contains a valid pyright configuration
with strict type checking settings as per project requirements.
"""

import tomllib
from pathlib import Path

import pytest


@pytest.fixture
def pyproject_toml_path():
    """Return the path to pyproject.toml at repository root."""
    return Path(__file__).parent.parent.parent / "pyproject.toml"


@pytest.fixture
def pyproject_data(pyproject_toml_path):
    """Load and parse pyproject.toml."""
    with open(pyproject_toml_path, "rb") as f:
        return tomllib.load(f)


def test_pyproject_toml_exists(pyproject_toml_path):
    """Test that pyproject.toml exists at repository root."""
    assert pyproject_toml_path.exists(), "pyproject.toml not found"


def test_pyproject_toml_valid_syntax(pyproject_data):
    """Test that pyproject.toml has valid TOML syntax."""
    assert pyproject_data is not None
    assert isinstance(pyproject_data, dict)


def test_pyright_section_exists(pyproject_data):
    """Test that [tool.pyright] section exists."""
    assert "tool" in pyproject_data, "[tool] section missing"
    assert "pyright" in pyproject_data["tool"], "[tool.pyright] section missing"


def test_pyright_type_checking_mode(pyproject_data):
    """Test that typeCheckingMode is set to 'strict'."""
    pyright_config = pyproject_data["tool"]["pyright"]
    assert "typeCheckingMode" in pyright_config, "typeCheckingMode not configured"
    assert pyright_config["typeCheckingMode"] == "strict", (
        f"Expected typeCheckingMode='strict', got '{pyright_config['typeCheckingMode']}'"
    )


def test_pyright_python_version(pyproject_data):
    """Test that pythonVersion is set to '3.11'."""
    pyright_config = pyproject_data["tool"]["pyright"]
    assert "pythonVersion" in pyright_config, "pythonVersion not configured"
    assert pyright_config["pythonVersion"] == "3.11", (
        f"Expected pythonVersion='3.11', got '{pyright_config['pythonVersion']}'"
    )


def test_pyright_unknown_type_warnings(pyproject_data):
    """Test that unknown type warnings are configured at warning level."""
    pyright_config = pyproject_data["tool"]["pyright"]

    # Check all four unknown type warning settings
    warning_settings = [
        "reportMissingTypeStubs",
        "reportUnknownMemberType",
        "reportUnknownArgumentType",
        "reportUnknownVariableType",
    ]

    for setting in warning_settings:
        assert setting in pyright_config, f"{setting} not configured in pyright config"
        assert pyright_config[setting] == "warning", (
            f"Expected {setting}='warning', got '{pyright_config[setting]}'"
        )
