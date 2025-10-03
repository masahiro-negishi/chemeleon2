"""Contract test for setup script.

This test validates that setup-dev.sh exists, is executable, and
performs the required setup steps correctly.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def setup_script_path():
    """Return the path to setup-dev.sh at repository root."""
    return Path(__file__).parent.parent.parent / "setup-dev.sh"


def test_setup_script_exists(setup_script_path):
    """Test that setup-dev.sh exists at repository root."""
    assert setup_script_path.exists(), "setup-dev.sh not found"


def test_setup_script_is_executable(setup_script_path):
    """Test that setup-dev.sh is executable."""
    assert os.access(setup_script_path, os.X_OK), (
        "setup-dev.sh is not executable (run: chmod +x setup-dev.sh)"
    )


def test_setup_script_has_shebang(setup_script_path):
    """Test that setup-dev.sh has a proper shebang."""
    with open(setup_script_path) as f:
        first_line = f.readline().strip()
    assert first_line.startswith("#!"), (
        "setup-dev.sh missing shebang (should start with #!/bin/bash)"
    )
    assert "bash" in first_line.lower(), "setup-dev.sh should use bash"


def test_setup_script_checks_python_version(setup_script_path):
    """Test that setup-dev.sh checks Python version."""
    with open(setup_script_path) as f:
        content = f.read()

    # Should check Python version
    assert "python" in content.lower(), "Script should check Python version"
    assert "version" in content.lower(), "Script should check Python version"


def test_setup_script_checks_uv(setup_script_path):
    """Test that setup-dev.sh checks for uv package manager."""
    with open(setup_script_path) as f:
        content = f.read()

    assert "uv" in content.lower(), "Script should check for uv package manager"
    assert "command -v uv" in content or "which uv" in content, (
        "Script should check if uv is installed"
    )


def test_setup_script_installs_precommit(setup_script_path):
    """Test that setup-dev.sh installs pre-commit."""
    with open(setup_script_path) as f:
        content = f.read()

    assert "pre-commit" in content, "Script should install pre-commit framework"
    assert "uv pip install" in content and "pre-commit" in content, (
        "Script should use 'uv pip install' to install pre-commit"
    )


def test_setup_script_installs_ruff(setup_script_path):
    """Test that setup-dev.sh installs Ruff."""
    with open(setup_script_path) as f:
        content = f.read()

    assert "ruff" in content.lower(), "Script should install Ruff"
    assert "uv pip install" in content and "ruff" in content.lower(), (
        "Script should use 'uv pip install' to install Ruff"
    )


def test_setup_script_installs_pyright(setup_script_path):
    """Test that setup-dev.sh installs pyright."""
    with open(setup_script_path) as f:
        content = f.read()

    assert "pyright" in content.lower(), "Script should install pyright"
    assert "uv pip install" in content and "pyright" in content.lower(), (
        "Script should use 'uv pip install' to install pyright"
    )


def test_setup_script_runs_precommit_install(setup_script_path):
    """Test that setup-dev.sh runs 'pre-commit install'."""
    with open(setup_script_path) as f:
        content = f.read()

    assert "pre-commit install" in content, "Script should run 'pre-commit install'"


def test_setup_script_has_exit_codes(setup_script_path):
    """Test that setup-dev.sh has proper exit codes for different failures."""
    with open(setup_script_path) as f:
        content = f.read()

    # Should have exit codes for different failure scenarios
    assert "exit 1" in content, (
        "Script should exit with code 1 for Python version failure"
    )
    assert "exit 2" in content, (
        "Script should exit with code 2 for installation failures"
    )
    assert "exit 3" in content, (
        "Script should exit with code 3 for pre-commit install failure"
    )


def test_setup_script_idempotent(setup_script_path):
    """Test that setup-dev.sh can be run multiple times without errors.

    This test verifies idempotency by checking that the script uses
    commands that are safe to run multiple times (uv pip install, pre-commit install).
    """
    with open(setup_script_path) as f:
        content = f.read()

    # Script should use uv pip install (which is idempotent)
    assert "uv pip install" in content, "Script should use 'uv pip install'"

    # Script should use pre-commit install (which is idempotent)
    assert "pre-commit install" in content, "Script should use pre-commit install"

    # Should not use destructive commands that would fail on second run
    destructive_commands = ["rm -rf", "git init", "mkdir -p"]
    for cmd in destructive_commands:
        if cmd in content:
            # If mkdir -p is used, that's fine (it's idempotent)
            if cmd == "mkdir -p":
                continue
            pytest.fail(f"Script contains potentially non-idempotent command: {cmd}")


def test_setup_script_creates_git_hooks(setup_script_path):
    """Test that running setup-dev.sh creates .git/hooks/pre-commit.

    This is a validation test that checks if the pre-commit hook exists.
    If setup-dev.sh has been run, the hook should exist.
    If it doesn't exist, we skip the test (setup may not have been run yet).
    """
    git_dir = setup_script_path.parent / ".git"
    if not git_dir.exists():
        pytest.skip("Not in a git repository")

    hooks_dir = git_dir / "hooks"
    precommit_hook = hooks_dir / "pre-commit"

    # Check if hook exists - if not, skip (setup may not have been run)
    if not precommit_hook.exists():
        pytest.skip(
            "pre-commit hook not found - setup-dev.sh may not have been run yet"
        )

    # If hook exists, it should be executable
    assert os.access(precommit_hook, os.X_OK), (
        "pre-commit hook exists but is not executable"
    )
