"""Resolve source-checkout defaults without depending on the launch directory."""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent


def project_root() -> Path:
    """Use the checkout when present, otherwise the caller's current directory.

    Wheels contain algorithms, not cohort metadata. Installed commands accept
    explicit metadata paths; their defaults are relative to the working project.
    """
    checkout = PACKAGE_ROOT.parents[1]
    if (checkout / "pyproject.toml").is_file() and (checkout / "metadata").is_dir():
        return checkout
    return Path.cwd().resolve()


PROJECT_ROOT = project_root()
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT.parent / f"{PROJECT_ROOT.name}-outputs"
