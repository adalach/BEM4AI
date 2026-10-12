"""Download the Copernicus BBH 2021 files listed in the stage 1 manifest.

Notebook 1 writes the manifest (``data/raw/CLMS/UA_BBH_2021/manifest.json``) from the
public catalogue; this script downloads the files it lists that are not on disk yet,
with a free Copernicus Data Space Ecosystem (CDSE) account. Re-running skips what
is already there.

Usage, from the repository root::

    python scripts/download_clms_bbh2021.py
    CDSE_TOTP=123456 python scripts/download_clms_bbh2021.py   # with two-factor login

The CDSE login is read from ``~/.config/cdse/credentials.json``
(``{"username": ..., "password": ...}``, readable only by you); it is never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from helpers.clms_bbh import download_files, missing_files  # noqa: E402

DEFAULT_DIR = ROOT / "data" / "raw" / "CLMS" / "UA_BBH_2021"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_DIR / "manifest.json")
    parser.add_argument("--files", type=Path, default=DEFAULT_DIR / "files")
    parser.add_argument("--credentials", type=Path,
                        default=Path.home() / ".config" / "cdse" / "credentials.json")
    args = parser.parse_args()

    if not args.manifest.exists():
        sys.exit(f"{args.manifest} not found: run the Copernicus BBH cell of BEM4AI_1_Cities.ipynb first.")
    manifest = json.loads(args.manifest.read_text())
    todo = missing_files(manifest, args.files)
    print(f"{len(manifest['files'])} files in the manifest, {len(todo)} to download "
          f"({sum(f['bytes'] for f in todo) / 1e6:.0f} MB).")
    if todo and not args.credentials.exists():
        sys.exit(f"{args.credentials} not found: save your CDSE login there first.")
    added = download_files(manifest, args.files, args.credentials, totp=os.environ.get("CDSE_TOTP"))
    print(f"Downloaded {added}; {len(missing_files(manifest, args.files))} still missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
