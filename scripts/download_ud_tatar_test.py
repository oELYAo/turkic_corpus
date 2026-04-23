#!/usr/bin/env python3
"""Download UD Tatar-NMCTT test split (only public split in the UD repo)."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

DEFAULT_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/UD_Tatar-NMCTT/"
    "master/tt_nmctt-ud-test.conllu"
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "ud" / "tt_nmctt-ud-test.conllu",
    )
    p.add_argument("--url", default=DEFAULT_URL)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(args.url, timeout=120)
    r.raise_for_status()
    args.out.write_bytes(r.content)
    print(f"Wrote {args.out} ({len(r.content)} bytes)")


if __name__ == "__main__":
    main()
