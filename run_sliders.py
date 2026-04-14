#!/usr/bin/env python3
"""Thin wrapper that delegates to :func:`sliders.cli.main`.

Keeps the repo-relative command ``uv run python run_sliders.py ...`` working
for people who have cloned the source tree. After ``pip install sliders``
you can use the installed ``sliders`` console script instead.
"""

from sliders.cli import main

if __name__ == "__main__":
    main()
