# -*- coding: utf-8 -*-
"""Compatibility shim for legacy imports.

The canonical AIDC computility entrypoint lives in ``scripts/aidc/computility_process.py``.
This module re-exports that implementation so existing imports under ``data_provider``
continue to work.
"""

from scripts.aidc.computility_process import *  # noqa: F401,F403
