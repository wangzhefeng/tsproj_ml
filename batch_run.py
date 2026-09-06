#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for recoverable canonical RawDesignGroup batch execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_pipeline.batch_runtime import (
    run_canonical_batch,
    verify_batch_results,
)


def args_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-yaml",
        action="append",
        required=True,
        help="Physical canonical model YAML; repeat for every config.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-workers", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = args_parse()
    report = run_canonical_batch(
        tuple(Path(value) for value in args.config_yaml),
        output_root=Path(args.output_root),
        config_workers=args.config_workers,
        resume=not args.no_resume,
    )
    verification = (
        verify_batch_results(report.state_path)
        if report.failed_count == 0
        else None
    )
    payload = {
        "input_count": report.input_count,
        "completed_count": report.completed_count,
        "failed_count": report.failed_count,
        "group_count": report.group_count,
        "raw_payload_load_count": report.raw_payload_load_count,
        "transform_cache": report.transform_cache,
        "peak_rss_bytes": report.peak_rss_bytes,
        "state_path": str(report.state_path),
        "verification": verification,
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 1 if report.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
