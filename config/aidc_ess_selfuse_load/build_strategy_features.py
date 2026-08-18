"""ESS 策略特征 v2 薄命令行入口。"""

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.aidc_ess_selfuse_load.strategy_features.pipeline import (
    build_strategy_features,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build causal ESS strategy features v2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--force", action="store_true")
    args = parser.parse_args()

    results = build_strategy_features(
        args.config,
        data_root=args.data_root,
        validate_only=args.validate_only,
        force=args.force,
    )
    mode_name = "validated" if args.validate_only else "written"
    for route, result in results.items():
        print(
            f"route {route}: {mode_name}; history={len(result.history)}, "
            f"future={len(result.future)}, lag_ready={int(result.future['lag_feature_ready'].sum())}, "
            f"template_ready={int(result.future['template_feature_ready'].sum())}"
        )


if __name__ == "__main__":
    main()
