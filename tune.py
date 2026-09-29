"""Canonical 单模型自动调参入口；搜索配方不替代模型 YAML。"""
import argparse
from pathlib import Path

from config.config_loader import load_yaml_config, load_yaml_document
from forecasting_core.specs import ForecastConfigSpec
from model_tuning.runtime import run_tuning
from model_tuning.specs import TuningSpec
from utils.runtime_env import ensure_runtime_environment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-yaml", required=True, type=Path)
    parser.add_argument("--search-yaml", required=True, type=Path)
    parser.add_argument("--study-dir", required=True, type=Path, help="New experiment directory; existing paths are rejected")
    args = parser.parse_args()
    ensure_runtime_environment()
    config = load_yaml_config(args.config_yaml)
    if not isinstance(config, ForecastConfigSpec):
        raise TypeError("tune.py supports canonical single-model configs only")
    search = TuningSpec.from_mapping(load_yaml_document(args.search_yaml))
    report = run_tuning(config, search, study_dir=args.study_dir)
    print(f"completed: best_trial={report['best_trial']} best_yaml={report['best_yaml']}")


if __name__ == "__main__":
    main()
