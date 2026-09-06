"""拆分接缝的行为合同：窗口顺序、并行参数、状态与异常传播。"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from forecasting_core.specs import FixedStepBacktestSpec
from model_pipeline.lifecycle import run_lifecycle
from model_pipeline.runner import CanonicalBaseModelRunner
from model_testing.contracts import BacktestRunner, FoldScoringRunner
from model_testing.fixed_step import run_fixed_step_backtest
from model_testing.scoring import FoldScoreResult

ROOT = Path(__file__).resolve().parents[1]


class BacktestLifecycleSplitTest(unittest.TestCase):
    def test_runner_supplies_declared_protocol_methods(self):
        for protocol in (FoldScoringRunner, BacktestRunner):
            for name, value in vars(protocol).items():
                if not name.startswith("_") and callable(value):
                    self.assertTrue(callable(getattr(CanonicalBaseModelRunner, name, None)), name)
        tree = ast.parse((ROOT / "model_pipeline/runner.py").read_text())
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        self.assertNotIn("_execute_lifecycle", names)
        self.assertNotIn("_run", names)

    def test_running_completed_and_failed_order_with_base_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            config = SimpleNamespace(fingerprint=lambda: "fixture")
            runner = SimpleNamespace(config=config)
            state_path = model_dir / "run_state.json"

            def execute(*args):
                self.assertEqual(json.loads(state_path.read_text())["status"], "running")
                return "result"

            with patch("model_pipeline.lifecycle._output_paths", return_value=(model_dir,) * 4):
                with patch("model_pipeline.lifecycle.execute_lifecycle", side_effect=execute):
                    self.assertEqual(run_lifecycle(runner), "result")
                self.assertEqual(json.loads(state_path.read_text())["status"], "completed")
                error = KeyboardInterrupt("test cancellation")
                with patch("model_pipeline.lifecycle.execute_lifecycle", side_effect=error):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        run_lifecycle(runner)
                self.assertIs(caught.exception, error)
                self.assertEqual(json.loads(state_path.read_text())["status"], "failed")

    def test_serial_and_parallel_folds_preserve_order_and_history_contract(self):
        # 合成接缝输入，不作为模型数值或性能证据。
        for workers in (1, 2):
            with self.subTest(workers=workers), tempfile.TemporaryDirectory() as directory:
                backtest = FixedStepBacktestSpec(
                    history_steps=10, train_window_steps=3, fold_count=2, stride_steps=2,
                )
                validation = dict(aggregate_weighting=None)
                validation = type("Validation", (dict,), {"backtest": backtest})(validation)
                config = SimpleNamespace(
                    problem=SimpleNamespace(targets=("load",)), validation=validation,
                )
                windows = tuple(SimpleNamespace(
                    window=i + 1, origin=pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
                    origin_index=i, train_indices=(i,), metadata={"window": i + 1},
                ) for i in range(2))
                builder = SimpleNamespace(audit=(), reset_audit=Mock())
                runner = Mock(spec=BacktestRunner,
                    config=config, builder=builder, execution_plan=SimpleNamespace(window_workers=workers),
                    backtest_windows=lambda: windows,
                    backtest_target_histories=Mock(return_value=("history0", "history1")),
                    fit=Mock(side_effect=lambda indices, **kwargs: (indices, kwargs)),
                    runtime_resources_payload=lambda: {},
                )
                scored = []

                def score(**kwargs):
                    scored.append(kwargs)
                    return FoldScoreResult(
                        window=kwargs["window"], origin=kwargs["origin"],
                        frame=pd.DataFrame({"window": [kwargs["window"]]}),
                        point_scores=pd.DataFrame({"score": [1.0]}),
                    )

                with patch("model_testing.fixed_step.score_holdout_fold", side_effect=score), patch(
                    "model_testing.fixed_step.write_backtest_results"
                ) as write:
                    metadata, tracker, audit = run_fixed_step_backtest(runner, Path(directory), mode="point")
                self.assertEqual([x["window"] for x in scored], [1, 2])
                self.assertEqual(write.call_args.args[1]["window"].tolist(), [1, 2])
                assert metadata is not None
                self.assertEqual(metadata["mode"], "fixed_steps")
                self.assertIsNone(tracker)
                self.assertEqual(audit, ())
                for i, item in enumerate(scored):
                    self.assertEqual(item["fit_result"][0], (i,))
                    self.assertEqual(item["fit_result"][1],
                                     {"target_history": f"history{i}", "force_serial": True} if workers > 1 else {})
                self.assertEqual(runner.backtest_target_histories.call_count, int(workers > 1))


if __name__ == "__main__":
    unittest.main()
