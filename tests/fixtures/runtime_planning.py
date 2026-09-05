"""资源测试的最小 workload；只调用正式 planner，不复制资源决策规则。"""

from model_forecasting.resource_planner import (
    build_runtime_workload,
    plan_runtime_execution,
    runtime_estimator_params,
)


def workload_for_config(config):
    # 不加载真实资产的几何探针；不能当成正式运行性能证据。
    return build_runtime_workload(
        config, training_rows=0, feature_count=1, design_bytes=0,
    )


def plan_for_config(config):
    return plan_runtime_execution(config, workload_for_config(config))


def estimator_params(config):
    return runtime_estimator_params(config, plan_for_config(config))


def fit_worker_plan(config):
    plan = plan_for_config(config)
    return plan.quantile_workers, plan.output_workers


def model_workers(config):
    return plan_for_config(config).output_workers


def scalar_fit_count(config):
    return workload_for_config(config).logical_output_count
