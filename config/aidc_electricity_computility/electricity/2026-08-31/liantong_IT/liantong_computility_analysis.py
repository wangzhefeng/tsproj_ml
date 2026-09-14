"""联通算力与电力目标的离线关联探索；不筛选模型配置，不宣称预测收益。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
DATA = ROOT / "dataset/aidc_electricity_computility/electricity/2026-08-31/liantong_IT"
LAGS = (0, 1, 3, 6, 12, 36, 72, 144, 288, 576)


def feature_names() -> list[str]:
    """固定场景基础特征合同；避免将质量、目标或新增未知列静默纳入。"""
    result = []
    for source in ("training", "inference"):
        for metric in ("cpu_util", "gpu_util", "memory_amount", "memory_total", "memory_util",
                       "gpu_memory_amount", "gpu_memory_total", "gpu_memory_util", "gpu_power_usage"):
            if metric.endswith("_util"):
                suffixes = ("sample_mean", "sample_std", "job_count")
            else:
                unit = "kw" if metric == "gpu_power_usage" else "raw"
                suffixes = (f"sum_{unit}", f"sample_mean_{unit}", "job_count")
            result.extend(f"{source}_{metric}_{suffix}" for suffix in suffixes)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(input_csv: Path, raw_power_csv: Path) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(input_csv, float_precision="round_trip")
    raw = pd.read_csv(raw_power_csv, usecols=["time", "value"], float_precision="round_trip")
    if set(frame.columns) != {"time", "value", *feature_names()}:
        raise ValueError("拼接表必须且只能包含 time/value 和固定的 54 个基础特征")
    for table in (frame, raw):
        table["time"] = pd.to_datetime(table.time, errors="raise")
        index = pd.DatetimeIndex(table.time)
        if (len(index) < 3 or index.hasnans or index.tz is not None
                or index[0].floor("5min") != index[0]
                or not index.equals(pd.date_range(index[0], index[-1], freq="5min"))):
            raise ValueError("输入必须为完整递增、无重复的本地 5min 时间轴")
    if not frame.time.equals(raw.time):
        raise ValueError("电力原表和拼接表时间轴不一致")
    if not np.isfinite(frame.drop(columns="time").to_numpy(dtype=float)).all():
        raise ValueError("拼接表存在非有限值")
    observed = raw.value.notna()
    if not observed.any() or not np.isfinite(raw.loc[observed, "value"]).all():
        raise ValueError("电力原表必须包含有限的非补值目标")
    if not np.array_equal(frame.loc[observed, "value"].to_numpy(), raw.loc[observed, "value"].to_numpy()):
        raise ValueError("拼接表改变了电力原表非补值目标")
    return frame, observed


def correlation(x: pd.Series, y: pd.Series, mask: pd.Series | None = None, minimum: int = 3) -> dict:
    valid = x.notna() & y.notna()
    if mask is not None:
        valid &= mask
    a, b = x.loc[valid], y.loc[valid]
    status = "insufficient" if len(a) < minimum else "constant" if a.nunique() < 2 or b.nunique() < 2 else "ok"
    return {"n": len(a), "pearson": a.corr(b) if status == "ok" else np.nan,
            "spearman": a.corr(b, method="spearman") if status == "ok" else np.nan, "status": status}


def daily_relationship(x, y, observed, date_groups, feature, lag, global_spearman):
    rows, coefficients = [], []
    for date, indices in date_groups.items():
        stats = correlation(x.loc[indices], y.loc[indices], observed.loc[indices], minimum=24)
        rows.append({"feature": feature, "date": str(date), "lag_steps": lag, **stats})
        if stats["status"] == "ok":
            coefficients.append(stats["spearman"])
    summary = {"valid_days": len(coefficients),
               "daily_spearman_median": float(np.median(coefficients)) if coefficients else np.nan,
               "daily_sign_agreement": float(np.mean(np.sign(coefficients) == np.sign(global_spearman))) if coefficients else np.nan}
    return rows, summary


def analyze(frame: pd.DataFrame, observed: pd.Series) -> dict[str, pd.DataFrame]:
    """先按完整网格移位，再掩码；所有排名仅作描述性 EDA。"""
    features = feature_names()
    y, dates = frame.value, frame.time.dt.date
    date_groups = frame.groupby(dates).groups
    summaries, daily, lagged = [], [], []
    difference_mask = observed & observed.shift(1, fill_value=False)
    for feature in features:
        x = frame[feature]
        row = {"feature": feature, "zero_fraction": float(x.eq(0).mean()), "unique_values": x.nunique()}
        # 非零子集只诊断零值驱动，不能等同于完整观测子集。
        for prefix, a, b, mask in (
            ("observed", x, y, observed), ("all", x, y, None),
            ("diff", x.diff(), y.diff(), difference_mask),
            ("nonzero", x, y, observed & x.ne(0)),
            ("within_day", x.where(observed) - x.where(observed).groupby(dates).transform("mean"),
             y.where(observed) - y.where(observed).groupby(dates).transform("mean"), observed),
        ):
            row.update({f"{prefix}_{key}": value for key, value in correlation(a, b, mask).items()})
        daily_rows, stability = daily_relationship(x, y, observed, date_groups, feature, 0, row["observed_spearman"])
        daily.extend(daily_rows)
        row.update(stability)
        summaries.append(row)
        for lag in LAGS:
            shifted = x.shift(lag)
            stats = {"feature": feature, "lag_steps": lag, "lag_minutes": lag * 5,
                     **correlation(shifted, y, observed)}
            stats.update({f"diff_{key}": value for key, value in correlation(shifted.diff(), y.diff(), difference_mask).items()})
            if lag in (288, 576):
                daily_rows, stability = daily_relationship(shifted, y, observed, date_groups, feature, lag, stats["spearman"])
                daily.extend(daily_rows)
                stats.update(stability)
            lagged.append(stats)
    summary = pd.DataFrame(summaries)
    lags = pd.DataFrame(lagged)
    same_time = summary.sort_values("observed_spearman", key=lambda x: x.abs(), ascending=False, kind="stable")
    day_lag = lags.loc[lags.lag_steps.eq(288)].sort_values("spearman", key=lambda x: x.abs(), ascending=False, kind="stable")
    redundancy = []
    matrix = frame.loc[observed, features].corr(method="spearman")
    for i, first in enumerate(features):
        for second in features[i + 1:]:
            coefficient = matrix.loc[first, second]
            if abs(coefficient) >= 0.95:
                redundancy.append({"feature_a": first, "feature_b": second, "spearman": coefficient})
    return {"feature_summary": summary, "lag_correlations": lags,
            "daily_correlations": pd.DataFrame(daily),
            "redundant_pairs": pd.DataFrame(redundancy, columns=["feature_a", "feature_b", "spearman"]),
            "ranking_contemporaneous": same_time, "ranking_day_lag": day_lag}


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """不为报告引入 tabulate 依赖。"""
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        values = ["N/A" if pd.isna(v) else f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(tables: dict, audit: dict) -> str:
    same = tables["ranking_contemporaneous"].head(12)
    lag = tables["ranking_day_lag"].head(12)
    return "\n\n".join([
        "# 联通 IT 算力—电力关联探索",
        f"时间：{audit['start']} 至 {audit['end']}；{audit['rows']} 行、{audit['features']} 个特征。目标 value 单位 kW。",
        f"主分析排除原表目标缺失的 {audit['excluded_target_rows']} 个标签时刻，涉及日期：{', '.join(audit['excluded_dates']) or '无'}。这不是模型计分规则变更。",
        "## 1. 同期关系：按绝对 Spearman 排名前 12",
        markdown_table(same, ["feature", "observed_pearson", "observed_spearman", "diff_spearman", "within_day_spearman", "daily_spearman_median", "daily_sign_agreement"]),
        "Pearson 衡量线性关系；Spearman 衡量单调关系。diff 是双方 5min 一阶差分；within_day 去除各自当日均值，仅作回看诊断，不是可实盘特征。daily_sign_agreement 是有效日的 Spearman 符号与全期符号一致比例；每天至少 24 个有效成对样本且双方非常数才计入。",
        "## 2. 前一天算力与当前目标：按绝对 Spearman 排名前 12",
        markdown_table(lag, ["feature", "lag_steps", "n", "pearson", "spearman", "diff_spearman", "daily_spearman_median", "daily_sign_agreement"]),
        "定义为 corr(X[t-288], y[t])，不是 X[t+288]。完整输出包含 0/1/3/6/12/36/72/144/288/576 步。当前预测为次日整日 288 点：288/576 步满足日期边界；采集发布时间尚待确认。短 lag/同期相关不能直接代表次日预测可用性。",
        "## 3. 如何解读与筛选",
        "- 同期高、差分或日内去均值后显著减弱：优先怀疑共同日间水平/任务状态，不直接认定跟踪短期负荷变化。\n- 全样本与排除补值、非零子集差异很大：检查补值和零任务区间的驱动作用。非零不是完整采集证明；本数据无记录按用户合同补零。\n- 分日符号不稳定或前一天相关弱：只能作为候选；不要用全月排名直接筛完再宣称回测无泄漏。\n- job_count、利用率和资源总量可能高度冗余，应按任务数/资源量/利用率/功率分组消融，而非一次投入所有强相关列。",
        f"共发现 {len(tables['redundant_pairs'])} 对绝对 Spearman ≥ 0.95 的特征；完整列表见 redundant_pairs.csv。相同指标不同统计量的高相关不代表独立信息增益。",
        "## 4. 限制与下一步",
        "这些是全月描述性关联，不是因果结论，不是显著性检验或预测增益验证。时间自相关、共同趋势、跨特征多重比较均可能抬高表观关系；没有报告独立同分布 p 值。主目标排除了全空补值，但电力原表仍可能有部分设备/相位缺失；该范围不能冒充完整物理计量。不同任务的 GPU 计量范围未确认，不据此反推机房能耗占比。",
        "后续在每折严格 14 天训练历史内选择特征，显式定义 observed_past provider 和采集延迟；分别验证负荷基线、加算力任务数、加资源量/利用率、加 GPU 功率、去冗余组合。用连续留出日期及正式每日 288 点回测比较误差。本脚本不训练或改写预测配置。",
        "## 5. 完整产物",
        "feature_summary.csv：全部特征、全样本/非补值/非零/差分/日内去均值相关及样本状态；lag_correlations.csv：全部滞后及其差分关系；daily_correlations.csv：同期和 288/576 步逐日关系；ranking_contemporaneous.csv / ranking_day_lag.csv：完整排名；redundant_pairs.csv：冗余对；analysis_audit.json：输入哈希、样本数、口径和各表行数。空相关写空单元格，status 解释 constant/insufficient，不解释为零相关。短 lag 未计算分日稳定性，对应 valid_days 为空而非零天。",
    ]) + "\n"


def run(input_csv: Path, raw_power_csv: Path, output_dir: Path, overwrite: bool = False) -> dict:
    table_names = ("feature_summary", "lag_correlations", "daily_correlations", "redundant_pairs",
                   "ranking_contemporaneous", "ranking_day_lag")
    destinations = [output_dir / f"{name}.csv" for name in table_names] + [output_dir / "report.md", output_dir / "analysis_audit.json"]
    if {p.resolve() for p in destinations} & {input_csv.resolve(), raw_power_csv.resolve()}:
        raise ValueError("输出不得覆盖输入")
    if not overwrite and any(p.exists() for p in destinations):
        raise FileExistsError("分析产物已存在；重建请指定 --overwrite")
    inputs = {str(p.resolve()): sha256(p) for p in (input_csv, raw_power_csv)}
    frame, observed = load_inputs(input_csv, raw_power_csv)
    tables = analyze(frame, observed)
    audit = {"version": "liantong_computility_analysis_v1", "inputs_sha256": inputs,
             "builder_sha256": sha256(Path(__file__)), "rows": len(frame), "features": len(feature_names()),
             "start": str(frame.time.iloc[0]), "end": str(frame.time.iloc[-1]),
             "excluded_target_rows": int((~observed).sum()),
             "excluded_dates": sorted(frame.loc[~observed, "time"].dt.strftime("%Y-%m-%d").unique().tolist()),
             "lag_definition": "corr(X[t-lag], y[t]); full grid shifted before masking",
             "lags_steps": LAGS, "min_pairs": 3, "daily_min_pairs": 24,
             "ranking": "absolute Spearman; descriptive only; not feature selection for backtest",
             "forecast_improvement_verified": False, "model_configs_modified": False,
             "outputs_rows": {name: len(table) for name, table in tables.items()}}
    if inputs != {str(p.resolve()): sha256(p) for p in (input_csv, raw_power_csv)}:
        raise ValueError("分析期间输入变化，拒绝发布")
    report = build_report(tables, audit)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    audit["outputs_sha256"] = {p.name: sha256(p) for p in destinations[:-1]}
    (output_dir / "analysis_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DATA / "power_computility_5min_20260801_20260831.csv")
    parser.add_argument("--raw-power-csv", type=Path, default=DATA / "df_power.csv")
    parser.add_argument("--output-dir", type=Path, default=DATA / "aidc_comp_liantong_5min/feature_analysis")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    audit = run(args.input_csv, args.raw_power_csv, args.output_dir, args.overwrite)
    print(json.dumps({"output_dir": str(args.output_dir), "rows": audit["rows"], "features": audit["features"],
                      "excluded_target_rows": audit["excluded_target_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
