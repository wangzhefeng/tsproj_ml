"""只运行1D的固定月份对照；所有候选为物理YAML，结果逐批落盘。"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from config.config_loader import load_yaml_config
from forecasting_core.specs.config import parse_model_config
from model_evaluation.point import evaluate_point_forecasts
from annual_reporting import annual_tensors
from annual_backtest import run_config, write_json
from cold_start import load_recipe
from generate_configs import model_document, apply_options

SITES = ('guangdianchang', 'xinnengyuan')
MONTHS = (2, 3, 6, 7, 10)
EVIDENCE = ROOT / '.hermes/plans'
EXPERIMENTS = Path(__file__).parent / 'experiments'
SCREEN = {'baseline': {}, 'l2': {'objective': 'regression'}, 'calendar': {'calendar': True},
          'weight30': {'halflife_days': 30}, 'weight60': {'halflife_days': 60}}


def metrics(frame: pd.DataFrame) -> dict:
    scores = evaluate_point_forecasts(*annual_tensors(frame))
    row = scores.loc[scores.scope == 'aggregate'].iloc[0]
    return {**{name: float(row[name]) for name in ('MAE', 'RMSE', 'Bias', 'MAPE')}, 'n_points': int(row.n_points)}


def reference(site: str) -> pd.DataFrame:
    manifest = json.loads((EVIDENCE / 'hongtaiyang-optimization-baseline.json').read_text())
    cfg = parse_model_config(model_document(site, 'demand_load', True, 'direct-pointwise'), source='reference')
    candidates = []
    for item in manifest:
        out = Path(item['path'])
        if json.loads((ROOT / out / 'audit.json').read_text())['config'] == cfg.fingerprint():
            candidates.append(ROOT / out / 'prediction.csv')
    if len(candidates) != 1:
        raise ValueError('expected one frozen pointwise reference per site')
    return pd.read_csv(candidates[0], parse_dates=['time'])


def run_candidate(name: str, options: dict, recipe: dict, months=MONTHS) -> dict:
    recipe_path = EXPERIMENTS / name / 'annual_recipe.json'
    documents = []
    for site in SITES:
        document = apply_options(model_document(site, 'demand_load', True, 'direct-pointwise'), options[site])
        document['output']['scenario_subpath'] = f'hongtaiyang_cesuan/experiments/{name}/{site}/freq_1day'
        path = EXPERIMENTS / name / site / 'lgbm_direct-pointwise.yaml'
        parse_model_config(document, source=path)
        documents.append((site, path, document))
    recipe_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(recipe_path, recipe)
    for _, path, document in documents:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    result = {'name': name, 'options': options, 'recipe': recipe, 'scores': {}, 'runs': {}}
    for site, path, _ in documents:
        status = run_config(path.resolve(), recipe_path=recipe_path, months=tuple(months))
        output = Path(status['output'])
        frames = [pd.read_csv(output / 'windows' / f'2025{m:02d}01.csv', parse_dates=['time']) for m in months]
        prediction = pd.concat(frames, ignore_index=True)
        actual = pd.read_csv(ROOT / load_yaml_config(path).data.sources[0].history_path, parse_dates=['time'])
        joined = prediction.merge(actual.rename(columns={'value': 'y_true'}), on='time', validate='one_to_one')
        if len(joined) != sum(pd.Timestamp(2025, m, 1).days_in_month for m in months):
            raise ValueError('selected diagnostic coverage is incomplete')
        if name == 'baseline':
            old = reference(site)
            warm = joined[joined.time.dt.month != 2].merge(old[['time', 'y_pred']], on='time', suffixes=('', '_old'))
            np.testing.assert_allclose(warm.y_pred, warm.y_pred_old, rtol=1e-12, atol=1e-10)
        result['scores'][site] = {}
        for label, subset in (('cold', joined[joined.time.dt.month == 2]), ('normal', joined[joined.time.dt.month != 2])):
            if not subset.empty:
                result['scores'][site][label] = metrics(subset)
        result['runs'][site] = str(output)
        print(name, site, result['scores'][site], flush=True)
    return result


def passed(candidate: dict, baseline: dict) -> bool:
    return candidate['MAE'] <= baseline['MAE'] * .98 and candidate['RMSE'] <= baseline['RMSE'] * 1.02


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('cold', 'screen', 'combined'), required=True)
    args = parser.parse_args()
    results = []
    output = EVIDENCE / f'hongtaiyang-opt-{args.stage}.json'
    if args.stage == 'cold':
        original = {site: metrics(reference(site).loc[lambda f: f.time.dt.month == 2]) for site in SITES}
        for method, transition in (('calendar_pointwise', 0), ('calendar_baseline', 0), ('calendar_baseline', 3)):
            recipe = {'recipe_version': 1, 'cold_start': {'method': method, 'normal_floor_ratio': .5, 'transition_days': transition}}
            result = run_candidate(f'cold_{method}_{transition}', {site: {'calendar': True} for site in SITES}, recipe, (2,))
            results.append(result)
            write_json(output, {'reference': original, 'candidates': results})
        eligible = [r for r in results if all(passed(r['scores'][site]['cold'], original[site]) for site in SITES)]
        if not eligible:
            raise ValueError('no nonrecursive cold-start candidate passed both sites')
        selected = min(eligible, key=lambda r: sum(r['scores'][s]['cold']['MAE'] / original[s]['MAE'] for s in SITES))
        write_json(output, {'reference': original, 'candidates': results, 'selected': selected['name'], 'selected_recipe': selected['recipe']})
    elif args.stage == 'screen':
        cold = json.loads((EVIDENCE / 'hongtaiyang-opt-cold.json').read_text())
        for name, options in SCREEN.items():
            results.append(run_candidate(name, {site: options for site in SITES}, cold['selected_recipe']))
            write_json(output, {'candidates': results})
        baseline = results[0]
        options = {site: {} for site in SITES}
        accepted = {}
        for site in SITES:
            good = [r for r in results[1:] if passed(r['scores'][site]['normal'], baseline['scores'][site]['normal'])]
            accepted[site] = [r['name'] for r in good]
            for group in (('l2',), ('calendar',), ('weight30', 'weight60')):
                subset = [r for r in good if r['name'] in group]
                if subset:
                    winner = min(subset, key=lambda r: r['scores'][site]['normal']['MAE'])
                    options[site].update(winner['options'][site])
        write_json(output, {'candidates': results, 'accepted_factors': accepted, 'combined_options': options,
                            'criterion': 'per-site normal-month MAE <= .98 baseline; RMSE <= 1.02 baseline'})
    else:
        cold = json.loads((EVIDENCE / 'hongtaiyang-opt-cold.json').read_text())
        screen = json.loads((EVIDENCE / 'hongtaiyang-opt-screen.json').read_text())
        result = run_candidate('combined', screen['combined_options'], cold['selected_recipe'])
        baseline = screen['candidates'][0]
        selected = {}
        for site in SITES:
            candidates = [r for r in [*screen['candidates'][1:], result]
                          if passed(r['scores'][site]['normal'], baseline['scores'][site]['normal'])]
            best = min(candidates, key=lambda r: r['scores'][site]['normal']['MAE']) if candidates else baseline
            selected[site] = {'name': best['name'], 'options': best['options'][site], 'scores': best['scores'][site]}
        write_json(output, {'combined': result, 'selected_by_site': selected, 'recipe': cold['selected_recipe']})
    print(output)


if __name__ == '__main__':
    main()
