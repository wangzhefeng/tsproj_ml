"""Registry 绑定的本地天气生成器；缓存只存通过哈希和可得性验证的结果。"""
from collections import OrderedDict
import copy
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd

from data_loading.weather_generator.assets import WeatherAssetStore
from data_loading.weather_generator.pipeline import generate_weather
from forecasting_core.specs.weather import WeatherGenerationSpec


class WeatherGenerator:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.store = WeatherAssetStore(self.base_dir)
        self.cache = OrderedDict()

    def __call__(self, source, request):
        spec = source.generator_options
        if not isinstance(spec, WeatherGenerationSpec):
            raise TypeError('weather requires WeatherGenerationSpec')
        # 每次访问先重读并验证传递文件；不可在结果缓存命中时跳过输入完整性。
        dependencies = tuple(dep for ref in spec.inputs for snapshot in self.store.load(ref) for dep in snapshot.dependency_hashes)
        recipe = json.dumps(spec.canonical_payload(), sort_keys=True)
        key = (recipe, dependencies, request.forecast_origin.isoformat(), tuple(request.forecast_times.asi8), str(request.forecast_times.tz), request.series_ids, source.time_col, source.series_id_cols)
        if key in self.cache:
            self.cache.move_to_end(key)
            return copy.deepcopy(self.cache[key])
        identities = request.series_ids if source.series_id_cols else ((),)
        if not identities:
            raise ValueError('global weather request needs explicit series identities')
        frames, proofs = [], []
        for identity in identities:
            identity = identity if isinstance(identity, tuple) else (identity,)
            if len(identity) != len(source.series_id_cols):
                raise ValueError('weather request identity arity mismatch')
            frame, proof = generate_weather(spec, self.base_dir, request.forecast_origin, request.forecast_times, identity, store=self.store)
            frame = frame.rename(columns={'time': source.time_col})
            for name, value in zip(source.series_id_cols, identity):
                frame[name] = value
            proofs.append({**proof, 'series_id': list(identity)})
            frames.append(frame)
        result = pd.concat(frames, ignore_index=True)
        result.attrs['weather_evidence'] = json.dumps(proofs, sort_keys=True, separators=(',', ':'))
        self.cache[key] = copy.deepcopy(result)
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return result


def weather_generator(source, request):
    """直接调用以 cwd 为根；SourceRegistry 将此内建入口绑定到自己的 base_dir。"""
    return WeatherGenerator(Path.cwd())(source, request)


def weather_implementation_hash():
    """覆盖运行内核与配置语义代码，不仅薄 wrapper。"""
    files = sorted(Path(__file__).parent.glob('*.py')) + [Path(inspect.getfile(WeatherGenerationSpec))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()
