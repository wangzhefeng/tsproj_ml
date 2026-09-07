"""天气配置合同；全部资产路径是明确的合成 fixture，不代表真实天气。"""
import copy
import unittest

from forecasting_core.specs.config import parse_data_spec


def weather_options():
    return {
        "inputs": [{"manifest": "fixture/manifest.json", "sha256": "a" * 64}],
        "location_map": [{"series_id": [], "location_id": "fixture-site"}],
        "variables": [{"name": "temperature", "input": "temperature_2m", "unit": "degC", "aggregation": "point"}],
        "native_features": [],
        "temporal": {"timezone": "Asia/Shanghai", "freq": "15min", "label": "left", "closed": "left", "upsampling": "hold", "max_age": "1h"},
        "scenario": "forecast", "vintage_policy": "latest_complete_snapshot", "proxy": None,
        "semantics_version": "weather_v1",
    }


def weather_data():
    return {"sources": [
        {"name": "target", "source_type": "file", "columns": [{"name": "load", "role": "target"}], "history_path": "fixture/load.csv", "time_col": "time", "availability": "source_time"},
        {"name": "weather", "source_type": "generated", "generator": "weather", "time_col": "time", "columns": [{"name": "temperature", "role": "known_future"}], "availability": "generator_defined", "generator_options": weather_options()},
    ]}


class WeatherSpecTest(unittest.TestCase):
    def test_weather_requires_explicit_time_column(self):
        payload = weather_data()
        payload['sources'][1].pop('time_col')
        with self.assertRaises(ValueError):
            parse_data_spec(payload, 'fixture')

    def test_output_projection_and_location_key_arity_are_exact(self):
        for mutation in ("projection", "mapping", "categorical", "ignored"):
            payload = weather_data()
            source = payload["sources"][1]
            if mutation == "projection":
                source["columns"][0]["name"] = "other"
            elif mutation == "mapping":
                source["generator_options"]["location_map"][0]["series_id"] = ["undeclared"]
            elif mutation == "categorical":
                source["columns"][0]["categorical"] = True
            else:
                source["columns"][0]["role"] = "ignored"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                parse_data_spec(payload, "fixture")

    def test_unknown_missing_invalid_nested_fields_raise(self):
        mutations = [
            lambda o: o.update(unknown=True),
            lambda o: o.update(inputs=[]),
            lambda o: o["inputs"][0].update(sha256="xyz"),
            lambda o: o["variables"][0].pop("unit"),
            lambda o: o["variables"][0].update(unit="unknown"),
            lambda o: o["temporal"].update(timezone="Missing/Zone"),
            lambda o: o["temporal"].update(freq="D"),
            lambda o: o["temporal"].update(max_age=3),
            lambda o: o.update(scenario="fallback"),
            lambda o: o.update(location_map=[]),
            lambda o: o.update(semantics_version="future"),
            lambda o: o.update(proxy={"years": 1}),
        ]
        for i, mutate in enumerate(mutations):
            payload = weather_data()
            mutate(payload["sources"][1]["generator_options"])
            with self.subTest(case=i), self.assertRaises((ValueError, TypeError)):
                parse_data_spec(payload, "fixture")

    def test_nonweather_forbids_options_and_weather_requires_them(self):
        for mutation in ("file", "calendar", "missing", "availability"):
            payload = weather_data()
            source = payload["sources"][1]
            if mutation == "file":
                source.update(source_type="file", generator=None, future_path="fixture/future.csv", time_col="time", availability="forecast_origin")
            elif mutation == "calendar":
                source["generator"] = "chinese_holiday"
            elif mutation == "missing":
                source.pop("generator_options")
            else:
                source.update(availability="forecast_origin", time_col="time")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                parse_data_spec(payload, "fixture")

    def test_proxy_requires_explicit_evidence_kind_and_leap_policy(self):
        payload = weather_data()
        options = payload["sources"][1]["generator_options"]
        options.update(scenario="prior_year_proxy", vintage_policy=None, proxy={"years": 1, "leap_day": "reject", "data_kind": "reanalysis"})
        data = parse_data_spec(payload, "fixture")
        self.assertEqual(data.sources[1].generator_options.proxy.data_kind, "reanalysis")
        options["proxy"]["years"] = True
        with self.assertRaises(ValueError):
            parse_data_spec(payload, "fixture")

    def test_options_are_immutable_and_semantic_payload_changes(self):
        payload = weather_data()
        data = parse_data_spec(payload, "fixture")
        before = data.canonical_payload()
        payload["sources"][1]["generator_options"]["inputs"][0]["sha256"] = "b" * 64
        self.assertEqual(data.canonical_payload(), before)
        self.assertNotEqual(parse_data_spec(payload, "fixture").canonical_payload(), before)

    def test_nonweather_payload_has_no_new_null_field(self):
        payload = weather_data()
        payload["sources"] = payload["sources"][:1]
        data = parse_data_spec(payload, "fixture")
        self.assertNotIn("generator_options", data.canonical_payload()["sources"][0])

    def test_weather_options_parse_and_roundtrip_without_file_io(self):
        data = parse_data_spec(weather_data(), "fixture")
        options = data.sources[1].generator_options
        self.assertEqual(type(options).__name__, "WeatherGenerationSpec")
        self.assertEqual(options.temporal.freq, "15min")
        self.assertEqual(options.canonical_payload(), weather_options())
        self.assertEqual(parse_data_spec(data.canonical_payload(), "roundtrip"), data)


if __name__ == "__main__":
    unittest.main()
