"""正式水温流量产物：逐缺口、逐特征及双版本资产证据。"""
import json
import unittest

from config.aidc_hvac_load_5min.scripts.v3.forecast_data.prepare_flow_temp import DEFAULT_ROOT, RECIPE
from config.aidc_hvac_load_5min.scripts.v3.forecast_data.verify_flow_temp import verify_assets


class FlowTempAssetsTest(unittest.TestCase):
    def test_real_assets_preserve_sources_and_reconstruct_every_gap(self):
        recipe = json.loads(RECIPE.read_text(encoding='utf-8'))
        result = verify_assets(DEFAULT_ROOT, recipe)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['rows'], 46656)
        self.assertEqual(result['selected_points'], 58)
        self.assertEqual(result['feature_count'], 76)
        self.assertEqual(result['filled_cells'], 167351)


if __name__ == '__main__':
    unittest.main()
