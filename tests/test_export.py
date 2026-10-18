"""CPU checks for relocated inputs and frozen evaluation sampling."""
import hashlib
import json
from pathlib import Path
import unittest

import torch
import yaml
from prepare_40ref_trial import sample_cases
from garmi_rl.reference_position_filter import ReferencePositionFilter

ROOT = Path(__file__).resolve().parents[1]


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank_path = ROOT / 'assets/grasp_bank/selected-40.json'
        if not cls.bank_path.exists():
            raise unittest.SkipTest('Import assets before running export integration checks')
        cls.mask = json.loads((ROOT / 'assets/reference_filter/position-mask.json').read_text())
        cls.cfg = yaml.safe_load((ROOT / 'configs/task.yaml').read_text())
        cls.refs = cls.cfg['garmi_task']['multi_object']['reference_ids']['bowl']

    def test_all_imported_file_hashes_match(self):
        manifest = json.loads((ROOT / 'assets/manifest.json').read_text())
        for relative, expected in manifest['files_sha256'].items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_published_reference_numbers_preserved(self):
        original = json.loads((ROOT / 'resources/bowl-40-references.json').read_text())
        runtime = json.loads(self.bank_path.read_text())['objects'][0]['references']
        self.assertEqual(original, runtime)
        self.assertEqual([r['id'] for r in runtime], self.refs)
        self.assertEqual(len(set(self.refs)), 40)

    def test_evaluation_sampling_is_allowed_balanced_and_disjoint(self):
        forbidden = set()
        cases = sample_cases(self.cfg, self.mask, self.refs, 100, 2026101801, 'eval', forbidden)
        audit = sample_cases(self.cfg, self.mask, self.refs, 10, 2026101802, 'audit', forbidden)
        self.assertEqual(len(forbidden), 4400)
        filt = ReferencePositionFilter(ROOT / 'assets/reference_filter/position-mask.json', self.bank_path, ROOT)
        allowed = filt.allowed([c['bowl_map_xy'] for c in cases + audit], 0.)
        ref_indices = torch.tensor([self.refs.index(c['reference']) for c in cases + audit])
        self.assertTrue(allowed[torch.arange(4400), ref_indices].all())
        for ref in self.refs:
            own = [c for c in cases if c['reference'] == ref]
            self.assertEqual(len(own), 100)
            self.assertEqual(sum(c['region'] == 'right' for c in own), 25)
        self.assertEqual(cases, sample_cases(self.cfg, self.mask, self.refs, 100, 2026101801, 'eval'))

    def test_mask_still_rejects_outside_and_wrong_yaw(self):
        filt = ReferencePositionFilter(ROOT / 'assets/reference_filter/position-mask.json', self.bank_path, ROOT)
        with self.assertRaises(ValueError):
            filt.allowed([[1.169, 7.43]], 0.)
        with self.assertRaises(ValueError):
            filt.allowed([[1.2, 7.43]], 10.)


if __name__ == '__main__':
    unittest.main()
