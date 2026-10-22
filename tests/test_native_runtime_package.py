import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('native_bundle',
    Path(__file__).resolve().parents[1] / 'scripts/build_native_runtime.py')
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class NativePackageTests(unittest.TestCase):
    def test_all_native_sources_match_the_original_manifest(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(len(bundle.verify(root)['files']), 146)

    def test_corrupt_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.py'
            source.write_text('original')
            m = root / bundle.MAP
            m.parent.mkdir(parents=True)
            m.write_text(json.dumps({'files': {'runtime.py': {
                'source': 'source.py', 'sha256': bundle.digest(source), 'mode': 420}}}))
            bundle.verify(root)
            source.write_text('modified')
            with self.assertRaises(ValueError):
                bundle.verify(root)

    def test_paths_must_stay_inside_the_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('../escape', '/absolute', '.'):
                with self.assertRaises(ValueError):
                    bundle.safe_path(Path(tmp), name)


if __name__ == '__main__':
    unittest.main()
