"""CPU checks for checkpoint integrity and verified dependency imports."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("checkpoint_package",
    Path(__file__).resolve().parents[1] / "scripts/package_mug_checkpoint.py")
bundle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bundle)


class CheckpointPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "runtime"
        self.source = Path(self.temp.name) / "source"
        self.root.mkdir()
        self.source.mkdir()
        (self.root / "model.pt").write_bytes(b"checkpoint")
        (self.source / "asset.usda").write_bytes(b"asset")
        entry = lambda p: dict(bytes=p.stat().st_size, sha256=bundle.sha(p))
        manifest = dict(update=6500, files={"model.pt": entry(self.root / "model.pt")},
                        code_sha256={}, dependencies={"asset.usda": dict(entry(self.source / "asset.usda"), role="simulation")})
        path = self.root / bundle.MANIFEST
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(manifest))

    def test_checkpoint_corruption_is_rejected(self):
        bundle.verify(self.root)
        (self.root / "model.pt").write_bytes(b"CHECKPOINT")
        with self.assertRaises(ValueError):
            bundle.verify(self.root)

    def test_verified_import_completes_runtime(self):
        with self.assertRaises(FileNotFoundError):
            bundle.verify(self.root, require_dependencies=True)
        bundle.import_inputs(self.root, self.source)
        bundle.verify(self.root, require_dependencies=True)

    def test_existing_input_is_not_overwritten(self):
        (self.root / "asset.usda").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            bundle.import_inputs(self.root, self.source)
        self.assertEqual((self.root / "asset.usda").read_bytes(), b"changed")

    def test_paths_and_symlinks_cannot_escape_workspace(self):
        for value in ("../outside", "/etc/passwd", "/workspace/../outside"):
            with self.assertRaises(ValueError):
                bundle.relative_path(value)
        (self.root / "link").symlink_to(self.source, target_is_directory=True)
        with self.assertRaises(ValueError):
            bundle.checked_file(self.root, "link/asset.usda")


if __name__ == "__main__":
    unittest.main()
