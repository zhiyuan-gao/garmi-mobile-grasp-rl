"""CPU checks for preserving old tasks and the new live deployment contract."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import hashlib
import io
import tarfile

HERE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('household_unpack',HERE/'unpack_assets.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DeploymentTests(unittest.TestCase):
    def test_asset_integrity_and_safe_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);archive=root/'data.tar.gz';manifest=root/'data.json';dest=root/'runtime';dest.mkdir()
            data=b'policy configuration';name='household/config.json'
            with tarfile.open(archive,'w:gz') as tar:
                info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
            record={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':{name:hashlib.sha256(data).hexdigest()}}
            manifest.write_text(json.dumps(record));module.unpack(dest,archive,manifest)
            self.assertEqual((dest/name).read_bytes(),data)
            record['files'][name]='0'*64;manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'checksum mismatch'):module.unpack(dest,archive,manifest)

if __name__=='__main__':unittest.main()
