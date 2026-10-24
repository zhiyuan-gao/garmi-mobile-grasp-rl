"""Fetch the immutable household asset release and verify before extraction."""
import argparse
import json
from pathlib import Path
import tempfile
import urllib.request
from unpack_assets import unpack
URL='https://github.com/zhiyuan-gao/garmi-mobile-grasp-rl/releases/download/household-assets-20261023/household-assets.tar.gz'

def main():
    root=Path(__file__).resolve().parents[2]
    manifest=Path(__file__).with_name('assets.json')
    with tempfile.TemporaryDirectory(prefix='household-assets-') as temp:
        archive=Path(temp)/'assets.tar.gz'
        urllib.request.urlretrieve(URL,archive)
        unpack(root,archive,manifest)
if __name__=='__main__':main()
