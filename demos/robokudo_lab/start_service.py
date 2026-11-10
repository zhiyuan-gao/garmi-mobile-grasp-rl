"""Start only the isolated perception Lab, after its Desktop becomes available."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
try:
    with urllib.request.urlopen('http://127.0.0.1:8782/api/status',timeout=2) as r:
        if json.load(r).get('task')=='robokudo':raise SystemExit(0)
    raise RuntimeError('Another task service owns port 8782')
except OSError:pass
out=ROOT/'results/cram-rl-web';out.mkdir(parents=True,exist_ok=True)
with (out/'preview-service.log').open('ab') as log:
    subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('server.py'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=ROOT)
