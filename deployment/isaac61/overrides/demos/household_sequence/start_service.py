"""Start the household Lab once when its remote desktop session becomes available."""
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[2]

def main():
    try:
        with urllib.request.urlopen('http://127.0.0.1:8782/api/status',timeout=2) as response:
            if json.load(response).get('task')=='household':return
    except (OSError,ValueError):pass
    folder=ROOT/'results/household-lab';folder.mkdir(parents=True,exist_ok=True)
    with (folder/'server.log').open('ab') as log:
        subprocess.Popen([sys.executable,'-u',str(Path(__file__).with_name('server.py')),'--autostart'],
            cwd=ROOT,env=dict(os.environ,DISPLAY=os.environ.get('DISPLAY',':1')),
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
if __name__=='__main__':main()
