"""Reuse NeemHub's scene/Plan/Knowledge UI for the isolated perception task."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('household_neemhub',ROOT/'demos/household_sequence/neemhub.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original_index=m.task_index

def index():
    value=original_index()
    for scene in value['scenes']:
        if scene['name']=='__live__':scene['task']='RoboKudo household task'
    return value
m.task_index=index
OriginalHandler=m.Handler
class Handler(OriginalHandler):
    def do_GET(self):
        if self.path.split('?')[0]=='/household-task.js':
            text=(m.ROOT/'household-task.js').read_text().replace('Run one bowl-to-table and mug-to-drawer task','Run one perception-guided bowl-to-table and mug-to-drawer task').replace('Connecting to Bowl and Mug','Connecting to RoboKudo household task')
            return self.send_text(text,'application/javascript')
        return super().do_GET()
m.Handler=Handler
if __name__=='__main__':m.main()
