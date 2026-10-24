"""Install the long-task desktop entry alongside the existing GARMI tasks."""
import argparse
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--lab',type=Path,required=True);p.add_argument('--home',type=Path,required=True);a=p.parse_args()
    text='[Desktop Entry]\nType=Application\nName=GARMI Household Task\nComment=Bowl to table, mug to drawer\nTerminal=false\nExec=/opt/conda/bin/python "'+str(ROOT/'demos/household_sequence/start_service.py')+'"\n'
    for folder in (a.lab/'demos/desktop',a.home/'Desktop'):
        folder.mkdir(parents=True,exist_ok=True);path=folder/'GARMI_Household.desktop';path.write_text(text);path.chmod(0o755)
    config=a.home/'.jupyter/jupyter_server_config.py'
    config.parent.mkdir(parents=True,exist_ok=True)
    content=config.read_text() if config.exists() else ''
    marker='# GARMI packaged Workspace handler'
    if marker not in content:
        content += '\n'+marker+'\nimport sys\n'+f'sys.path.insert(0, {str(ROOT / "demos/household_sequence")!r})\n'+ 'c.ContentsManager.files_handler_class = "workspace_handler.WorkspaceFilesHandler"\n'
        config.write_text(content)
    print('HOUSEHOLD_DESKTOP_INSTALLED')
if __name__=='__main__':main()
