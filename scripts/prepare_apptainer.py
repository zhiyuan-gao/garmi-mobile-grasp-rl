"""Create bind destinations required by --nv in a writable, user-owned sandbox."""
from pathlib import Path
import shutil
import sys


def prepare(root):
    root = Path(root).resolve(strict=True)
    (root / 'workspace').mkdir(exist_ok=True)
    (root / 'home/rl').mkdir(parents=True, exist_ok=True)
    paths = [Path('/etc/localtime')]
    for line in Path('/etc/apptainer/nvliblist.conf').read_text().splitlines():
        entry = line.split('#', 1)[0].strip()
        if not entry or entry.endswith('.so'):
            continue
        source = entry if entry.startswith('/') else shutil.which(entry)
        if source and Path(source).is_file():
            paths.append(Path(source))
    for source in paths:
        target = root / source.relative_to('/')
        if not target.exists() and not target.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()


if __name__ == '__main__':
    prepare(sys.argv[1])
