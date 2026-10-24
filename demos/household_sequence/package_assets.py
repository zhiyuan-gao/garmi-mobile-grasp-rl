"""Package the validated household scene and mug policy for native deployment."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lab-workspace',type=Path,required=True)
    p.add_argument('--bowl-workspace',type=Path,required=True)
    p.add_argument('--scene',type=Path,required=True);p.add_argument('--mug-workspace',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    inputs={}
    for name in ('garmi_dual_preserved.usda','garmi_dual.urdf','apartment.usda','drawer_visual.usda','manifest.json','configuration/materials/textures/face.jpg'):
        inputs['household/assets/'+name]=a.scene/name
    # The IK parser only reads joint geometry, but retain the mesh aliases for a portable URDF.
    for path in (a.scene/'mesh_aliases').iterdir():
        inputs['household/assets/mesh_aliases/'+path.name]=a.lab_workspace/str(path.readlink()).removeprefix('/home/jovyan/cram-vrb-lab/')
    for name in ('runs/mug-physical-success-review/ppo.yaml','runs/mug-ppo-physical-success/ppo-6500/checkpoints/update_6500.pt','resources/validated-ycb-v6/mug/bank.json'):
        inputs['household/mug/'+name]=a.mug_workspace/name
    for path in (a.mug_workspace/'assets/ycb/025_mug/surface-v3').rglob('*'):
        if path.is_file() and path.suffix in ('.usda','.usd','.obj','.mtl','.png','.jpg'):
            inputs['household/mug/'+str(path.relative_to(a.mug_workspace))]=path
    inputs['household/assets/configuration/materials/textures/fabric.jpg']=a.bowl_workspace/'rl/assets/configuration/materials/textures/fabric.jpg'
    hashes={};original={}
    with (a.out/'household-assets.tar.gz').open('wb') as stream,gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for name,path in sorted(inputs.items()):
            data=path.read_bytes();original[name]=hashlib.sha256(data).hexdigest()
            if name.startswith('household/assets/') and path.suffix in ('.usda','.urdf'):
                data=data.replace(b'/demo/assets/mug-drawer',b'/workspace/household/assets')
                data=data.replace(b'/workspace/rl/assets/configuration/materials/textures/fabric.jpg',b'/workspace/household/assets/configuration/materials/textures/fabric.jpg')
            # Mug USD textures were authored with absolute source-workspace paths.
            if name.startswith('household/mug/') and path.suffix in ('.usda','.mtl'):
                data=data.replace(b'/workspace/assets/ycb/',b'/workspace/household/mug/assets/ycb/')
            hashes[name]=hashlib.sha256(data).hexdigest()
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
            tar.addfile(info,io.BytesIO(data))
    manifest={'schema_version':1,'files':hashes,'source_files':original,'archive_sha256':hashlib.sha256((a.out/'household-assets.tar.gz').read_bytes()).hexdigest()}
    (a.out/'household-assets.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('HOUSEHOLD_ASSETS_PACKAGED',len(hashes))
if __name__=='__main__':main()
