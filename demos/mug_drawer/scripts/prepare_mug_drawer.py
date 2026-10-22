"""Build isolated dual-arm and passive-drawer assets from existing local assets."""
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
from pathlib import Path
import copy, hashlib, json, sys
import xml.etree.ElementTree as ET
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf, Gf, Tf
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
root=Path('/demo/assets/mug-drawer'); root.mkdir(parents=True,exist_ok=True)
original=Path('/home/jovyan/cram-vrb-lab')
sys.path.insert(0,str(original))
from cram_vrb_lab.robots.garmi.joints import load_patched_urdf
source=ET.fromstring(load_patched_urdf())
tree=ET.parse('/workspace/rl/assets/garmi_training.urdf'); robot=tree.getroot()
for name in [f'left_fr3_joint{i}' for i in range(1,8)]+['left_fr3_finger_joint1','left_fr3_finger_joint2']:
 old=robot.find(f"joint[@name='{name}']"); new=copy.deepcopy(source.find(f"joint[@name='{name}']"))
 mimic=new.find('mimic')
 if mimic is not None:new.remove(mimic)
 robot.remove(old);robot.append(new)
for m in robot.findall('.//mesh'):
 path=m.get('filename').replace('/workspace/cram-vrb-lab',str(original))
 if path.startswith('/tmp/garmi_mesh_aliases/'):
  filename=Path(path).name
  matches=list((original/'assets/garmi_description').rglob(filename.replace('_','-')))
  if not matches: raise FileNotFoundError(path)
  path=str(matches[0])
 if not Path(path).exists():raise FileNotFoundError(path)
 # URDF importer derives USD prim names from mesh filenames. Hyphens in
 # original names such as end-cover.stl produce invalid USD paths. Keep a
 # stable alias inside the generated assets rather than relying on /tmp.
 mesh=Path(path)
 if Tf.MakeValidIdentifier(mesh.stem)!=mesh.stem:
  aliases=root/'mesh_aliases';aliases.mkdir(exist_ok=True)
  alias=aliases/(Tf.MakeValidIdentifier(mesh.stem)+'_'+hashlib.sha256(str(mesh).encode()).hexdigest()[:8]+mesh.suffix)
  if not alias.is_symlink():alias.symlink_to(mesh)
  path=str(alias)
 m.set('filename',path)
tree.write(root/'garmi_dual.urdf')
converter=UrdfConverter(UrdfConverterCfg(asset_path=str(root/'garmi_dual.urdf'),usd_dir=str(root),usd_file_name='garmi_dual.usd',fix_base=False,merge_fixed_joints=True,force_usd_conversion=True,make_instanceable=False,self_collision=True,joint_drive=UrdfConverterCfg.JointDriveCfg(gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=10000.,damping=200.))))
# Preserve the original training asset's collision approximations and structural filters.
oldstage=Usd.Stage.Open('/workspace/rl/assets/garmi.usd'); newstage=Usd.Stage.Open(converter.usd_path)
if not newstage.GetDefaultPrim() or not newstage.GetPrimAtPath('/garmi/joints'):
 raise RuntimeError('URDF conversion did not produce the expected GARMI articulation')
oldroot=oldstage.GetDefaultPrim().GetPath();newroot=newstage.GetDefaultPrim().GetPath()
for prim in oldstage.Traverse():
 target=newstage.GetPrimAtPath(prim.GetPath().ReplacePrefix(oldroot,newroot))
 if not target:continue
 for attr in prim.GetAttributes():
  if attr.GetName() in ['physics:approximation','physxCollision:contactOffset','physxCollision:restOffset']:
   target.CreateAttribute(attr.GetName(),attr.GetTypeName()).Set(attr.Get())
 for rel in prim.GetRelationships():
  if rel.GetName()=='physics:filteredPairs':
   target.CreateRelationship(rel.GetName()).SetTargets([p.ReplacePrefix(oldroot,newroot) for p in rel.GetTargets()])
newstage.GetRootLayer().Save()
# Keep the full trained right-arm subtrees, including materials and collision
# overrides inside referenced geometry, rather than reauthoring their physics.
for stage in [oldstage,newstage]:
 while True:
  instances=[p for p in stage.Traverse() if p.IsInstance()]
  if not instances:break
  for prim in instances:prim.SetInstanceable(False)
preserved=root/'garmi_dual_preserved.usda'
newstage.Flatten().Export(str(preserved)); newstage=Usd.Stage.Open(str(preserved))
flat=oldstage.Flatten()
for prim in oldstage.GetDefaultPrim().GetChildren():
 if prim.GetName().startswith('right_fr3') or prim.GetName()=='Looks':
  Sdf.CopySpec(flat,prim.GetPath(),newstage.GetRootLayer(),prim.GetPath())
for joint in oldstage.GetPrimAtPath('/garmi/joints').GetChildren():
 Sdf.CopySpec(flat,joint.GetPath(),newstage.GetRootLayer(),joint.GetPath())
for prim in newstage.Traverse():
 rel=prim.GetRelationship('physics:filteredPairs')
 if rel:rel.SetTargets([p for p in rel.GetTargets() if newstage.GetPrimAtPath(p)])
UsdPhysics.FilteredPairsAPI.Apply(newstage.GetPrimAtPath('/garmi/left_fr3_link1')).CreateFilteredPairsRel().AddTarget('/garmi/right_fr3_link0')
newstage.GetRootLayer().Save()
# Split the authored upper drawer out of the frozen kitchen. Its original mesh
# and collision pieces are kept; the slide is passive and moved by hand contact.
scene=json.loads(Path('/workspace/assets/kitchen/scene.json').read_text())
kitchen=Usd.Stage.Open('/workspace/assets/kitchen/kitchen.usda')
kitchen.Flatten().Export(str(root/'kitchen.usda'))
kitchen=Usd.Stage.Open(str(root/'kitchen.usda'))
drawer=Usd.Stage.CreateNew(str(root/'drawer.usda')); dp=UsdGeom.Xform.Define(drawer,'/Drawer').GetPrim();drawer.SetDefaultPrim(dp)
UsdGeom.SetStageUpAxis(drawer,'Z');UsdGeom.SetStageMetersPerUnit(drawer,1.)
UsdPhysics.RigidBodyAPI.Apply(dp);UsdPhysics.MassAPI.Apply(dp).CreateMassAttr(2.)
origin=np.array([1.053045,.39,.8])
flat=kitchen.Flatten()
items=[x for x in scene['colliders'] if '/drawer_1/' in x['path']]
cache=UsdGeom.XformCache()
for i,item in enumerate(items):
 src=kitchen.GetPrimAtPath(item['output_path']); dest=f'/Drawer/piece_{i:02}'
 Sdf.CopySpec(flat,src.GetPath(),drawer.GetRootLayer(),dest)
 copied=drawer.GetPrimAtPath(dest); mat=np.array(cache.GetLocalToWorldTransform(src)); mat[3,:3]-=origin
 xf=UsdGeom.Xformable(copied);xf.ClearXformOpOrder();xf.AddTransformOp().Set(Gf.Matrix4d(mat.tolist()))
 UsdGeom.Imageable(copied).CreateVisibilityAttr('inherited');UsdGeom.Imageable(copied).CreatePurposeAttr('default')
 kitchen.RemovePrim(src.GetPath())
# Remap physics material separately.
Sdf.CopySpec(flat,'/Kitchen/Material',drawer.GetRootLayer(),'/Drawer/Material')
for prim in drawer.Traverse():
 for rel in prim.GetRelationships():
  if rel.GetName().startswith('material:binding'):rel.SetTargets(['/Drawer/Material'])
# World prismatic rail: translation only along task X; negative means open.
j=UsdPhysics.PrismaticJoint.Define(drawer,'/Drawer/rail');j.CreateBody1Rel().SetTargets(['/Drawer'])
j.CreateAxisAttr('X');j.CreateLowerLimitAttr(-.466);j.CreateUpperLimitAttr(0.)
j.CreateLocalPos0Attr(Gf.Vec3f(*origin));j.CreateLocalPos1Attr(Gf.Vec3f(0,0,0))
j.CreateExcludeFromArticulationAttr(True)
PhysxSchema.PhysxJointAPI.Apply(j.GetPrim()).CreateJointFrictionAttr(.1)
kitchen.GetRootLayer().Save();drawer.GetRootLayer().Save()
(root/'manifest.json').write_text(json.dumps(dict(drawer_origin=origin.tolist(),pieces=len(items),passive_slide=True),indent=2))
print('ASSETS_READY',len(items),flush=True)
app.close()
