"""Check equal-first-contact centering, and prepare collision-free pitcher variants."""
import copy
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from trimesh.collision import CollisionManager
from try_handle_roll import OUT, read, save, host, prepare_mask
from select_references import ROOT, HAND_OFFSET, sha
from prepare_rl_validation import hand_parts
from rebalance_mug import FingerContacts

for name in ['mug','pitcher']:
    source=OUT/name;bank=read(source/'bank.json');obj=bank['objects'][0]
    mesh=trimesh.load(host(obj['mesh']),force='mesh',process=False)
    boundary=trimesh.load(source/'cooked-boundary.obj',force='mesh',process=False)
    scan,cooked=FingerContacts(mesh),FingerContacts(boundary)
    cm=CollisionManager();cm.add_object('object',trimesh.util.concatenate([mesh,boundary]))
    hand=trimesh.util.concatenate(hand_parts());hand.apply_transform(HAND_OFFSET)
    records=[];refs=[]
    for ref in obj['references']:
        t=np.array(ref['mesh_T_gripper']);a,b=cooked.query(t);shift=(a['q_m']-b['q_m'])/2
        t[:3,3]+=shift*t[:3,0]
        v=trimesh.transform_points(hand.vertices,t)
        clearance=min(v[:,2].min(),v[:,2].min()-.08*t[2,2])-mesh.bounds[0,2]
        hits=[]
        for depth in np.linspace(0,.08,17):
            h=t.copy();h[:3,3]-=depth*h[:3,2]
            if cm.in_collision_single(hand,transform=h):hits.append(float(depth))
        c0,c1=scan.query(t),cooked.query(t)
        compatible=all(a is not None and b is not None and abs(a['q_m']-b['q_m'])<=.002 for a,b in zip(c0,c1))
        row=dict(reference=ref['id'],additional_center_shift_m=float(shift),table_clearance_m=float(clearance),
                 approach_collision_depths_m=hits,contacts=c1,scan_contacts=c0,mesh_T_gripper=t.tolist(),
                 contact_surfaces_agree=compatible)
        records.append(row)
        if clearance<.002 or hits or not compatible:continue
        if name=='pitcher':
            d=np.array([.071,.069,0]);d/=np.linalg.norm(d);split=np.array([-.015,.03,0])+d*.068
            if not all((np.array(c['point_m'])-split)@d>0 for c in c0+c1):continue
        else:
            if not all(c['point_m'][0]>.021 for c in c0+c1):continue
        r=copy.deepcopy(ref);h=t@HAND_OFFSET
        r.update(id=r['id'].replace('roll90','roll90-centered'),mesh_T_gripper=t.tolist(),position=h[:3,3].tolist(),
            quaternion=Rotation.from_matrix(h[:3,:3]).as_quat()[[3,0,1,2]].tolist(),
            contacts=dict(scan=c0,cooked=c1),additional_center_shift_m=float(shift))
        r['translation_in_rolled_graspgen_frame_m'][0]+=float(shift);refs.append(r)
    save(source/'recentering-check.json',dict(scope='Equal-first-contact center adjustment; static geometry only.',results=records))
    if not refs:continue
    out=OUT/(name+'-centered');out.mkdir(exist_ok=True)
    obj['references']=refs;save(out/'bank.json',bank);save(out/'references.json',refs)
    boundary.export(out/'cooked-boundary.obj')
    prepare_mask(ROOT/f'resources/validated-ycb-v{4 if name=="mug" else 5}/{name}',out,bank,mesh,hand_parts())
    save(out/'provenance.json',dict(parent_report=str((source/'recentering-check.json').relative_to(ROOT)),
        scope='Equal-first-contact centering; same local Z rotation; no physical success inherited.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source/'bank.json',source/'recentering-check.json',ROOT/'graspgen/center_handle_roll.py')}))
