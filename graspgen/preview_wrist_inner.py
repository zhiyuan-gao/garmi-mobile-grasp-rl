"""Measured inner/outer handle holds, displayed as ordinary static images."""
import argparse
import numpy as np,trimesh,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_wrist_v6 import OUT
from prepare_rl_validation import hand_parts,save
from preview_strict_successes import pose,add_mesh
from select_references import ROOT,sha
from try_handle_roll import read
from review_wrist_v6 import matching_condition


def main(name):
    status=read(OUT/'status.json')['objects'][name]
    ids=[r for r in status['accepted_reference_ids'] if 'innerouter' in r]
    assert ids
    reference=ids[0];endpoints=status['endpoint_evidence'][reference]
    evidence=next(a for a in status['approach_evidence'][reference] if any(matching_condition(e['condition'],a['result']) for e in endpoints))
    obj=read(ROOT/endpoints[0]['source']/'bank.json')['objects'][0]
    ref=next(r for r in obj['references'] if r['id']==reference);row=evidence['result']
    trace=ROOT/row['file'].removeprefix('/workspace/');assert sha(trace)==row['sha256']
    with np.load(trace,allow_pickle=False) as d:x=d['observations'][-1].astype(float)
    hand=pose(ref['position'],ref['quaternion'])@np.linalg.inv(pose(x[30:33],x[33:37]));qs=x[7:9]+.04
    assert np.max(np.abs(np.linalg.inv(hand)[:3,3]-x[27:30]))<5e-6
    parts=hand_parts()
    for i,part in enumerate(parts):
        if 2<=i<=5:part.apply_translation([0,qs[0]-.04,0])
        if 6<=i<=9:part.apply_translation([0,-(qs[1]-.04),0])
        part.apply_transform(hand)
    mesh=trimesh.load(ROOT/obj['mesh'].removeprefix('/workspace/'),force='mesh',process=False)
    fig=plt.figure(figsize=(14,6))
    for i,(elevation,delta) in enumerate([(8,0),(25,-35)]):
        ax=fig.add_subplot(1,2,i+1,projection='3d');add_mesh(ax,mesh,'#289e98')
        for k,part in enumerate(parts):add_mesh(ax,part,'#9aa5b1' if k<2 else '#efa91a' if k<6 else '#a855d1',.28 if k<2 else 1.)
        allp=np.vstack([mesh.vertices,*[p.vertices for p in parts]]);center=(allp.max(0)+allp.min(0))/2;radius=np.ptp(allp,axis=0).max()*.51
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(elev=elevation,azim=(134 if name=='pitcher' else 90)+delta)
    fig.suptitle(f'{name.upper()} | measured successful inner/outer handle hold\nPurple: inner finger | Orange: outer finger | Gray: wrist/palm',fontsize=13)
    fig.text(.5,.02,f'{reference} | lift {row["height"]*100:.1f} cm | slip {row["slip"]*1000:.1f} mm | hold {row["hold_seconds"]:.2f} s',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.05,1,.89]);path=OUT/name/'verified-inner-outer.png';fig.savefig(path,dpi=160);plt.close(fig)
    save(path.with_suffix('.json'),dict(reference=reference,trace=str(trace.relative_to(ROOT)),trace_sha256=row['sha256'],object_T_hand=hand.tolist(),finger_joints=qs.tolist()))
    print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('object',choices=['mug','pitcher']);a=p.parse_args();main(a.object)
