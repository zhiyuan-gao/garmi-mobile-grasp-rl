"""Regression for contacts that the old SRDF hid from the expert planner."""
from pathlib import Path
import xml.etree.ElementTree as ET
from garmi_rl.collision_pairs import adjacent_body_pairs,planner_body_pairs,planner_collision_settings


def test_planner_exclusions_do_not_hide_observed_training_self_collisions():
    root=Path(__file__).resolve().parents[1]
    urdf=ET.parse(root/'rl/assets/garmi_training.urdf').getroot()
    bodies={'base_link',*[f'right_fr3_link{i}' for i in range(8)],
            'right_fr3_leftfinger','right_fr3_rightfinger'}
    excluded=adjacent_body_pairs(urdf,bodies)
    # Both contacts have nonzero force records in the strict physical verifier.
    assert frozenset(('right_fr3_link5','right_fr3_link7')) not in excluded
    assert frozenset(('right_fr3_leftfinger','right_fr3_rightfinger')) not in excluded
    # The fixed hand link belongs to link7; its finger joints remain adjacent.
    assert frozenset(('right_fr3_link7','right_fr3_leftfinger')) in excluded
    assert frozenset(('right_fr3_link7','right_fr3_rightfinger')) in excluded
    assert frozenset(('right_fr3_link6','right_fr3_link7')) in excluded
    # USD retains link0 as a rigid body despite its fixed connection to the base.
    assert frozenset(('base_link','right_fr3_link0')) in excluded
    assert frozenset(('right_fr3_link0','right_fr3_link1')) in excluded
    assert frozenset(('base_link','right_fr3_link1')) not in excluded


def test_exclusions_match_actual_usd_joint_body_pairs():
    from pxr import Usd,UsdPhysics
    root=Path(__file__).resolve().parents[1]
    stage=Usd.Stage.Open(str(root/'rl/assets/garmi.usd'))
    bodies={p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    actual=set()
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):continue
        joint=UsdPhysics.Joint(prim)
        pair=frozenset(str(t).rsplit('/',1)[-1] for rel in [joint.GetBody0Rel(),joint.GetBody1Rel()] for t in rel.GetTargets())
        if len(pair)==2:actual.add(pair)
    assert adjacent_body_pairs(ET.parse(root/'rl/assets/garmi_training.urdf').getroot(),bodies)==actual


def test_historical_and_new_planner_profiles_are_explicit():
    assert planner_collision_settings({})==dict(self_collision_model='legacy_srdf',self_collision_buffer_m=.03,distal_collision_buffer_m=.05)
    assert planner_collision_settings({'expert_collision':{'self_buffer_m':.005}})==dict(self_collision_model='usd_adjacent',self_collision_buffer_m=.005,distal_collision_buffer_m=.05)
    assert planner_collision_settings({'expert_collision':{'model':'usd_adjacent','self_buffer_m':.005,'distal_buffer_m':.003}})['distal_collision_buffer_m']==.003


def test_legacy_srdf_exclusions_are_kept_only_for_legacy_profiles():
    root=Path(__file__).resolve().parents[1]
    urdf=ET.parse(root/'rl/assets/garmi_training.urdf').getroot();srdf=ET.parse(root/'assets/garmi.srdf').getroot()
    bodies={'base_link',*[f'right_fr3_link{i}' for i in range(8)],'right_fr3_leftfinger','right_fr3_rightfinger'}
    pair=frozenset(('right_fr3_link5','right_fr3_link7'))
    assert pair in planner_body_pairs(urdf,bodies,'legacy_srdf',srdf)
    assert pair not in planner_body_pairs(urdf,bodies,'usd_adjacent',srdf)
