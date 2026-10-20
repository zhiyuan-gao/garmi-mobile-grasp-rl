"""New-object expert initialization only: no planning commands or trajectories."""
import json
from pathlib import Path
import pytest
from garmi_rl.giskard_expert import ApproachExpert


@pytest.mark.parametrize('ycb,name', [('019_pitcher_base','pitcher'),('025_mug','mug')])
def test_expert_uses_each_objects_references_and_physx_hulls(ycb,name):
    root=Path(__file__).resolve().parents[1]
    manifest=root/'runs/ycb-rl-validation'/ycb/'expert-bank.json'
    if not manifest.exists():
        pytest.skip('Prepare YCB validation assets first')
    bank=json.loads(manifest.read_text())['objects'][0]
    meta=json.loads((root/'assets/ycb'/ycb/'cooked.json').read_text())
    expert=ApproachExpert(manifest=str(manifest),object_name=name)
    assert list(expert.references)==[r['id'] for r in bank['references']]
    assert len(expert.references)==40
    assert len(expert.bowl.collision.shapes)==meta['convex_count']
    with pytest.raises(ValueError,match='snapshot object'):
        expert.sync({'snapshot':{'object':'bowl'}})


def test_changed_object_geometry_is_rejected_before_planning(tmp_path):
    root=Path(__file__).resolve().parents[1]
    source=root/'runs/ycb-rl-validation/025_mug/expert-bank.json'
    if not source.exists():pytest.skip('Prepare YCB validation assets first')
    bank=json.loads(source.read_text())
    bank['objects'][0]['cooked_sha256']='0'*64
    path=tmp_path/'bank.json';path.write_text(json.dumps(bank))
    with pytest.raises(ValueError,match='asset changed: cooked'):
        ApproachExpert(manifest=str(path),object_name='mug')
