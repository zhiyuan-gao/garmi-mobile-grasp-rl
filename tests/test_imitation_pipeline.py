"""Reject split leakage and wrong-object teacher settings before corrective training."""
import copy

import pytest

from garmi_rl.imitation_pipeline import (validate_partitions, sample_correction_cases,
                                        teacher_arguments, aggregate_labels)


def row(split, i):
    return dict(id=f'{split}-{i}', split=split, reference='handle', reason='success',
        handover_trigger=1, handover_distance=.002, handover_angle=.02,
        close_start_distance=.004, close_start_angle=.04, height=.12, contact=True,
        supported=False, slip=.005, hold_seconds=2.05, collision_bodies={},
        initial_case=dict(id=f'{split}-{i}',object='mug',reference='handle',
            robot_state=[i,0],object_state=[1,2]))


def data():
    return (dict(status='complete',command_version='joint',demonstrations=[row('train',0),row('train',1)]),
            dict(status='complete',command_version='joint',demonstrations=[row('validation',2)]),
            [dict(object='mug',reference='handle',robot_state=[3,0],object_state=[1,2])])


def validate(t,v,h):
    return validate_partitions(t,v,h,'mug',2,1,1)


def test_partition_check_rejects_shared_initial_state_even_with_different_ids():
    t,v,h=data();assert validate(t,v,h)==['handle']
    v['demonstrations'][0]['initial_case']['robot_state']=[0,0]
    with pytest.raises(AssertionError):validate(t,v,h)


def test_heldout_and_wrong_object_cannot_enter_training():
    t,v,h=data();h[0]['robot_state']=[1,0]
    with pytest.raises(AssertionError):validate(t,v,h)
    t,v,h=data();t['demonstrations'][0]['initial_case']['object']='pitcher'
    with pytest.raises(AssertionError):validate(t,v,h)


def test_aggregate_rejects_validation_labels_in_train_and_changed_parent_state():
    t,v,h=data();cases=sample_correction_cases(t,1,42)
    sample=dict(id='query',initial_case=cases[0],reference='handle',expert_action_executed=False)
    labels=[dict(command_version='joint',count=1,samples=[sample])]
    assert aggregate_labels(t,labels)['dagger_labels']==1
    with pytest.raises(KeyError):aggregate_labels(v,labels)
    changed=copy.deepcopy(labels);changed[0]['samples'][0]['initial_case']['robot_state']=[99,0]
    with pytest.raises(AssertionError):aggregate_labels(t,changed)
    with pytest.raises(AssertionError):aggregate_labels(t,labels+labels)


def test_teacher_uses_mug_contact_and_geometry_parameters_not_bowl_defaults():
    profile=dict(garmi_task=dict(multi_object=dict(objects=['mug'],manifest='mug-bank.json'),
        contact_policy='fingers_and_wrist'),expert_precision=dict(translation_threshold=.002,orientation_threshold=.02),
        expert_collision=dict(model='usd_adjacent',self_buffer_m=.005,distal_buffer_m=.003))
    args=teacher_arguments(profile);values=dict(zip(args[::2],args[1::2]))
    assert values['--object']=='mug' and values['--manifest']=='mug-bank.json'
    assert values['--contact-policy']=='fingers_and_wrist'
    assert values['--self-collision-model']=='usd_adjacent'
    assert float(values['--distal-collision-buffer-m'])==.003
    assert float(values['--translation-threshold'])==.002
