"""Split and teacher-configuration checks for independent object imitation runs."""
import collections
import copy
import random

from .expert_collection import strict_success


def case_key(case):
    return tuple(case['robot_state'] + case['object_state'])


def validate_partitions(train, validation, heldout, object_name, train_per_ref=300,
                        validation_per_ref=25, reference_count=40):
    seen_ids, seen_states, references = set(), set(), None
    heldout_states = {case_key(c) for c in heldout}
    assert len(heldout_states) == len(heldout)
    assert all(c['object'] == object_name for c in heldout)
    for split, manifest, quota in [('train', train, train_per_ref),
                                    ('validation', validation, validation_per_ref)]:
        assert manifest['status'] == 'complete'
        rows = manifest['demonstrations']
        counts = collections.Counter(r['reference'] for r in rows)
        assert len(counts) == reference_count and set(counts.values()) == {quota}
        if references is None:
            references = set(counts)
        assert set(counts) == references
        for row in rows:
            case = row['initial_case']
            assert row['split'] == split and strict_success(row)
            assert case['object'] == object_name and case['reference'] == row['reference']
            assert row['id'] not in seen_ids and case_key(case) not in seen_states
            assert case_key(case) not in heldout_states
            seen_ids.add(row['id']); seen_states.add(case_key(case))
    assert set(c['reference'] for c in heldout) == references
    return sorted(references)


def sample_correction_cases(manifest, per_reference, seed):
    pools = collections.defaultdict(list)
    for row in manifest['demonstrations']:
        case = copy.deepcopy(row['initial_case'])
        case['source_initial_id'] = row['id']
        pools[row['reference']].append(case)
    rng = random.Random(seed)
    result = [c for ref in sorted(pools) for c in rng.sample(pools[ref], per_reference)]
    rng.shuffle(result)
    return result


def teacher_arguments(profile):
    task = profile['garmi_task']; multi = task['multi_object']
    assert len(multi['objects']) == 1
    precision, collision = profile['expert_precision'], profile['expert_collision']
    return ['--manifest', multi['manifest'], '--object', multi['objects'][0],
            '--contact-policy', task['contact_policy'],
            '--translation-threshold', str(precision['translation_threshold']),
            '--orientation-threshold', str(precision['orientation_threshold']),
            '--self-collision-buffer-m', str(collision['self_buffer_m']),
            '--self-collision-model', collision['model'],
            '--distal-collision-buffer-m', str(collision['distal_buffer_m'])]


def aggregate_labels(base, label_sets):
    parents = {r['id']: r for r in base['demonstrations']}
    ids = set()
    for entry in label_sets:
        assert entry['command_version'] == base['command_version']
        assert entry['count'] == len(entry['samples']) > 0
        for sample in entry['samples']:
            case = sample['initial_case']
            parent = parents[case.get('source_initial_id', case['id'])]
            assert case_key(case) == case_key(parent['initial_case'])
            assert case['object'] == parent['initial_case']['object']
            assert sample['reference'] == parent['reference']
            assert sample['expert_action_executed'] is False and sample['id'] not in ids
            ids.add(sample['id'])
    return dict(base, dagger_label_sets=label_sets, dagger_labels=len(ids))
