"""Compare Giskard collision distances at saved before/after contact states."""
import argparse
import json
from pathlib import Path

from garmi_rl.giskard_expert import ApproachExpert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--queries', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    expert = ApproachExpert()
    rows = []
    for query in json.loads(Path(args.queries).read_text()):
        command = expert.command(query)
        expert.sync(query)
        manager = expert.world.collision_manager
        contacts = manager.compute_collisions().contacts
        rows.append(dict(id=query['id'], reference=query['reference'], command=command,
                         contacts=[dict(a=str(c.body_a.name), b=str(c.body_b.name),
                                        distance=float(c.distance),
                                        buffer=float(manager.get_buffer_zone_distance(c.body_a, c.body_b)))
                                   for c in contacts if c.distance < .055]))
    Path(args.out).write_text(json.dumps(rows, indent=2))
    print('COLLISION_DIAGNOSTIC_DONE', len(rows))


if __name__ == '__main__':
    main()
