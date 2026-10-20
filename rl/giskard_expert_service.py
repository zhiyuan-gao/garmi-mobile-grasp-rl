"""CPU-only batch/JSON-lines interface to the installed Giskard interpreter."""
import argparse
import concurrent.futures
import contextlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback


class Worker:
    expert = None

    @classmethod
    def initialize(cls, manifest=None, object_name='bowl', translation_threshold=.008, orientation_threshold=.08, contact_policy='fingers_only', self_collision_buffer_m=.03, self_collision_model='legacy_srdf', distal_collision_buffer_m=.05):
        from garmi_rl.giskard_expert import ApproachExpert
        cls.expert = ApproachExpert(manifest=manifest, object_name=object_name,
            translation_threshold=translation_threshold, orientation_threshold=orientation_threshold, contact_policy=contact_policy, self_collision_buffer_m=self_collision_buffer_m,
            self_collision_model=self_collision_model,distal_collision_buffer_m=distal_collision_buffer_m)

    @classmethod
    def run(cls, request):
        started = time.perf_counter()
        try:
            return cls.expert.command(request)
        except Exception as error:
            # A failed query is recorded, never converted into a zero-action label.
            return dict(id=request['id'], status='planning_failed', error=repr(error),
                traceback=traceback.format_exc(), planning_seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--requests')
    parser.add_argument('--out')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--manifest')
    parser.add_argument('--object', default='bowl')
    parser.add_argument('--translation-threshold', type=float, default=.008)
    parser.add_argument('--orientation-threshold', type=float, default=.08)
    parser.add_argument('--contact-policy', choices=['fingers_only','fingers_and_wrist'], default='fingers_only')
    parser.add_argument('--self-collision-buffer-m', type=float, default=.03)
    parser.add_argument('--self-collision-model',choices=['legacy_srdf','usd_adjacent'],default='legacy_srdf')
    parser.add_argument('--distal-collision-buffer-m',type=float,default=.05)
    args = parser.parse_args()
    channel = sys.stdout
    with contextlib.redirect_stdout(sys.stderr):
        pool = None
        if args.workers > 1:
            pool = concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context('fork'), initializer=Worker.initialize,
                initargs=(args.manifest, args.object, args.translation_threshold, args.orientation_threshold, args.contact_policy, args.self_collision_buffer_m, args.self_collision_model, args.distal_collision_buffer_m))
        else:
            Worker.initialize(args.manifest, args.object, args.translation_threshold, args.orientation_threshold, args.contact_policy, args.self_collision_buffer_m, args.self_collision_model, args.distal_collision_buffer_m)
        try:
            if args.serve:
                print(json.dumps({'ready': True}), file=channel, flush=True)
                for line in sys.stdin:
                    started = time.perf_counter()
                    requests = json.loads(line)
                    results = list(pool.map(Worker.run, requests)) if pool else [Worker.run(q) for q in requests]
                    print(json.dumps(dict(results=results, batch_seconds=time.perf_counter()-started)), file=channel, flush=True)
            else:
                requests = json.loads(Path(args.requests).read_text())
                started = time.perf_counter()
                results = list(pool.map(Worker.run, requests)) if pool else [Worker.run(q) for q in requests]
                Path(args.out).write_text(json.dumps(dict(status='complete', results=results,
                    seconds=time.perf_counter()-started), indent=2))
                print('GISKARD_BATCH_DONE', len(results), file=channel, flush=True)
        finally:
            if pool:
                pool.shutdown()


if __name__ == '__main__':
    main()
