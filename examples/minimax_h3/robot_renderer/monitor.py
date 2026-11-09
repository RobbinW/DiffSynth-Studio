"""Read-only SSH monitor and validation mirror for the local training dashboard."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

if __package__:
    from .metrics import save_summaries, summarize
else:
    from metrics import save_summaries, summarize


REMOTE = r'''
import json, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
rank = int(sys.argv[2])
run = root / 'run'
config = json.loads((root / 'config.json').read_text())
processes = subprocess.check_output(['ps', '-eo', 'args='], text=True).splitlines()
workers = sum(line.startswith('/work/venv/bin/python -u -m examples.minimax_h3.robot_renderer.train ')
              and str(root.name) in line for line in processes)
busy = []
for path in sorted(pathlib.Path('/sys/class/drm').glob('card*/device/gpu_busy_percent')):
    busy.append(int(path.read_text()))
validations = []
for path in sorted((run / 'validation').glob('step_*/*/*/metrics.json')):
    if not path.with_name('generated.mp4').exists():
        continue
    item = json.loads(path.read_text())
    item.update(step=int(path.parents[2].name.split('_')[1]),
                directory=str(path.parent.relative_to(run)), completed_at=path.stat().st_mtime)
    validations.append(item)
log = root / 'logs' / ('train_node%d.log' % rank)
tail = log.read_text(errors='replace').splitlines()[-160:] if log.exists() else []
errors = [line for line in tail if any(tag in line for tag in
          ('Traceback (', 'OutOfMemoryError', 'ValueError: Nonfinite', 'RuntimeError:'))]
result = dict(checked_at=time.time(), connected=True, workers=workers, gpu_busy=busy,
              validations=validations, errors=errors[-5:])
if rank == 0:
    path = run / 'train.jsonl'
    # Ignore a partially written final line during an optimizer-log append.
    history = [json.loads(line) for line in path.read_text().splitlines(keepends=True)
               if line.endswith('\n')] if path.exists() else []
    for row in history:
        for key in ('loss', 'grad_norm', 'seconds', 'lr'):
            if not __import__('math').isfinite(row[key]):
                row[key] = None
    groups = {}
    for line in (root / 'manifest.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['split'] == 'val':
            groups.setdefault(row['source'], set()).add(row['group'])
    result.update(history=history, config=config,
        last_update_at=path.stat().st_mtime if path.exists() else None,
        expected_validation=sum(min(2,len(v)) for v in groups.values()) * len(config['validation_seeds']),
        checkpoints=[int(p.name.split('_')[1]) for p in sorted(run.glob('step_*'))
                     if (p / 'training.pt').exists() and (p / 'adapter.safetensors').exists()])
print(json.dumps(result, allow_nan=False))
'''


def command(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, timeout=90, **kwargs)
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-600:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--hosts', nargs=4, required=True)
    parser.add_argument('--remote-root', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'validation').mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).with_name('dashboard.html'), args.out / 'index.html')
    destination = args.out / 'snapshot.json'
    previous = json.loads(destination.read_text()) if destination.exists() else {}

    def collect(pair):
        rank, host = pair
        old = next((node for node in previous.get('nodes', []) if node['host'] == host), {})
        try:
            output = command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                              shlex.join(['python3', '-', args.remote_root, str(rank)])], input=REMOTE)
            node = json.loads(output)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            return dict(old, host=host, connected=False, error=str(error))
        try:
            command(['rsync', '-rt', '-e', 'ssh -o BatchMode=yes -o ConnectTimeout=10',
                     '--include=*/', '--include=generated.mp4', '--include=metrics.json',
                     '--include=frame_metrics.json', '--include=comparison.jpg', '--include=comparison.mp4', '--exclude=*',
                     host + ':' + args.remote_root + '/run/validation/',
                     str(args.out / 'validation') + '/'])
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            node['media_error'] = str(error)
        return dict(node, host=host)

    while True:
        with ThreadPoolExecutor(max_workers=4) as pool:
            nodes = list(pool.map(collect, enumerate(args.hosts)))
        lead = nodes[0]
        videos = []
        for node in nodes:
            for item in node.get('validations', []):
                folder = args.out / item['directory']
                if (folder / 'generated.mp4').exists():
                    videos.append(dict(item, comparison=(folder / 'comparison.mp4').exists()))
        snapshot = dict(collected_at=time.time(), refresh_seconds=args.interval,
                        run_name=args.remote_root.rsplit('/', 1)[-1], nodes=nodes,
                        history=lead.get('history', []), config=lead.get('config', {}),
                        last_update_at=lead.get('last_update_at'),
                        checkpoints=lead.get('checkpoints', []),
                        expected_validation=lead.get('expected_validation', 0), videos=videos)
        remote_videos = [item for node in nodes for item in node.get('validations', [])]
        expected_keys = {(v['metadata']['id'], v['seed']) for v in remote_videos if v['step'] == 0}
        if len(expected_keys) != snapshot['expected_validation']:
            expected_keys = set()
        summaries = summarize(remote_videos, snapshot['checkpoints'], expected_keys)
        snapshot['metric_history'] = summaries
        save_summaries(args.out, summaries)
        shutil.copyfile(Path(__file__).with_name('dashboard.html'), args.out / 'index.html')
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False, allow_nan=False))
        temporary.replace(destination)
        previous = snapshot
        print(json.dumps(dict(time=snapshot['collected_at'],
            update=snapshot['history'][-1]['update'] if snapshot['history'] else 0,
            connected=sum(n['connected'] for n in nodes), videos=len(videos))), flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
