"""Launch the native recipe through the established four-node DCU container stack.

Run on the controller. A new remote root owns immutable source/config/manifests,
preparation logs, cached tensors and the training run.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import random
import shlex
import subprocess
import time

from .config import Config
from .sampling import Samples, validation_rows


PACKAGE = "examples/minimax_h3/robot_renderer"
MODULE = "examples.minimax_h3.robot_renderer"
ENV = {"NCCL_SOCKET_IFNAME": "eth0", "NCCL_IB_GID_INDEX": "3",
       "NCCL_IB_DISABLE": "0", "NCCL_NET_GDR_LEVEL": "2",
       "NCCL_IB_QPS_PER_CONNECTION": "4", "NCCL_IB_TC": "160",
       "NCCL_IB_TIMEOUT": "22", "GLOO_SOCKET_IFNAME": "eth0",
       "NCCL_DEBUG": "WARN", "OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1",
       "HIP_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
       "HSA_FORCE_FINE_GRAIN_PCIE": "1",
       "PYTORCH_ALLOC_CONF": "expandable_segments:True", "PYTHONUNBUFFERED": "1"}


def run(command):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"{shlex.join(command)} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--nodes", nargs=4, required=True, help="Controller SSH aliases")
    parser.add_argument("--peers", nargs=4, required=True, help="Inter-node SSH aliases")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--container-root", required=True)
    parser.add_argument("--container", default="h3-train-v3")
    parser.add_argument("--python", default="/work/venv/bin/python")
    parser.add_argument("--model", default="/work/models/MiniMax-H3")
    parser.add_argument("--port", type=int, default=29816)
    parser.add_argument("--config")
    parser.add_argument("--prepared", action="store_true", help="Reuse completed preparation shards")
    parser.add_argument("--resume", help="Container checkpoint folder available on all nodes")
    parser.add_argument("--skip-initial-validation", action="store_true")
    parser.add_argument("--revision", help="Published source commit when the controller is outside a Git checkout")
    parser.add_argument("--status", required=True)
    args = parser.parse_args()
    status = Path(args.status)
    status.mkdir(parents=True, exist_ok=True)
    config = Config.load(args.config)
    config.save(status / "config.json")
    root, cr = args.remote_root, args.container_root
    workers = []
    revision = args.revision or run(["git", "rev-parse", "HEAD"])
    manifest_sha256 = hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()

    def record(phase, **extra):
        result = dict(phase=phase, time=time.time(), config=json.loads((status / "config.json").read_text()),
                      nodes=args.nodes, remote_root=root, workers=workers, code_commit=revision,
                      manifest_sha256=manifest_sha256, **extra)
        (status / "launch.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(dict(phase=phase, time=result["time"], **extra)), flush=True)

    def container(command, **extra):
        env = dict(ENV, PYTHONPATH=cr, **extra)
        return ["docker", "exec", *[value for key, item in env.items() for value in ("-e", key+"="+item)],
                args.container, args.python, *command]

    def prepare(node_rank):
        node = args.nodes[node_rank]
        run(["ssh", node, "mkdir -p " + shlex.quote(root+"/"+PACKAGE) + " " + shlex.quote(root+"/cache") + " " + shlex.quote(root+"/logs")])
        run(["rsync", "-rt", "--exclude=__pycache__", PACKAGE+"/", node+":"+root+"/"+PACKAGE+"/"])
        run(["scp", "-q", args.manifest, node+":"+root+"/manifest.jsonl"])
        run(["scp", "-q", str(status/"config.json"), node+":"+root+"/config.json"])
        if args.prepared:
            run(["scp", "-q", node+":"+root+"/cache/index.*.jsonl", str(status)+"/"])
            return
        processes = []
        for gpu in range(8):
            shard = node_rank*8+gpu
            command = container(["-m", MODULE+".prepare", "--model", args.model,
                "--manifest", cr+"/manifest.jsonl", "--out", cr+"/cache", "--config", cr+"/config.json",
                "--shard", str(shard), "--shards", "32"], HIP_VISIBLE_DEVICES=str(gpu))
            remote = shlex.join(command)+" > "+shlex.quote(root+f"/logs/prepare_{shard:02d}.log")+" 2>&1"
            processes.append(subprocess.Popen(["ssh", node, remote]))
        codes = [process.wait() for process in processes]
        if any(codes):
            raise RuntimeError(f"Preparation failed on {node}: {codes}; inspect {root}/logs/prepare_*.log")
        run(["scp", "-q", node+":"+root+"/cache/index.*.jsonl", str(status)+"/"])

    try:
        record("preparing_native_158f_cache")
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(prepare, range(4)))
        rows = [json.loads(line) for path in sorted(status.glob("index.*.jsonl"))
                for line in path.read_text().splitlines() if line.strip()]
        if len(list(status.glob("index.*.jsonl"))) != 32:
            raise ValueError("All 32 preparation index shards must be complete before training")
        samples = Samples([row for row in rows if row["split"] == "train"])
        validation = validation_rows(rows)
        owners = {row["file"]: int(path.name.split(".")[1])//8
                  for path in status.glob("index.*.jsonl")
                  for row in map(json.loads, path.read_text().splitlines())}
        record("distributing_planned_cache", train_windows=sum(row["split"]=="train" for row in rows))
        # Continuous training can eventually sample every window. Finite runs
        # retain the cheaper exact-schedule replication.
        def distribute(rank):
            node = args.nodes[rank]
            wanted = {row["file"] for row in validation[rank::4]}
            if config.steps is None:
                wanted.update(row["file"] for row in rows if row["split"] == "train")
            else:
                for update in range(1, config.steps+1):
                    for micro in range(config.accumulation):
                        rng = random.Random(config.seed+(update*config.accumulation+micro)*4+rank)
                        wanted.add(samples.choose(rng)["file"])
            for owner, peer in enumerate(args.peers):
                if owner == rank:
                    continue
                listing = status/f"copy_{rank}_{owner}.txt"
                listing.write_text("".join(name+"\n" for name in sorted(wanted) if owners[name]==owner))
                run(["scp", "-q", str(listing), node+":"+root+"/copy.txt"])
                run(["ssh", node, shlex.join(["rsync", "-rt", "--files-from="+root+"/copy.txt",
                                             peer+":"+root+"/cache/", root+"/cache/"])])
            # Encoders run as container root. Skip identical existing shards;
            # rsync can atomically replace files through the host-owned directory.
            run(["rsync", "-r", "--checksum", *map(str, sorted(status.glob("index.*.jsonl"))),
                 node+":"+root+"/cache/"])
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(distribute, range(4)))
        master = run(["ssh", args.nodes[0], "hostname -I"]).split()[0]
        record("encoding_missing_audio_silence")
        command = container(["-m", MODULE+".audio", "--model", args.model,
            "--out", cr+"/cache/silence.pt", "--fps", str(config.fps),
            "--frames", *map(str, sorted({row["frames"] for row in rows}))], HIP_VISIBLE_DEVICES="0")
        run(["ssh", args.nodes[0], shlex.join(command)+" > "+shlex.quote(root+"/logs/silence.log")+" 2>&1"])
        for node in args.nodes[1:]:
            run(["ssh", node, shlex.join(["rsync", "-r", "--checksum",
                args.peers[0]+":"+root+"/cache/silence.pt", root+"/cache/"])])
        if config.decfg_path:
            record("qualifying_training_only_decfg")
            command = container(["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                "-m", MODULE+".tests.check_decfg", "--model", args.model,
                "--sample", cr+"/cache/"+validation[0]["file"], "--config", cr+"/config.json",
                "--out", cr+"/decfg_qualification.json",
                *(["--adapter", args.resume+"/adapter.safetensors"] if args.resume else [])])
            run(["ssh", args.nodes[0], shlex.join(command)+" > "+shlex.quote(root+"/logs/decfg_qualification.log")+" 2>&1"])
        for node in args.nodes:
            evaluator = container(["-m", MODULE+".metrics", "--run", cr+"/run",
                                   "--fps", str(config.fps), "--watch"],
                                  OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            run(["ssh", node, "nohup " + shlex.join(evaluator) + " >> " +
                 shlex.quote(root+"/logs/metrics.log") + " 2>&1 </dev/null & echo $!"])
        for rank, node in enumerate(args.nodes):
            resume = ["--resume", args.resume] if args.resume else []
            if args.skip_initial_validation:
                resume.append("--skip-initial-validation")
            command = container(["-m", "torch.distributed.run", "--nnodes=4", "--nproc_per_node=8",
                "--node_rank="+str(rank), "--master_addr="+master, "--master_port="+str(args.port),
                "-m", MODULE+".train", "--model", args.model, "--cache", cr+"/cache",
                "--out", cr+"/run", "--config", cr+"/config.json", *resume])
            remote = "nohup "+shlex.join(command)+" > "+shlex.quote(root+f"/logs/train_node{rank}.log")+" 2>&1 </dev/null & echo $!"
            pid = run(["ssh", node, remote])
            workers.append(dict(node=node, pid=pid, command=command))
        record("training_submitted", world=32, effective_batch=8)
        for node in args.nodes:
            run(["scp", "-q", str(status/"launch.json"), node+":"+root+"/launch.json"])
    except Exception as error:
        record("failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
