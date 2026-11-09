"""Comparable RGB metrics on saved videos; independent of model/GPU execution."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
from pathlib import Path
import time

PROTOCOL = "rgb-decoded-v1"
METRICS = ("rgb_mae", "psnr", "ssim", "rgb_rmse")
DEFINITION = {
    "protocol": PROTOCOL,
    "prediction": "decoded generated.mp4, RGB [0,1]",
    "reference": "original real video, LANCZOS resize, nearest timestamp at (start+i)/fps",
    "frames": "all valid frames, including first frame; no padded frames",
    "psnr": "mean frame PSNR, MSE floor 1e-12 (120 dB ceiling)",
    "ssim": "RGB channel mean; Gaussian 11x11 sigma=1.5; K1=.01 K2=.03; population covariance; crop 5-pixel border",
    "aggregation": "frame mean per clip/seed, equal clip/seed mean per checkpoint",
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def frame_metrics(prediction, reference):
    import numpy as np
    from scipy.ndimage import gaussian_filter

    x = np.asarray(prediction, dtype=np.float32) / 255
    y = np.asarray(reference, dtype=np.float32) / 255
    if x.shape != y.shape or x.ndim != 3 or x.shape[2] != 3 or min(x.shape[:2]) < 11:
        raise ValueError(f"Expected matching RGB images >=11 pixels, got {x.shape}, {y.shape}")
    error = x - y
    mse = float(np.mean(error * error, dtype=np.float64))

    def smooth(value):
        return gaussian_filter(value, sigma=(1.5, 1.5, 0), truncate=3.5, mode="reflect")

    ux, uy = smooth(x), smooth(y)
    vx, vy = smooth(x * x) - ux * ux, smooth(y * y) - uy * uy
    covariance = smooth(x * y) - ux * uy
    score = ((2 * ux * uy + .01**2) * (2 * covariance + .03**2)) / (
        (ux * ux + uy * uy + .01**2) * (vx + vy + .03**2))
    return dict(rgb_mae=float(np.mean(np.abs(error), dtype=np.float64)),
                psnr=float(-10 * np.log10(max(mse, 1e-12))),
                ssim=float(np.mean(score[5:-5, 5:-5], dtype=np.float64)),
                rgb_rmse=float(np.sqrt(mse)))


def evaluate_file(path, fps):
    import av
    import numpy as np
    from .media import read_video, resample

    path = Path(path)
    item = json.loads(path.read_text())
    metadata = item["metadata"]
    source, pts = read_video(metadata["real"], metadata["width"], metadata["height"])
    real = resample(source, pts, (metadata["start"] + np.arange(metadata["valid"])) / fps)
    scores = []
    with av.open(str(path.with_name("generated.mp4"))) as video:
        for index, frame in enumerate(video.decode(video=0)):
            if index == len(real):
                break
            scores.append(frame_metrics(frame.to_ndarray(format="rgb24"), real[index]))
    if len(scores) != len(real):
        raise ValueError(f"{path}: expected {len(real)} valid frames, decoded {len(scores)}")
    result = {key: float(np.mean([score[key] for score in scores])) for key in METRICS}
    # Keep the original pre-encoding score explicitly separate from this protocol.
    if "psnr" in item and "metric_protocol" not in item:
        item["psnr_preencode"] = item["psnr"]
    item.update(result, metric_protocol=PROTOCOL, metric_definition=DEFINITION,
                metric_frames=len(scores), metric_fps=fps, metrics_computed_at=time.time())
    atomic_json(path.with_name("frame_metrics.json"), dict(protocol=PROTOCOL, frames=scores))
    atomic_json(path, item)
    return str(path), result


def summarize(videos, checkpoints, expected_keys):
    """Never treat a partial/different validation panel as a comparable checkpoint."""
    expected_keys = set(expected_keys)
    summaries = []
    for step in sorted(set(checkpoints) | {v["step"] for v in videos}):
        rows = [v for v in videos if v["step"] == step and v.get("metric_protocol") == PROTOCOL]
        keys = [(v["metadata"]["id"], v["seed"]) for v in rows]
        complete = bool(expected_keys) and set(keys) == expected_keys and len(keys) == len(expected_keys)
        def means(items):
            return {key: sum(v[key] for v in items) / len(items) for key in METRICS} if items else {}
        summaries.append(dict(step=step, protocol=PROTOCOL,
            status="complete" if complete else "pending" if not rows else "partial",
            count=len(rows), expected=len(expected_keys),
            metrics=means(rows) if complete else {},
            partial_metrics=means(rows) if rows and not complete else {},
            by_source={source: dict(count=len(group), **means(group))
                       for source in sorted({v["metadata"]["source"] for v in rows})
                       for group in [[v for v in rows if v["metadata"]["source"] == source]]},
            samples=[dict(id=v["metadata"]["id"], seed=v["seed"], source=v["metadata"]["source"],
                          **{key: v[key] for key in METRICS}) for v in rows]))
    return summaries


def save_summaries(out, summaries):
    out = Path(out)
    atomic_json(out / "metrics_history.json", dict(definition=DEFINITION, checkpoints=summaries))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["step", "status", "count", "expected", *METRICS])
    writer.writeheader()
    for row in summaries:
        writer.writerow({**{key: row[key] for key in ("step", "status", "count", "expected")}, **row["metrics"]})
        atomic_json(out / "validation" / f"step_{row['step']:06d}" / "metrics_summary.json", row)
    temporary = out / "metrics_history.csv.tmp"
    temporary.write_text(buffer.getvalue())
    temporary.replace(out / "metrics_history.csv")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    # One evaluator owns a node's output tree, including across launcher restarts.
    import fcntl
    args.run.mkdir(parents=True, exist_ok=True)
    with (args.run / "metrics.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Metrics evaluator already active", flush=True)
            return
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while True:
                paths = sorted((args.run / "validation").glob("step_*/*/*/metrics.json"), reverse=True)
                pending = []
                for path in paths:
                    try:
                        item = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        continue  # An older trainer may still be writing its completion marker.
                    if item.get("metric_protocol") != PROTOCOL and path.with_name("generated.mp4").exists():
                        pending.append(path)
                from itertools import repeat
                for path, scores in pool.map(evaluate_file, pending, repeat(args.fps)):
                    print(json.dumps(dict(path=path, **scores)), flush=True)
                if not args.watch:
                    break
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
