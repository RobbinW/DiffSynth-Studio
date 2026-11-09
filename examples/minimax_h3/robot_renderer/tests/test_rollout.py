import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from ..media import read_video, write_video
from ..rollout import image_hash, stitch
from ..temporal import rollout_windows


class RolloutMediaTests(unittest.TestCase):
    def test_lossless_boundary_survives_save_and_reload(self):
        pixels = np.random.default_rng(42).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        reference = Image.fromarray(pixels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last_frame.png"
            reference.save(path)
            self.assertEqual(image_hash(reference), image_hash(Image.open(path)))
            changed = pixels.copy()
            changed[0, 0, 0] ^= 1
            self.assertNotEqual(image_hash(reference), image_hash(Image.fromarray(changed)))

    def test_encoded_rollout_keeps_timeline_and_short_tail(self):
        # Frame number encoded as two 4-bit grayscale patches, robust to H.264.
        timeline = []
        for index in range(316):
            pixels = np.zeros((32, 64, 3), dtype=np.uint8)
            pixels[:, :32] = (index % 16) * 16
            pixels[:, 32:] = (index // 16) * 12
            timeline.append(Image.fromarray(pixels))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, window in enumerate(rollout_windows(len(timeline))):
                segment = root / "segments" / f"{index:06d}"
                segment.mkdir(parents=True)
                write_video(timeline[window.start:window.start + window.valid], segment / "generated.mp4", 24)
                (segment / "segment.json").write_text(json.dumps(dict(width=64, height=32, valid=window.valid)))
            self.assertEqual(stitch(root, 24), len(timeline))
            actual, timestamps = read_video(root / "generated.mp4", 64, 32)
            self.assertEqual(len(actual), len(timeline))
            np.testing.assert_allclose(timestamps, np.arange(316) / 24, atol=1e-6)
            for index, frame in enumerate(actual):
                pixels = np.asarray(frame)
                decoded = round(float(pixels[8:24, 8:24].mean()) / 16)
                decoded += 16 * round(float(pixels[8:24, 40:56].mean()) / 12)
                self.assertEqual(decoded, index)


if __name__ == "__main__":
    unittest.main()
