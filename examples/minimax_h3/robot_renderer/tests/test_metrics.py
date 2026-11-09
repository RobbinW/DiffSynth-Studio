import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from ..metrics import PROTOCOL, evaluate_file, frame_metrics, summarize


class MetricTests(unittest.TestCase):
    def test_identity_and_constant_offset(self):
        x = np.full((24, 32, 3), 80, dtype=np.uint8)
        identical = frame_metrics(x, x)
        self.assertEqual(identical['rgb_mae'], 0)
        self.assertEqual(identical['psnr'], 120)
        self.assertEqual(identical['ssim'], 1)
        offset = frame_metrics(x, x + 10)
        self.assertAlmostEqual(offset['rgb_mae'], 10 / 255, places=6)
        self.assertAlmostEqual(offset['psnr'], 20 * np.log10(255 / 10), places=5)
        expected_ssim = (2 * (80 / 255) * (90 / 255) + .01**2) / (
            (80 / 255)**2 + (90 / 255)**2 + .01**2)
        self.assertAlmostEqual(offset['ssim'], expected_ssim, places=4)

    def test_matches_standard_ssim(self):
        try:
            from skimage.metrics import structural_similarity
        except ImportError:
            self.skipTest('scikit-image is optional reference validation dependency')
        rng = np.random.default_rng(43)
        x = rng.integers(0, 256, (41, 39, 3), dtype=np.uint8)
        y = np.clip(x.astype(float) + rng.normal(0, 20, x.shape), 0, 255).astype(np.uint8)
        expected = structural_similarity(x.astype(np.float32) / 255, y.astype(np.float32) / 255,
            channel_axis=2, data_range=1, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        self.assertAlmostEqual(frame_metrics(x, y)['ssim'], expected, places=6)

    def test_saved_video_alignment_and_valid_frame_limit(self):
        from PIL import Image
        from ..media import read_video, write_video
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = [Image.fromarray(np.full((24, 32, 3), level, dtype=np.uint8))
                      for level in (20, 60, 100, 140)]
            write_video(frames, root / 'real.mp4', 24)
            write_video(frames[1:], root / 'generated.mp4', 24)
            metadata = dict(real=str(root / 'real.mp4'), width=32, height=24, start=1, valid=2)
            path = root / 'metrics.json'
            path.write_text(json.dumps(dict(metadata=metadata, seed=42, psnr=19.0)))
            evaluate_file(path, 24)
            item = json.loads(path.read_text())
            self.assertEqual(item['metric_frames'], 2)
            self.assertEqual(item['psnr_preencode'], 19.0)
            generated, _ = read_video(root / 'generated.mp4', 32, 24)
            real, _ = read_video(root / 'real.mp4', 32, 24)
            expected = np.mean([frame_metrics(generated[i], real[i + 1])['rgb_mae'] for i in range(2)])
            self.assertEqual(item['rgb_mae'], expected)
            self.assertEqual(len(json.loads((root / 'frame_metrics.json').read_text())['frames']), 2)

    def test_panel_completeness_and_protocol_separation(self):
        def row(step, seed, **extras):
            return dict(step=step, seed=seed, metadata=dict(id='clip', source='test'),
                        metric_protocol=PROTOCOL, rgb_mae=.1, psnr=20, ssim=.8, rgb_rmse=.2, **extras)
        old = row(0, 1)
        old.pop('metric_protocol')
        rows = [old, row(100, 1), row(200, 1), row(200, 2)]
        history = summarize(rows, [0, 100, 200, 300], [('clip', 1), ('clip', 2)])
        self.assertEqual([x['status'] for x in history], ['pending', 'partial', 'complete', 'pending'])
        self.assertEqual(history[1]['metrics'], {})
        self.assertEqual(history[2]['metrics']['psnr'], 20)
        self.assertEqual(summarize([row(200, 1)] * 2, [200], [('clip', 1)])[0]['status'], 'partial')


if __name__ == '__main__':
    unittest.main()
