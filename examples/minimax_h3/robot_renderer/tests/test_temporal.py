import unittest

from ..config import Config
from ..temporal import legal_frames, latent_frames, pad_tail, rollout_windows, training_windows, valid_latents


class TemporalTests(unittest.TestCase):
    def test_minimal_padding_never_truncates(self):
        for count in range(1, 159):
            padded = legal_frames(count)
            self.assertGreaterEqual(padded, count)
            self.assertLess(padded - count, 17)
            self.assertEqual((padded - 5) % 17, 0)
        self.assertEqual(legal_frames(140), 141)
        self.assertEqual(legal_frames(150), 158)

    def test_training_has_full_windows_and_only_a_tail(self):
        windows = training_windows(456)
        self.assertEqual([(w.start, w.valid, w.frames) for w in windows],
                         [(0, 158, 158), (158, 158, 158), (316, 140, 141)])

    def test_rollout_never_duplicates_or_loses_boundary_frames(self):
        for total in (1, 5, 157, 158, 159, 314, 315, 316, 1000):
            output = []
            for index, window in enumerate(rollout_windows(total)):
                output.extend(range(window.start + int(index > 0), window.start + window.valid))
                self.assertEqual(window.frames, 158)
            self.assertEqual(output, list(range(total)))

    def test_padded_temporal_groups_are_excluded(self):
        self.assertEqual(latent_frames(158), 47)
        self.assertTrue(all(valid_latents(158, 158)))
        self.assertFalse(valid_latents(140, 141)[-1])
        self.assertTrue(valid_latents(140, 141)[-2])
        self.assertEqual(pad_tail([1, 2, 3], 5), [1, 2, 3, 3, 3])

    def test_continuous_training_scheduler_and_checkpoint_frequency(self):
        config = Config()
        self.assertIsNone(config.steps)
        self.assertEqual(config.frames, 158)
        self.assertAlmostEqual(config.lr(50), 1e-5)
        self.assertAlmostEqual(config.lr(1000), 1e-6)
        self.assertAlmostEqual(config.lr(10000), 1e-6)
        self.assertEqual([i for i in range(1501) if config.validate_at(i)], [0, 100, 250, 500, 1000, 1500])
        self.assertTrue(config.validate_at(1000))
        self.assertTrue(Config(steps=123).validate_at(123))


if __name__ == "__main__":
    unittest.main()
