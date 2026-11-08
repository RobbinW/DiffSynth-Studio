import unittest
from types import SimpleNamespace

import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import patchify_video_latents

from ..attention import StableAttention
from ..model import LoRALinear, parameter_dtype, without_adapter
from ..runtime import NativeRuntime, training_inputs, unpack
from ..train import video_loss


class NumericTests(unittest.TestCase):
    def test_dtype_query_accepts_fsdp_tensor_views(self):
        layer = torch.nn.Linear(4, 4, bias=False)
        view = layer.weight.detach()
        del layer._parameters["weight"]
        layer.weight = view
        self.assertEqual(list(layer.parameters()), [])
        self.assertEqual(parameter_dtype(layer), torch.float32)
        self.assertEqual(parameter_dtype(torch.nn.Sequential(torch.nn.Linear(4, 4))), torch.float32)

    def test_preparation_keeps_prompt_outside_layout_only_pipeline(self):
        class Layout:
            def __call__(self, **kwargs):
                assert "prompt_embeds" not in kwargs
                return SimpleNamespace(get=lambda key: None)

        runtime = NativeRuntime.__new__(NativeRuntime)
        runtime.pipe, runtime.device, runtime.steps = Layout(), "cpu", 50
        prompt = torch.randn(1, 3, 32)
        prepared = runtime.prepare({"prompt_embeds": prompt}, dict(height=32, width=32, frames=158), 42)
        self.assertIs(prepared["prompt_embeds"], prompt)

    def test_sampling_resets_stateful_schedulers_between_clips(self):
        from diffusers import MiniMaxH3Scheduler

        runtime = NativeRuntime.__new__(NativeRuntime)
        runtime.device, runtime.steps = "cpu", 50
        runtime.pipe = SimpleNamespace(scheduler=MiniMaxH3Scheduler(shift=12),
                                       audio_scheduler=MiniMaxH3Scheduler(shift=3))
        runtime.pipe.scheduler.set_timesteps(51)
        runtime.pipe.audio_scheduler.set_timesteps(51)
        video, audio = torch.randn(2, 96), torch.randn(3, 32)
        prepared = {key: None for key in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices", "prompt_embeds")}
        prepared.update(latents=video, audio_latents=audio, num_condition_video_rows=0,
                        num_condition_audio_rows=0, timesteps=runtime.pipe.scheduler.timesteps,
                        audio_timesteps=runtime.pipe.audio_scheduler.timesteps,
                        row_timestep_plan=[(torch.tensor([0.5]), torch.zeros(1, dtype=torch.long))] * 50)

        class ZeroVelocity(torch.nn.Module):
            def forward(self, **inputs):
                return torch.zeros_like(inputs["hidden_states"]), torch.zeros_like(inputs["audio_hidden_states"])

        zero_velocity = ZeroVelocity()

        first, actual_audio = runtime.sample(zero_velocity, prepared, dict(frames=5, height=32, width=32))
        second, _ = runtime.sample(zero_velocity, prepared, dict(frames=5, height=32, width=32))
        torch.testing.assert_close(actual_audio, audio)
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        torch.testing.assert_close(first, unpack(video, 2, 2, 2))

    def test_attention_gradient_matches_explicit_softmax(self):
        torch.manual_seed(13)
        values = [torch.randn(1, 3, length, 8, requires_grad=True) for length in (7, 11, 11)]
        q, k, v = values
        expected = ((q @ k.transpose(-1, -2) / 8**0.5).softmax(-1)) @ v
        upstream = torch.randn_like(expected)
        reference = torch.autograd.grad(expected, values, upstream)
        actual = StableAttention.apply(*values)
        gradients = torch.autograd.grad(actual, values, upstream)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        for a, b in zip(gradients, reference):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)

    def test_zero_adapter_preserves_base_and_base_is_frozen(self):
        base = torch.nn.Linear(8, 4).requires_grad_(False)
        layer = LoRALinear(base, 2)
        x = torch.randn(3, 8)
        torch.testing.assert_close(layer(x), base(x), atol=0, rtol=0)
        layer(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertGreater(layer.lora_B.weight.grad.abs().sum(), 0)
        with torch.no_grad():
            layer.lora_B.weight.add_(0.1)
        with without_adapter(layer):
            torch.testing.assert_close(layer(x), base(x), atol=0, rtol=0)
        self.assertTrue(layer.enabled)

    def test_native_flow_sign_audio_and_pack_roundtrip(self):
        target = torch.randn(1, 24, 2, 4, 4)
        clean = patchify_video_latents(target, (1, 2, 2))
        noise = torch.randn_like(clean)
        condition = torch.randn(3, clean.shape[-1])
        audio_noise, silence = torch.randn(7, 32), torch.randn(5, 32)
        prepared = {"latents": torch.cat((condition, noise)), "num_condition_video_rows": 3,
                    "timesteps": torch.tensor([0.25]), "audio_latents": audio_noise,
                    "num_condition_audio_rows": 2, "audio_timesteps": torch.tensor([0.6])}
        video, actual_audio, velocity = training_inputs(prepared, target, silence, 0)
        torch.testing.assert_close(video[:3], condition)
        torch.testing.assert_close(video[3:] + 0.75 * velocity, clean)
        torch.testing.assert_close(actual_audio[:2], audio_noise[:2])
        torch.testing.assert_close(actual_audio[2:], .6 * silence + .4 * audio_noise[2:])
        torch.testing.assert_close(prepared["audio_latents"], audio_noise)
        torch.testing.assert_close(unpack(clean, 2, 4, 4), target)

    def test_tail_padding_cannot_reduce_reported_video_loss(self):
        metadata = {"frames": 22, "valid": 20, "height": 32, "width": 32}
        target = torch.zeros(7, 96)
        prediction = torch.ones_like(target)
        expected = video_loss(prediction, target, metadata)
        prediction[-1] = 10000
        torch.testing.assert_close(video_loss(prediction, target, metadata), expected)


if __name__ == "__main__":
    unittest.main()
