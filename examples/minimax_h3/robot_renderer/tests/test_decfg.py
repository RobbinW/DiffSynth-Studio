import unittest

import torch
from torch import nn

from ..decfg import FrozenDeCFGLinear, convert_decfg, decfg_mode
from ..model import LoRALinear, adapter_parameters


class DeCFGTests(unittest.TestCase):
    def test_qkv_interleave_and_swiglu_gate_mapping(self):
        torch.manual_seed(17)
        heads, dim, hidden, rank = 2, 3, 5, 2
        a = torch.randn(rank, hidden)
        qkv_b = torch.randn(heads * 3 * dim, rank)
        mlp_b = torch.randn(14, rank)
        state = {}
        for name, b in (("blocks.0.attn.qkv_proj", qkv_b), ("token_refiner.blocks.0.mlp.fc1", mlp_b)):
            state[name + ".lora_A.default.weight"] = a
            state[name + ".lora_B.default.weight"] = b
        mapped = convert_decfg(state, heads, dim)
        x = torch.randn(4, hidden, requires_grad=True)
        source = ((x @ a.T) @ qkv_b.T).reshape(4, heads, 3, dim)
        converted = []
        for index, name in enumerate("qkv"):
            aa, bb = mapped["transformer_blocks.0.attn.to_" + name]
            value = (x @ aa.T) @ bb.T
            torch.testing.assert_close(value, source[:, :, index].reshape(4, -1))
            converted.append(value)
        expected_gradient = torch.autograd.grad(source.square().sum(), x, retain_graph=True)[0]
        actual_gradient = torch.autograd.grad(sum(y.square().sum() for y in converted), x)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        gate, value = ((x @ a.T) @ mlp_b.T).chunk(2, -1)
        aa, bb = mapped["token_refiner.refiner_blocks.0.ff.net.0.proj"]
        native_value, native_gate = ((x @ aa.T) @ bb.T).chunk(2, -1)
        torch.testing.assert_close(value * gate.sigmoid() * gate,
                                   native_value * native_gate.sigmoid() * native_gate)

    def test_frozen_decfg_preserves_input_gradients_and_task_updates(self):
        torch.manual_seed(23)
        base = nn.Linear(5, 4, bias=False).requires_grad_(False)
        layer = LoRALinear(base, 2)
        a, b = torch.randn(3, 5), torch.randn(4, 3)
        layer.base = FrozenDeCFGLinear(base, a, b)
        x = torch.randn(2, 5, requires_grad=True)
        baseline = base(x).detach()
        torch.testing.assert_close(layer(x), baseline, atol=0, rtol=0)
        frozen = {name: value.detach().clone() for name, value in layer.named_parameters()
                  if not value.requires_grad}
        optimizer = torch.optim.AdamW(adapter_parameters(layer).values(), lr=0.01)
        with decfg_mode(layer, True):
            expected = base(x) + (x @ a.T) @ b.T
            actual = layer(x)
            torch.testing.assert_close(actual, expected)
            expected_gradient = torch.autograd.grad(expected.square().mean(), x, retain_graph=True)[0]
            actual.square().mean().backward()
            torch.testing.assert_close(x.grad, expected_gradient)
            self.assertGreater(layer.lora_B.weight.grad.norm(), 0)
            with decfg_mode(layer, False):
                torch.testing.assert_close(layer(x), baseline, atol=0, rtol=0)
            self.assertTrue(layer.base.enabled)
        self.assertFalse(layer.base.enabled)
        optimizer.step()
        for name, value in layer.named_parameters():
            if name in frozen:
                self.assertIsNone(value.grad)
                torch.testing.assert_close(value, frozen[name], atol=0, rtol=0)
        self.assertFalse(torch.equal(layer(x), baseline))


if __name__ == "__main__":
    unittest.main()
