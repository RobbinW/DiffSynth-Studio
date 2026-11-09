"""Offline native image/video conditioning and paired real-video latents."""

import torch
import numpy as np
from accelerate import dispatch_model
from diffusers import ComponentsManager, ModularPipeline
from diffusers.modular_pipelines import SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference, MiniMaxH3VideoReference
from diffusers.modular_pipelines.minimax_h3.decoders import MiniMaxH3VideoDecodeStep

from .media import structured_prompt
from .temporal import pad_tail


class Encoder:
    def __init__(self, model_path, device, text_devices=None):
        blocks = ModularPipeline.from_pretrained(model_path).blocks.get_workflow("ref2va").sub_blocks
        manager = ComponentsManager()

        def chain(names):
            return SequentialPipelineBlocks.from_blocks_dict(
                {name: blocks[name] for name in names}
            ).init_pipeline(model_path, components_manager=manager)

        self.conditioned = chain(("before_encode", "text_encoder", "vae_encoder"))
        self.conditioned.load_components(dtype=torch.bfloat16)
        encoder = self.conditioned.text_encoder
        encoder.set_attn_implementation("sdpa")
        mapping = {"model.visual": device, "model.language_model.embed_tokens": device,
                   "model.language_model.norm": device, "model.language_model.rotary_emb": device,
                   "lm_head": device}
        layers = len(encoder.model.language_model.layers)
        layer_devices = ([text_devices[index * len(text_devices) // layers] for index in range(layers)]
                         if text_devices else [device if index < 30 else "cpu" for index in range(layers)])
        mapping.update({f"model.language_model.layers.{index}": owner
                        for index, owner in enumerate(layer_devices)})
        dispatch_model(encoder, device_map=mapping, main_device=device, force_hooks=True)
        self.conditioned.vae.to(device)
        self.target = chain(("before_encode", "vae_encoder"))
        self.target.update_components(**{
            name: getattr(self.conditioned, name) for name in self.target.pretrained_component_names
        })

    def canvas(self, metadata):
        for pipe in (self.conditioned, self.target):
            pipe.register_to_config(canvas_short_edge=min(metadata["width"], metadata["height"]),
                                    canvas_max_pixels=metadata["width"] * metadata["height"])

    @torch.no_grad()
    def conditions(self, reference, proxy, metadata, prompt):
        self.canvas(metadata)
        # Qwen reads at2fps and needs two sampled frames; a free video reference
        # can be longer than a5-frame target without adding target supervision.
        proxy = pad_tail(proxy, max(22, len(proxy)))
        # Only the front-end duration validator needs >=5s. Actual reference media
        # and the subsequent native target layout retain the short tail's length.
        state = self.conditioned(
            prompt=structured_prompt(prompt),
            references=[MiniMaxH3ImageReference(image=reference),
                        MiniMaxH3VideoReference(frames=np.asarray(proxy), fps=24)],
            height=metadata["height"], width=metadata["width"],
            num_frames=max(124, metadata["frames"]))
        return {key: state.get(key) for key in ("prompt_embeds", "text_token_tags", "condition_latents",
                                               "audio_condition_latents", "normalized_references")}

    @torch.no_grad()
    def video(self, real, metadata):
        self.canvas(metadata)
        state = self.target(references=[MiniMaxH3VideoReference(frames=np.asarray(real), fps=24)],
                            height=metadata["height"], width=metadata["width"],
                            num_frames=max(124, metadata["frames"]))
        return state.get("condition_latents")[0].float()


class Decoder:
    def __init__(self, model_path, device):
        self.pipe = SequentialPipelineBlocks.from_blocks_dict({"decode": MiniMaxH3VideoDecodeStep()}).init_pipeline(model_path)
        self.pipe.load_components(dtype=torch.float32)
        self.pipe.vae.to(device)

    @torch.no_grad()
    def __call__(self, latents):
        return self.pipe(latents=latents, output_type="pil").get("videos")[0]
