import torch
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.core import load_state_dict
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download


pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    # model_configs=[
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    #     ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth"),
    # ],
    model_configs=[
    ModelConfig(["/data/dex/Motus/pretrained_models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors",
                 "/data/dex/Motus/pretrained_models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors",
                 "/data/dex/Motus/pretrained_models/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors"]),
    ModelConfig(["/data/dex/Motus/pretrained_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"]),
    ModelConfig(["/data/dex/Motus/pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]),
    ]
)
pipe.load_lora(pipe.dit, "/data/dex/DiffSynth-Studio/models/train/RoboTwin_Wan2.2_5B_LoRA_121_32/epoch-1.safetensors", alpha=1)

input_image = VideoData("/data/dex/DiffSynth-Studio/data/RoboTwin/wanvideo_dataset/video_000101.mp4", height=480, width=640)[0]

video = pipe(
    prompt="Using the left arm, grasp the orange bottle on the table. Make sure the robot's hand firmly holds the bottle.",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    input_image=input_image,
    num_frames=121,
    seed=42, tiled=True,
    height=480, width=640,
)
save_video(video, "video_Wan2.2-TI2V-5B1114.mp4", fps=15, quality=5)
