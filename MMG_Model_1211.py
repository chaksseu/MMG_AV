import torch
from diffusers import PNDMScheduler, UNet2DConditionModel
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer, CrossModalTransformer,
    Downsample, Upsample, TimestepBlock
)
from lvdm.common import checkpoint
from einops import rearrange
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config

from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn
from train_MMG_Model_1210 import CrossModalCoupledUNet
from safetensors.torch import load_file
from tqdm import tqdm

def main():
    """
    Main function: Loads models, prepares inputs, runs the multi-modal UNet forward pass,
    and compares the outputs with the original models.
    """
    batch_size = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    ############################# Prepare Audio #############################

    audio_unet = UNet2DConditionModel.from_pretrained("auffusion/auffusion-full", subfolder="unet").to(device, dtype)
    audio_unet.eval()
    
    # Prepare audio input data
    audio_height = 256  # Adjust as needed
    audio_width = 320  # Adjust as needed
    generator = torch.Generator(device=device).manual_seed(42)
    audio_encoder_hidden_states = torch.randn((batch_size, 77, audio_unet.config.cross_attention_dim), device=device, dtype=dtype)

    audio_latents = torch.randn(
        (batch_size, audio_unet.config.in_channels, audio_height // 8, audio_width // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    # Set audio timesteps
    scheduler = PNDMScheduler.from_pretrained("auffusion/auffusion", subfolder="scheduler")
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    t = timesteps[0]

    # Compute original audio UNet output
    print("Computing original audio UNet output...")
    with torch.no_grad():
        audio_original_output = audio_unet(
            audio_latents,
            t,
            encoder_hidden_states=audio_encoder_hidden_states,
            return_dict=False,
        )[0]

    
    ############################# Prepare Video #############################

    video_config_path = 'configs/inference_t2v_512_v2.0.yaml'  # Update to actual path
    video_ckpt_path = 'scripts/evaluation/model.ckpt'  # Update to actual path

    video_config = OmegaConf.load(video_config_path)
    video_model = instantiate_from_config(video_config.model)
    video_model.load_state_dict(torch.load(video_ckpt_path)['state_dict'], strict=False)
    video_unet = video_model.model.diffusion_model.eval()

    # Generate input data
    video_fps = 40
    video_latents = torch.randn(batch_size, 4, video_fps, 32, 32)  # (B, C, T, H, W)
    video_timestep = torch.tensor([10])
    video_context = torch.randn(1, 77, 1024)  # Example: text embeddings

    # Move models and tensors to device
    video_unet = video_unet.to(device)
    video_latents = video_latents.to(device)
    video_timestep = video_timestep.to(device)
    video_context = video_context.to(device)

    # Compute original video UNet output
    print("Computing original video UNet output...")
    with torch.no_grad():
        video_original_output = video_unet(video_latents, video_timestep, context=video_context, fps=video_fps)



    ################ CMCU #################
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device
    }


    # Initialize the combined model
    model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config)
    checkpoint_path="mmg_checkpoints/lr_1e-06_batch_1536_global_step_1200_vggsound_sparse/model.safetensors"
    checkpoint = load_file(checkpoint_path) 
    model.load_state_dict(checkpoint)
    model = model.to(device=device, dtype=dtype)

    print("Computing multi-modal UNet output...")
    with torch.no_grad():
        multi_modal_output = model(
            audio_latents=audio_latents,
            audio_timestep=t,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            video_latents=video_latents,
            video_timestep=video_timestep,
            video_context=video_context,
            video_fps=video_fps
        )


    ######################### Compare Outputs #########################
    def compare_outputs(audio_output1: torch.Tensor, audio_output2: torch.Tensor, 
                       video_output1: torch.Tensor, video_output2: torch.Tensor, tol: float = 1e-5):
        """
        Compares two sets of audio and video output tensors.
        """
        if torch.equal(audio_output1, audio_output2):
            print("Audio outputs are exactly identical.")
        elif torch.allclose(audio_output1, audio_output2, atol=tol):
            print(f"Audio outputs are identical within tolerance {tol}.")
        else:
            print("Audio outputs differ.")
            diff = torch.abs(audio_output1 - audio_output2)
            print(f"Max audio difference: {diff.max().item()}")
            print(f"Mean audio difference: {diff.mean().item()}")
        
        if torch.equal(video_output1, video_output2):
            print("Video outputs are exactly identical.")
        elif torch.allclose(video_output1, video_output2, atol=tol):
            print(f"Video outputs are identical within tolerance {tol}.")
        else:
            print("Video outputs differ.")
            diff = torch.abs(video_output1 - video_output2)
            print(f"Max video difference: {diff.max().item()}")
            print(f"Mean video difference: {diff.mean().item()}")

    # Compare the outputs
    print("Comparing outputs...")
    audio_mm_output, video_mm_output = multi_modal_output
    compare_outputs(
        audio_output1=audio_original_output,
        audio_output2=audio_mm_output,
        video_output1=video_original_output,
        video_output2=video_mm_output
    )


if __name__ == "__main__":
    main()
