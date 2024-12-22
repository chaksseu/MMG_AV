import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

import torch
from packaging import version
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
    randn_tensor,
    replace_example_docstring,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, PNDMScheduler, DDIMScheduler
from transformers import PretrainedConfig, AutoTokenizer
import torch.nn as nn
import os, json, PIL
import numpy as np
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from diffusers.utils.outputs import BaseOutput
import matplotlib.pyplot as plt


from train_MMG_Model_1210 import CrossModalCoupledUNet
from auffusion_pipe_functions import (
    _execution_device, _encode_prompt, prepare_extra_step_kwargs, 
    prepare_latents, ConditionAdapter, import_model_class_from_model_name_or_path
)



LRELU_SLOPE = 0.1
MAX_WAV_VALUE = 32768.0


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_config(config_path):
    config = json.loads(open(config_path).read())
    config = AttrDict(config)
    return config

def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def apply_weight_norm(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        weight_norm(m)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        # self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3)) # change: 80 --> 512
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            if (k-u) % 2 == 0:
                self.ups.append(weight_norm(
                    ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2)))
            else:
                self.ups.append(weight_norm(
                    ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2+1, output_padding=1)))
            
            # self.ups.append(weight_norm(
            #     ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)),
            #                     k, u, padding=(k-u)//2)))
            

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None):
        if subfolder is not None:
            pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, subfolder)
        config_path = os.path.join(pretrained_model_name_or_path, "config.json")
        ckpt_path = os.path.join(pretrained_model_name_or_path, "vocoder.pt")

        config = get_config(config_path)
        vocoder = cls(config)

        state_dict_g = torch.load(ckpt_path)
        vocoder.load_state_dict(state_dict_g["generator"])
        vocoder.eval()
        vocoder.remove_weight_norm()
        return vocoder
    
    @torch.no_grad()
    def inference(self, mels, lengths=None):
        self.eval()
        with torch.no_grad():
            wavs = self(mels).squeeze(1)

        wavs = (wavs.cpu().numpy() * MAX_WAV_VALUE).astype("int16")

        if lengths is not None:
            wavs = wavs[:, :lengths]

        return wavs
    

def encode_audio_prompt(
    text_encoder_list,
    tokenizer_list,
    adapter_list,
    tokenizer_model_max_length,
    dtype,
    prompt,
    device,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    do_classifier_free_guidance=False
):

    assert len(text_encoder_list) == len(tokenizer_list), "Number of text_encoders must match number of tokenizers"
    if adapter_list is not None:
        assert len(text_encoder_list) == len(adapter_list), "Number of text_encoders must match number of adapters"

    def get_prompt_embeds(prompt_list, device):
        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        prompt_embeds_list = []
        for prompt in prompt_list:
            encoder_hidden_states_list = []

            # Generate condition embedding
            for j in range(len(text_encoder_list)):
                # get condition embedding using condition encoder
                input_ids = tokenizer_list[j](prompt, return_tensors="pt").input_ids.to(device)            
                cond_embs = text_encoder_list[j](input_ids).last_hidden_state # [bz, text_len, text_dim]
                # padding to max_length
                if cond_embs.shape[1] < tokenizer_model_max_length: 
                    cond_embs = torch.functional.F.pad(cond_embs, (0, 0, 0, tokenizer_model_max_length - cond_embs.shape[1]), value=0)
                else:
                    cond_embs = cond_embs[:, :tokenizer_model_max_length, :]

                # use condition adapter
                if adapter_list is not None:
                    cond_embs = adapter_list[j](cond_embs)
                    encoder_hidden_states_list.append(cond_embs)

            prompt_embeds = torch.cat(encoder_hidden_states_list, dim=1)
            prompt_embeds_list.append(prompt_embeds)

        prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        return prompt_embeds


    if prompt_embeds is None:           
        prompt_embeds = get_prompt_embeds(prompt, device)

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds).to(dtype=prompt_embeds.dtype, device=device)
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    return prompt_embeds

def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs



latents = None
batch_text = ["text"]
shape = latents.shape()
device = "cuda"
seed = 1234
dtype = torch.float32
num_train_timesteps = 1000
num_inference_steps = 25
batch_size = 1
guidance_scale =7.5
generator = torch.Generator(device=device).manual_seed(seed)
eta = 0.0
output_type = "pt"
pretrained_model_name_or_path = "auffusion/auffusion-full"
duration = 3.2
audio_length = int(duration * 16000)


vocoder = Generator.from_pretrained(pretrained_model_name_or_path, subfolder="vocoder").to(device, dtype)

pretrained_model_name_or_path = (
        snapshot_download(pretrained_model_name_or_path)
        if not os.path.isdir(pretrained_model_name_or_path) 
        else pretrained_model_name_or_path
    )

with torch.no_grad():
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae").to(device, dtype).eval()        
vae.requires_grad_(False)
vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)
condition_json_path = os.path.join(pretrained_model_name_or_path, "condition_config.json")
condition_json_list = json.loads(open(condition_json_path).read())
text_encoder_list, tokenizer_list, adapter_list = [], [], []

with torch.no_grad():
    for i, condition_item in enumerate(condition_json_list):
        text_encoder_path = os.path.join(pretrained_model_name_or_path, condition_item["text_encoder_name"])
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
        tokenizer_list.append(tokenizer)
    
        text_encoder_cls = import_model_class_from_model_name_or_path(text_encoder_path)
        text_encoder = text_encoder_cls.from_pretrained(text_encoder_path).to(device, dtype)
        text_encoder.requires_grad_(False)
        text_encoder_list.append(text_encoder)
    
        adapter_path = os.path.join(pretrained_model_name_or_path, condition_item["condition_adapter_name"])
        adapter = ConditionAdapter.from_pretrained(adapter_path).to(device, dtype)
        adapter.requires_grad_(False)
        adapter_list.append(adapter)

from omegaconf import OmegaConf
from utils.utils import instantiate_from_config



# Audio UNet 준비
audio_unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
audio_unet.eval()

# Video UNet 준비
video_config = OmegaConf.load('configs/inference_t2v_512_v2.0.yaml')
video_model = instantiate_from_config(video_config.model)
state_dict = torch.load('scripts/evaluation/model.ckpt')['state_dict']
video_model.load_state_dict(state_dict, strict=True).eval()
video_unet = video_model.model.diffusion_model.eval()


# CrossModal Config
cross_modal_config = {
    'layer_channels': [320, 640, 1280, 1280, 1280, 640],
    'd_head': 64,
    'device': device
}


# Initialize the combined model
model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config)


do_classifier_free_guidance = guidance_scale > 1.0

# Prepare timesteps
scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps)
scheduler.set_timesteps(num_inference_steps, device=device)
timesteps = scheduler.timesteps


# Encode input prompt


with torch.no_grad():
    audio_text_embed = encode_audio_prompt(
        text_encoder_list=text_encoder_list,
        tokenizer_list=tokenizer_list,
        adapter_list=adapter_list,
        tokenizer_model_max_length=77,
        dtype=dtype,
        prompt=batch_text,
        device=device
        )

    video_text_embed = video_model.get_learned_conditioning(batch_text)
    if do_classifier_free_guidance:
        negative_prompt_embeds = torch.zeros_like(video_text_embed).to(dtype=video_text_embed.dtype, device=device)
        video_text_embed = torch.cat([negative_prompt_embeds, video_text_embed])


batch_size = len(batch_text)
audio_shape = (batch_size, 4, 256, 320)  
video_shape = (batch_size, 4, 40, 32, 32) 

# Prepare latent variables
audio_latents = randn_tensor(audio_shape, generator=generator, device=device, dtype=dtype)
video_latents = randn_tensor(video_shape, generator=generator, device=device, dtype=dtype)

audio_latents = audio_latents * scheduler.init_noise_sigma
video_latents = video_latents * scheduler.init_noise_sigma

# 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
extra_step_kwargs = prepare_extra_step_kwargs(generator, eta)


# Denoising loop
for i, t in enumerate(timesteps):


    # expand the latents if we are doing classifier free guidance
    audio_latents = torch.cat([audio_latents] * 2) if do_classifier_free_guidance else audio_latents
    audio_latents = scheduler.scale_model_input(audio_latents, t)
    video_latents = torch.cat([video_latents] * 2) if do_classifier_free_guidance else video_latents
    video_latents = scheduler.scale_model_input(video_latents, t)


    # Forward
    audio_output, video_output = model(
        audio_latents=audio_latents,
        audio_timestep=timesteps,
        audio_encoder_hidden_states=audio_text_embed,
        video_latents=video_latents,
        video_timestep=timesteps,
        video_context=video_text_embed,
        video_fps=video_latents.shape[2]
        )

    # perform guidance
    if do_classifier_free_guidance:
        audio_output_uncond, audio_output_text = audio_output.chunk(2)
        audio_output = audio_output_uncond + guidance_scale * (audio_output_text - audio_output_uncond)
        video_output_uncond, video_output_text = video_output.chunk(2)
        video_output = video_output_uncond + guidance_scale * (video_output_text - video_output_uncond)


    # compute the previous noisy sample x_t -> x_t-1
    audio_latents = scheduler.step(audio_output, t, audio_latents, **extra_step_kwargs, return_dict=False)[0]
    video_latents = scheduler.step(video_output, t, video_latents, **extra_step_kwargs, return_dict=False)[0]



audio_image = vae.decode(audio_latents / vae.config.scaling_factor, return_dict=False)[0]
do_denormalize = [True] * audio_image.shape[0]
audio_image = image_processor.postprocess(audio_image, output_type=output_type, do_denormalize=do_denormalize)

def denormalize_spectrogram(
    data: torch.Tensor,
    max_value: float = 200, 
    min_value: float = 1e-5, 
    power: float = 1, 
) -> torch.Tensor:
    
    assert len(data.shape) == 3, "Expected 3 dimensions, got {}".format(len(data.shape))

    max_value = np.log(max_value)
    min_value = np.log(min_value)
    # Flip Y axis: image origin at the top-left corner, spectrogram origin at the bottom-left corner
    data = torch.flip(data, [1])    
    if data.shape[0] == 1:
        data = data.repeat(3, 1, 1)        
    assert data.shape[0] == 3, "Expected 3 channels, got {}".format(data.shape[0])
    data = data[0]
    # Reverse the power curve
    data = torch.pow(data, 1 / power)
    # Rescale to max value
    spectrogram = data * (max_value - min_value) + min_value

    return spectrogram

# Generate audio
spectrograms, audios = []
for img in audio_image:
    spectrogram = denormalize_spectrogram(img)
    audio = vocoder.inference(spectrogram, lengths=audio_length)[0]
    audios.append(audio)
    spectrograms.append(spectrogram)



from scipy.io.wavfile import write

audio_savedir="MMG_output/audio"
for i, audio in enumerate(audios):
    audio_path = os.path.join(audio_savedir, f"audio_{batch_text[i]}_batch_idx*i")
    write(audio_path, 16,000, audio)


import torchvision

def save_videos(batch_tensors, savedir, filenames, fps=12.5):
    # b,c,t,h,w
    for idx, vid_tensor in enumerate(batch_tensors):
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1., 1.)
        video = video.permute(1, 0, 2, 3) # t,c,h,w
        frame_grids = [torchvision.utils.make_grid(framesheet, nrow=1) for framesheet in video] #[3, 1*h, n*w]
        grid = torch.stack(frame_grids, dim=0) # stack in temporal dim [t, 3, h, w]
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
        savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
        torchvision.io.write_video(savepath, grid, fps=fps, video_codec='h264', options={'crf': '10'})

video_savedir="MMG_output/video"
video_filenames = f"video_{batch_text[i]}_batch_idx*i"

batch_images = video_model.decode_first_stage_2DAE(video_latents)
save_videos(batch_images, video_savedir, video_filenames, fps=12.5)