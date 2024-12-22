import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

from accelerate import Accelerator
from diffusers import PNDMScheduler, DDPMScheduler, UNet2DConditionModel
from einops import rearrange
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb

from utils.utils import instantiate_from_config
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    CrossModalTransformer, 
    Downsample, 
    Upsample, 
    TimestepBlock, 
    SpatialTransformer, 
    TemporalTransformer
)

# 환경 변수 설정 (필요한 경우에만 설정)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

# wandb 초기화
wandb.init(project="my_diffusion_project", name="train_run")

#######################################
# Dataset & Dataloader
#######################################
class LatentsDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, csv_file):
        import pandas as pd
        self.root_dir = root_dir
        self.data_info = pd.read_csv(csv_file)

        self.video_latents_dir = os.path.join(root_dir, "video_latents")
        self.audio_latents_dir = os.path.join(root_dir, "audio_latents")
        self.video_text_embeds_dir = os.path.join(root_dir, "video_text_embeds")
        self.audio_text_embeds_dir = os.path.join(root_dir, "audio_text_embeds")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        video_file = self.data_info.iloc[idx]["Video"]
        audio_file = self.data_info.iloc[idx]["Audio"]
        video_text_embed_file = self.data_info.iloc[idx]["Video_text_embed"]
        audio_text_embed_file = self.data_info.iloc[idx]["Audio_text_embed"]

        video_latent = torch.load(os.path.join(self.video_latents_dir, video_file))
        audio_latent = torch.load(os.path.join(self.audio_latents_dir, audio_file))
        video_text_embed = torch.load(os.path.join(self.video_text_embeds_dir, video_text_embed_file))
        audio_text_embed = torch.load(os.path.join(self.audio_text_embeds_dir, audio_text_embed_file))

        sample = {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "video_text_embed": video_text_embed,
            "audio_text_embed": audio_text_embed
        }
        return sample

def get_dataloader(root_dir, csv_file, batch_size=4, shuffle=True, num_workers=0):
    dataset = LatentsDataset(root_dir=root_dir, csv_file=csv_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

#######################################
# CrossModalCoupledUNet
#######################################
class CrossModalCoupledUNet(nn.Module):
    """
    Audio-Video Cross Modal Coupled UNet:
    - Audio UNet와 Video UNet를 결합하고, CrossModalTransformer를 이용해 중간 레벨에서 상호작용.
    - Audio/Video UNet는 Freeze 상태에서, CrossModalTransformer 파트만 학습.
    """
    def __init__(self, audio_unet, video_unet, cross_modal_config):
        super(CrossModalCoupledUNet, self).__init__()
        # Audio UNet Freeze
        self.audio_unet = audio_unet
        for param in self.audio_unet.parameters():
            param.requires_grad = False

        # Video UNet Freeze
        self.video_unet = video_unet
        for param in self.video_unet.parameters():
            param.requires_grad = False

        # Cross Modal Transformer 초기화
        self.audio_cmt = nn.ModuleList()
        self.video_cmt = nn.ModuleList()
        layer_channels = cross_modal_config['layer_channels']
        d_head = cross_modal_config.get('d_head', 64)

        for channel in layer_channels:
            n_heads = channel // d_head
            audio_transformer = CrossModalTransformer(
                in_channels=channel, n_heads=n_heads, d_head=d_head, depth=1,
                context_dim=channel, use_linear=True, use_checkpoint=True,
                disable_self_attn=False, img_cross_attention=False
            )
            video_transformer = CrossModalTransformer(
                in_channels=channel, n_heads=n_heads, d_head=d_head, depth=1,
                context_dim=channel, use_linear=True, use_checkpoint=True,
                disable_self_attn=False, img_cross_attention=False
            )
            self.initialize_cross_modal_transformer(audio_transformer)
            self.initialize_cross_modal_transformer(video_transformer)
            self.audio_cmt.append(audio_transformer)
            self.video_cmt.append(video_transformer)

    def initialize_basic_transformer_block(self, block):
        for name, param in block.named_parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif param.dim() == 1:
                nn.init.zeros_(param)

    def initialize_cross_modal_transformer(self, transformer):
        if isinstance(transformer.proj_in, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(transformer.proj_in.weight)
            if transformer.proj_in.bias is not None:
                nn.init.zeros_(transformer.proj_in.bias)

        for block in transformer.transformer_blocks:
            self.initialize_basic_transformer_block(block)

        if isinstance(transformer.proj_out, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(transformer.proj_out.weight)
            if transformer.proj_out.bias is not None:
                nn.init.zeros_(transformer.proj_out.bias)

    def audio_down_blocks(self, down_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, output_states):
        """
        Audio down_block 처리 함수
        """
        if getattr(down_block, "has_cross_attention", False):
            for resnet, attn in zip(down_block.resnets, down_block.attentions):
                hidden_states = resnet(hidden_states, emb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )[0]
                output_states += (hidden_states,)
        else:
            for resnet in down_block.resnets:
                hidden_states = resnet(hidden_states, emb)
                output_states += (hidden_states,)
        return hidden_states, output_states

    def audio_mid_blocks(self, audio_unet, hidden_states, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs):
        """
        Audio mid_block 처리
        """
        if getattr(audio_unet.mid_block, "has_cross_attention", False):
            hidden_states = audio_unet.mid_block.resnets[0](hidden_states, emb)
            for resnet, attn in zip(audio_unet.mid_block.resnets[1:], audio_unet.mid_block.attentions):
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )[0]
                hidden_states = resnet(hidden_states, emb)
        else:
            for resnet in audio_unet.mid_block.resnets:
                hidden_states = resnet(hidden_states, emb)
        return hidden_states

    def audio_up_blocks(self, up_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, res_hidden_states_tuple):
        """
        Audio up_block 처리
        """
        if getattr(up_block, "has_cross_attention", False):
            for resnet, attn in zip(up_block.resnets, up_block.attentions):
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
                hidden_states = resnet(hidden_states, emb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )[0]
        else:
            for resnet in up_block.resnets:
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
                hidden_states = resnet(hidden_states, emb)
        return hidden_states

    def process_video_sublayer(self, sublayer, h, video_emb, video_context, batch_size):
        """
        Video sublayer 처리
        """
        if isinstance(sublayer, TimestepBlock):
            h = sublayer(h, video_emb, batch_size=batch_size)
        elif isinstance(sublayer, SpatialTransformer):
            h = sublayer(h, video_context)
        elif isinstance(sublayer, TemporalTransformer):
            h = rearrange(h, '(b f) c h w -> b c f h w', b=batch_size)
            h = sublayer(h, video_context)
            h = rearrange(h, 'b c f h w -> (b f) c h w')
        else:
            h = sublayer(h)
        return h

    def video_down_block(self, block_idx, video_unet, h, video_emb, video_context, batch_size, hs):
        """
        Video down_block 처리
        """
        for sublayer in video_unet.input_blocks[block_idx]:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        if block_idx == 0 and video_unet.addition_attention:
            h = video_unet.init_attn(h, video_emb, context=video_context, batch_size=batch_size)
        hs.append(h)
        return h, hs

    def video_up_block(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
        """
        Video up_block 처리
        """
        h = torch.cat([h, hs.pop()], dim=1)
        for sublayer in video_unet.output_blocks[block_idx]:
            if isinstance(sublayer, Upsample):
                break
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        return h, hs

    def video_upsample(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
        """
        Video upsample 처리
        """
        h = self.process_video_sublayer(video_unet.output_blocks[block_idx][-1], h, video_emb, video_context, batch_size)
        return h, hs

    def cross_modal_transform(self, audio_hidden_states, h, audio_block_idx):
        """
        Audio/Video latent를 CrossModalTransformer에 통과 시키는 로직을 함수로 분리
        """
        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k=k)
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f')

        cross_video_latent_token = self.video_cmt[audio_block_idx](cross_video_latent_token, cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[audio_block_idx](cross_audio_latent_token, cross_video_latent_token)

        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k)
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a)

        return audio_hidden_states, h

    def forward(
        self,
        audio_latents, 
        audio_timestep, 
        audio_encoder_hidden_states,
        video_latents, 
        video_timestep, 
        video_context=None, 
        video_fps=8,
        audio_attention_mask=None, 
        audio_cross_attention_kwargs=None
    ):
        """
        Forward pass:
        - Audio와 Video 모두 Noisy Latents와 timestep을 받아서 처리
        - Audio/Video UNet의 하위/중간/상위 블록을 거치며 CrossModalTransformer를 통해 상호 정보교환
        """
        # Audio 전처리
        if self.audio_unet.config.center_input_sample:
            audio_latents = 2 * audio_latents - 1.0

        if not torch.is_tensor(audio_timestep):
            audio_timestep = torch.tensor([audio_timestep], dtype=torch.int64, device=audio_latents.device)
        audio_timestep = audio_timestep.expand(audio_latents.shape[0])
        audio_t_emb = self.audio_unet.time_proj(audio_timestep).to(dtype=audio_latents.dtype)
        audio_emb = self.audio_unet.time_embedding(audio_t_emb)
        if self.audio_unet.time_embed_act is not None:
            audio_emb = self.audio_unet.time_embed_act(audio_emb)

        if self.audio_unet.encoder_hid_proj is not None:
            audio_encoder_hidden_states = self.audio_unet.encoder_hid_proj(audio_encoder_hidden_states)

        audio_hidden_states = self.audio_unet.conv_in(audio_latents)
        audio_down_block_res_samples = (audio_hidden_states,)

        # Video 전처리
        video_emb = self.video_unet.time_embed(timestep_embedding(video_timestep, self.video_unet.model_channels))
        if self.video_unet.fps_cond:
            video_fps_tensor = torch.full_like(video_timestep, video_fps) if isinstance(video_fps, int) else video_fps
            video_emb += self.video_unet.fps_embedding(timestep_embedding(video_fps_tensor, self.video_unet.model_channels))

        b, _, t, _, _ = video_latents.shape
        if video_context is not None:
            video_context = video_context.repeat_interleave(repeats=t, dim=0)
        video_emb = video_emb.repeat_interleave(repeats=t, dim=0)
        h = rearrange(video_latents, 'b c t h w -> (b t) c h w').type(self.video_unet.dtype)
        video_emb = video_emb.to(h.dtype)
        hs = []

        # Down Blocks
        # Audio Down Block 0
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            down_block=self.audio_unet.down_blocks[0],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            output_states=audio_down_block_res_samples
        )
        # Video Down Blocks
        h, hs = self.video_down_block(0, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(1, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(2, self.video_unet, h, video_emb, video_context, b, hs)

        # Cross Modal 0
        audio_hidden_states, h = self.cross_modal_transform(audio_hidden_states, h, 0)

        # Audio Downsample
        audio_hidden_states = self.audio_unet.down_blocks[0].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        # Video Downsample
        h, hs = self.video_down_block(3, self.video_unet, h, video_emb, video_context, b, hs)

        # Audio Down Block 1
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            down_block=self.audio_unet.down_blocks[1],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            output_states=audio_down_block_res_samples
        )
        # Video Down Blocks
        h, hs = self.video_down_block(4, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(5, self.video_unet, h, video_emb, video_context, b, hs)

        # Cross Modal 1
        audio_hidden_states, h = self.cross_modal_transform(audio_hidden_states, h, 1)

        # Audio Downsample
        audio_hidden_states = self.audio_unet.down_blocks[1].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        # Video Downsample
        h, hs = self.video_down_block(6, self.video_unet, h, video_emb, video_context, b, hs)

        # Audio Down Block 2
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            down_block=self.audio_unet.down_blocks[2],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            output_states=audio_down_block_res_samples
        )
        h, hs = self.video_down_block(7, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(8, self.video_unet, h, video_emb, video_context, b, hs)

        # Cross Modal 2
        audio_hidden_states, h = self.cross_modal_transform(audio_hidden_states, h, 2)

        # Audio Downsample
        audio_hidden_states = self.audio_unet.down_blocks[2].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        # Video Downsample
        h, hs = self.video_down_block(9, self.video_unet, h, video_emb, video_context, b, hs)

        # Audio Down Block 3
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            down_block=self.audio_unet.down_blocks[3],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            output_states=audio_down_block_res_samples
        )
        h, hs = self.video_down_block(10, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(11, self.video_unet, h, video_emb, video_context, b, hs)

        # Mid Block
        audio_hidden_states = self.audio_mid_blocks(
            self.audio_unet, audio_hidden_states, audio_emb,
            audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs
        )
        for sublayer in self.video_unet.middle_block:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, b)

        # Up Blocks
        # Audio Up Block 0
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[0].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[0].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[0],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            res_hidden_states_tuple=audio_res_samples
        )
        h, hs = self.video_up_block(0, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(1, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(2, self.video_unet, h, hs, video_emb, video_context, b)

        # Cross Modal 3
        audio_hidden_states, h = self.cross_modal_transform(audio_hidden_states, h, 3)

        # Audio Upsample
        audio_hidden_states = self.audio_unet.up_blocks[0].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(2, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio Up Block 1
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[1].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[1].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[1],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            res_hidden_states_tuple=audio_res_samples
        )
        h, hs = self.video_up_block(3, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(4, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(5, self.video_unet, h, hs, video_emb, video_context, b)

        # Cross Modal 4
        audio_hidden_states, h = self.cross_modal_transform(audio_hidden_states, h, 4)
        audio_hidden_states = self.audio_unet.up_blocks[1].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(5, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio Up Block 2
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[2].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[2].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[2],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            res_hidden_states_tuple=audio_res_samples
        )
        h, hs = self.video_up_block(6, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(7, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(8, self.video_unet, h, hs, video_emb, video_context, b)

        # Cross Modal 5
        audio_hidden_states, h = self.cross_modal_transform(audio_hidden_states, h, 5)
        audio_hidden_states = self.audio_unet.up_blocks[2].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(8, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio Up Block 3
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[3].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[3].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[3],
            hidden_states=audio_hidden_states,
            encoder_hidden_states=audio_encoder_hidden_states,
            emb=audio_emb,
            attention_mask=audio_attention_mask,
            cross_attention_kwargs=audio_cross_attention_kwargs,
            res_hidden_states_tuple=audio_res_samples
        )
        h, hs = self.video_up_block(9, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(10, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(11, self.video_unet, h, hs, video_emb, video_context, b)

        # Output Layers
        if self.audio_unet.conv_norm_out is not None:
            audio_hidden_states = self.audio_unet.conv_norm_out(audio_hidden_states)
            if self.audio_unet.conv_act is not None:
                audio_hidden_states = self.audio_unet.conv_act(audio_hidden_states)
        audio_hidden_states = self.audio_unet.conv_out(audio_hidden_states)

        for sublayer in self.video_unet.out:
            h = sublayer(h)
        h = rearrange(h, '(b f) c h w -> b c f h w', b=b)

        audio_output, video_output = audio_hidden_states, h
        return audio_output, video_output


#######################################
# main
#######################################
def main():
    # Accelerate 초기화
    accelerator = Accelerator(
        mixed_precision="no",  # 필요시 bf16
        gradient_accumulation_steps=1
    )
    device = accelerator.device

    # DataLoader
    root_dir = "latents_data_32s_40frames"
    csv_file = os.path.join(root_dir, "dataset_info.csv")
    batch_size = 6
    dataloader = get_dataloader(root_dir, csv_file, batch_size=batch_size, shuffle=True, num_workers=0)

    # Audio UNet
    audio_unet = UNet2DConditionModel.from_pretrained("auffusion/auffusion-full", subfolder="unet")
    audio_unet.eval()

    # Video UNet
    video_config = OmegaConf.load('configs/inference_t2v_512_v2.0.yaml')
    video_model = instantiate_from_config(video_config.model)
    state_dict = torch.load('scripts/evaluation/model.ckpt')['state_dict']
    video_model.load_state_dict(state_dict, strict=True)
    video_unet = video_model.model.diffusion_model.eval()

    # CrossModal Config
    cross_modal_config = {
        'layer_channels': [320, 640, 1280, 1280, 1280, 640],
        'd_head': 64,
        'device': device
    }

    # 모델 초기화
    model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config)

    # trainable한 파라미터만 optimizer에 전달
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-5)

    num_train_timesteps = 1000
    noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    # Null Text Embed (가정)
    audio_null_text_embed = torch.load("null_text_embedding/audio_text_embeds/audio_text_embed_null.pt")
    audio_null_text_embed = audio_null_text_embed.unsqueeze(0).expand(batch_size, -1, -1)
    video_null_text_embed = torch.load("null_text_embedding/video_text_embeds/video_text_embed_null.pt")
    video_null_text_embed = video_null_text_embed.unsqueeze(0).expand(batch_size, -1, -1)

    num_epochs = 100
    global_step = 0
    for epoch in range(num_epochs):
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                audio_latents = batch["audio_latent"].to(device)
                video_latents = batch["video_latent"].to(device)
                audio_text_embed = batch["audio_text_embed"].to(device)
                video_text_embed = batch["video_text_embed"].to(device)

                # 일정 확률로 null embedding 사용
                if torch.rand(1).item() < 0.1:
                    audio_text_embed = audio_null_text_embed.to(device)
                    video_text_embed = video_null_text_embed.to(device)

                bsz = audio_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=device).long()

                noise_audio = torch.randn_like(audio_latents)
                noise_video = torch.randn_like(video_latents)

                noised_audio_latents = noise_scheduler.add_noise(audio_latents, noise_audio, timesteps)
                noised_video_latents = noise_scheduler.add_noise(video_latents, noise_video, timesteps)

                audio_encoder_hidden_states = audio_text_embed
                video_context = video_text_embed
                video_fps = video_latents.shape[2]

                # Forward
                audio_output, video_output = model(
                    audio_latents=noised_audio_latents,
                    audio_timestep=timesteps,
                    audio_encoder_hidden_states=audio_encoder_hidden_states,
                    video_latents=noised_video_latents,
                    video_timestep=timesteps,
                    video_context=video_context,
                    video_fps=video_fps
                )

                # Loss 계산
                loss_audio = F.mse_loss(audio_output, noise_audio)
                loss_video = F.mse_loss(video_output, noise_video)
                loss = loss_audio + loss_video

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                tepoch.set_postfix(loss_video=loss_video.item(), loss_audio=loss_audio.item())

                # wandb 로깅
                if accelerator.is_main_process and batch_idx % 100 == 0:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/loss_audio": loss_audio.item(),
                        "train/loss_video": loss_video.item(),
                        "epoch": epoch,
                        "step": global_step
                    })
                if accelerator.is_main_process and global_step % 1000 == 0:
                    checkpoint_path = f"mmg_checkpoints/global_step_{global_step}_batch_{batch_size}"
                    accelerator.save_state(checkpoint_path)

    print("Training Done.")


if __name__ == "__main__":
    main()
