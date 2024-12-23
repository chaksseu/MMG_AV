import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from diffusers import PNDMScheduler, UNet2DConditionModel, DDPMScheduler
from lvdm.models.utils_diffusion import timestep_embedding
from lvdm.modules.networks.openaimodel3d import (
    ResBlock, SpatialTransformer, TemporalTransformer, CrossModalTransformer,
    Downsample, Upsample, TimestepBlock
)
from einops import rearrange
from omegaconf import OmegaConf
from utils.utils import instantiate_from_config
from accelerate import Accelerator
from tqdm import tqdm
import wandb


def normalize_data(data, data_min, data_max):
    # Normalize to [-1, 1]
    return 2 * (data - data_min) / (data_max - data_min) - 1

def denormalize_data(data, data_min, data_max):
    # Reverse normalization for reconstruction
    return (data + 1) / 2 * (data_max - data_min) + data_min


class LatentsDataset(torch.utils.data.Dataset):
    """
    A dataset class that loads video and audio latents, along with text embeddings.
    """
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

        # 파일 경로 가져오기
        video_file = self.data_info.iloc[idx]["Video"]
        audio_file = self.data_info.iloc[idx]["Audio"]
        video_text_embed_file = self.data_info.iloc[idx]["Video_text_embed"]
        audio_text_embed_file = self.data_info.iloc[idx]["Audio_text_embed"]

        # map_location을 현재 GPU로 설정
        map_location = torch.device(f"cuda:{torch.cuda.current_device()}")

        # torch.load에 map_location 추가
        video_latent = torch.load(os.path.join(self.video_latents_dir, video_file), map_location=map_location)
        audio_latent = torch.load(os.path.join(self.audio_latents_dir, audio_file), map_location=map_location)
        video_text_embed = torch.load(os.path.join(self.video_text_embeds_dir, video_text_embed_file), map_location=map_location)
        audio_text_embed = torch.load(os.path.join(self.audio_text_embeds_dir, audio_text_embed_file), map_location=map_location)

        # 샘플 반환
        sample = {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "video_text_embed": video_text_embed,
            "audio_text_embed": audio_text_embed
        }
        
        return sample


def get_dataloader(root_dir, csv_file, batch_size=4, shuffle=True, num_workers=0):
    dataset = LatentsDataset(root_dir=root_dir, csv_file=csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class CrossModalCoupledUNet(nn.Module):
    """
    A coupled UNet model that fuses features from audio and video UNets via cross-modal transformers.
    Audio and video UNets are frozen, and only cross-modal layers are trainable.
    """
    def __init__(self, audio_unet, video_unet, cross_modal_config):
        super(CrossModalCoupledUNet, self).__init__()
        # Freeze audio_unet
        self.audio_unet = audio_unet
        for param in self.audio_unet.parameters():
            param.requires_grad = False

        # Freeze video_unet
        self.video_unet = video_unet
        for param in self.video_unet.parameters():
            param.requires_grad = False

        self.audio_cmt = nn.ModuleList()
        self.video_cmt = nn.ModuleList()
        layer_channels = cross_modal_config['layer_channels']


        for channel in layer_channels:
            d_head = cross_modal_config.get('d_head', 64)
            n_heads = channel // d_head

            audio_transformer = CrossModalTransformer(
                in_channels=channel, n_heads=n_heads, d_head=d_head, depth=1, context_dim=channel,
                use_linear=True, use_checkpoint=True, disable_self_attn=False, img_cross_attention=False
            )
            self.initialize_cross_modal_transformer(audio_transformer)
            self.audio_cmt.append(audio_transformer)

            video_transformer = CrossModalTransformer(
                in_channels=channel, n_heads=n_heads, d_head=d_head, depth=1, context_dim=channel,
                use_linear=True, use_checkpoint=True, disable_self_attn=False, img_cross_attention=False
            )
            self.initialize_cross_modal_transformer(video_transformer)
            self.video_cmt.append(video_transformer)


    def initialize_linear_layers(self):
        # Initialize linear layers
        for linear in self.audio_adaptation_in:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for linear in self.video_adaptation_in:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)

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
        # Process one down_block for the audio UNet
        if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
            for resnet, attn in zip(down_block.resnets, down_block.attentions):
                hidden_states = resnet(hidden_states, emb)
                hidden_states = attn(
                    hidden_states, encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False
                )[0]
                output_states += (hidden_states,)
        else:
            for resnet in down_block.resnets:
                hidden_states = resnet(hidden_states, emb)
                output_states += (hidden_states,)
        return hidden_states, output_states

    def audio_mid_blocks(self, audio_unet, hidden_states, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs):
        # Process mid blocks for the audio UNet
        if hasattr(audio_unet.mid_block, "has_cross_attention") and audio_unet.mid_block.has_cross_attention:
            hidden_states = audio_unet.mid_block.resnets[0](hidden_states, emb)
            for resnet, attn in zip(audio_unet.mid_block.resnets[1:], audio_unet.mid_block.attentions):
                hidden_states = attn(
                    hidden_states, encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                hidden_states = resnet(hidden_states, emb)
        else:
            for resnet in audio_unet.mid_block.resnets:
                hidden_states = resnet(hidden_states, emb)
        return hidden_states

    def audio_up_blocks(self, up_block, hidden_states, encoder_hidden_states, emb, attention_mask, cross_attention_kwargs, res_hidden_states_tuple):
        # Process one up_block for the audio UNet
        if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
            for resnet, attn in zip(up_block.resnets, up_block.attentions):
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
                hidden_states = resnet(hidden_states, emb)
                hidden_states = attn(
                    hidden_states, encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask, cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
        else:
            for resnet in up_block.resnets:
                res_hidden = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden], dim=1)
                hidden_states = resnet(hidden_states, emb)
        return hidden_states

    def process_video_sublayer(self, sublayer, h, video_emb, video_context, batch_size):
        # Process a single layer of the video UNet block (down or up)
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
        # Process a video down_block
        for sublayer in video_unet.input_blocks[block_idx]:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        if block_idx == 0 and video_unet.addition_attention:
            h = video_unet.init_attn(h, video_emb, context=video_context, batch_size=batch_size)
        hs.append(h)
        return h, hs

    def video_up_block(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
        # Process a video up_block
        h = torch.cat([h, hs.pop()], dim=1)
        for sublayer in video_unet.output_blocks[block_idx]:
            if isinstance(sublayer, Upsample):
                break
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, batch_size)
        return h, hs

    def video_upsample(self, block_idx, video_unet, h, hs, video_emb, video_context, batch_size):
        # Process the Upsample layer of a video up_block
        h = self.process_video_sublayer(video_unet.output_blocks[block_idx][-1], h, video_emb, video_context, batch_size)
        return h, hs

    def linear_cmt(self, audio_hidden_states, h, index):
        
        # Cross-modal transformer step
        b_a, _, _, t_a = audio_hidden_states.shape
        b_v, _, h_v, w_v = h.shape
        k = int(b_v / (b_a * t_a))

        cross_video_latent_token = rearrange(h, '(b k) c h w -> b c (k h w)', k=k)
        cross_audio_latent_token = rearrange(audio_hidden_states, 'b c f t -> (b t) c f')

        condition_cross_audio_latent_token = cross_audio_latent_token
        condition_cross_video_latent_token = cross_video_latent_token

        # Cross-modal attention
        cross_video_latent_token = self.video_cmt[index](cross_video_latent_token, condition_cross_audio_latent_token)
        cross_audio_latent_token = self.audio_cmt[index](cross_audio_latent_token, condition_cross_video_latent_token)

        h = rearrange(cross_video_latent_token, 'b c (k h w) -> (b k) c h w', h=h_v, w=w_v, k=k)
        audio_hidden_states = rearrange(cross_audio_latent_token, '(b t) c f -> b c f t', t=t_a)
        
        return audio_hidden_states, h

    def forward(self, audio_latents, audio_timestep, audio_encoder_hidden_states,
                video_latents, video_timestep, video_context=None, video_fps=8,
                audio_attention_mask=None, audio_cross_attention_kwargs=None):
        # ---- Prepare Audio Branch ----
        audio_timesteps = audio_timestep
        if not torch.is_tensor(audio_timesteps):
            dtype = torch.int64 if isinstance(audio_timestep, int) else torch.float32
            audio_timesteps = torch.tensor([audio_timestep], dtype=dtype, device=audio_latents.device)
        elif audio_timesteps.dim() == 0:
            audio_timesteps = audio_timesteps[None].to(audio_latents.device)
        audio_timesteps = audio_timesteps.expand(audio_latents.shape[0])

        audio_t_emb = self.audio_unet.time_proj(audio_timesteps).to(dtype=audio_latents.dtype)
        audio_emb = self.audio_unet.time_embedding(audio_t_emb)
        if self.audio_unet.time_embed_act is not None:
            audio_emb = self.audio_unet.time_embed_act(audio_emb)
        if self.audio_unet.encoder_hid_proj is not None:
            audio_encoder_hidden_states = self.audio_unet.encoder_hid_proj(audio_encoder_hidden_states)
        audio_hidden_states = self.audio_unet.conv_in(audio_latents)
        audio_down_block_res_samples = (audio_hidden_states,)

        # ---- Prepare Video Branch ----
        video_emb = self.video_unet.time_embed(timestep_embedding(video_timestep, self.video_unet.model_channels))
        if self.video_unet.fps_cond:
            video_fps_tensor = torch.full_like(video_timestep, video_fps) if isinstance(video_fps, int) else video_fps
            video_emb += self.video_unet.fps_embedding(timestep_embedding(video_fps_tensor, self.video_unet.model_channels))
        b, _, t, _, _ = video_latents.shape
        video_context = video_context.repeat_interleave(repeats=t, dim=0) if video_context is not None else None
        video_emb = video_emb.repeat_interleave(repeats=t, dim=0)
        h = rearrange(video_latents, 'b c t h w -> (b t) c h w').type(self.video_unet.dtype)
        video_emb = video_emb.to(h.dtype)
        hs = []

        # ---- Audio & Video Down Blocks ----
        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            self.audio_unet.down_blocks[0],
            audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs,
            audio_down_block_res_samples
        )
        h, hs = self.video_down_block(0, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(1, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(2, self.video_unet, h, video_emb, video_context, b, hs)

        # ---- Cross-Modal Transformer #0 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 0)

        # Audio downsample & video down_block
        audio_hidden_states = self.audio_unet.down_blocks[0].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        h, hs = self.video_down_block(3, self.video_unet, h, video_emb, video_context, b, hs)

        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            self.audio_unet.down_blocks[1],
            audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs,
            audio_down_block_res_samples
        )
        h, hs = self.video_down_block(4, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(5, self.video_unet, h, video_emb, video_context, b, hs)

        # ---- Cross-Modal Transformer #1 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 1)

        # Audio downsample & video down_block
        audio_hidden_states = self.audio_unet.down_blocks[1].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        h, hs = self.video_down_block(6, self.video_unet, h, video_emb, video_context, b, hs)

        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            self.audio_unet.down_blocks[2],
            audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs,
            audio_down_block_res_samples
        )
        h, hs = self.video_down_block(7, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(8, self.video_unet, h, video_emb, video_context, b, hs)

        # ---- Cross-Modal Transformer #2 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 2)

        # Audio downsample & video down_block
        audio_hidden_states = self.audio_unet.down_blocks[2].downsamplers[0](audio_hidden_states)
        audio_down_block_res_samples += (audio_hidden_states,)
        h, hs = self.video_down_block(9, self.video_unet, h, video_emb, video_context, b, hs)

        audio_hidden_states, audio_down_block_res_samples = self.audio_down_blocks(
            self.audio_unet.down_blocks[3],
            audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs,
            audio_down_block_res_samples
        )
        h, hs = self.video_down_block(10, self.video_unet, h, video_emb, video_context, b, hs)
        h, hs = self.video_down_block(11, self.video_unet, h, video_emb, video_context, b, hs)

        # ---- Mid Blocks ----
        audio_hidden_states = self.audio_mid_blocks(
            self.audio_unet, audio_hidden_states, audio_emb,
            audio_encoder_hidden_states, audio_attention_mask, audio_cross_attention_kwargs
        )
        for sublayer in self.video_unet.middle_block:
            h = self.process_video_sublayer(sublayer, h, video_emb, video_context, b)

        # ---- Up Blocks ----
        # Audio up_block #0
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[0].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[0].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[0], audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs, audio_res_samples
        )
        h, hs = self.video_up_block(0, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(1, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(2, self.video_unet, h, hs, video_emb, video_context, b)

        # ---- Cross-Modal Transformer #3 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 3)

        audio_hidden_states = self.audio_unet.up_blocks[0].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(2, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio up_block #1
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[1].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[1].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[1], audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs, audio_res_samples
        )
        h, hs = self.video_up_block(3, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(4, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(5, self.video_unet, h, hs, video_emb, video_context, b)

        # ---- Cross-Modal Transformer #4 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 4)

        audio_hidden_states = self.audio_unet.up_blocks[1].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(5, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio up_block #2
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[2].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[2].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[2], audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs, audio_res_samples
        )
        h, hs = self.video_up_block(6, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(7, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(8, self.video_unet, h, hs, video_emb, video_context, b)

        # ---- Cross-Modal Transformer #5 ----
        audio_hidden_states, h = self.linear_cmt(audio_hidden_states, h, 5)

        audio_hidden_states = self.audio_unet.up_blocks[2].upsamplers[0](audio_hidden_states)
        h, hs = self.video_upsample(8, self.video_unet, h, hs, video_emb, video_context, b)

        # Audio up_block #3
        audio_res_samples = audio_down_block_res_samples[-len(self.audio_unet.up_blocks[3].resnets):]
        audio_down_block_res_samples = audio_down_block_res_samples[:-len(self.audio_unet.up_blocks[3].resnets)]
        audio_hidden_states = self.audio_up_blocks(
            self.audio_unet.up_blocks[3], audio_hidden_states, audio_encoder_hidden_states, audio_emb,
            audio_attention_mask, audio_cross_attention_kwargs, audio_res_samples
        )
        h, hs = self.video_up_block(9, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(10, self.video_unet, h, hs, video_emb, video_context, b)
        h, hs = self.video_up_block(11, self.video_unet, h, hs, video_emb, video_context, b)

        # ---- Output Layers ----
        if self.audio_unet.conv_norm_out is not None:
            audio_hidden_states = self.audio_unet.conv_norm_out(audio_hidden_states)
            if self.audio_unet.conv_act is not None:
                audio_hidden_states = self.audio_unet.conv_act(audio_hidden_states)
        audio_hidden_states = self.audio_unet.conv_out(audio_hidden_states)

        for sublayer in self.video_unet.out:
            h = sublayer(h)
        h = rearrange(h, '(b f) c h w -> b c f h w', b=b)

        return audio_hidden_states, h


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Modal Coupled UNet Training Script")

    # 데이터 관련 인자
    parser.add_argument('--processed_data_dir', type=str, default="latents_data_32s_40frames_vggsound_sparse_new_normalization",
                        help='데이터가 저장된 루트 디렉토리 경로')
    parser.add_argument('--csv_file', type=str, default=None,
                        help='데이터셋 정보가 담긴 CSV 파일 경로 (root_dir이 설정되지 않은 경우 필수)')

    # 모델 관련 인자
    parser.add_argument('--audio_model_name', type=str, default="auffusion/auffusion-full",
                        help='오디오 모델의 이름 또는 경로')
    parser.add_argument('--videocrafter_ckpt', type=str, default='scripts/evaluation/model.ckpt',
                        help='비디오 Crafter 모델의 체크포인트 경로')
    parser.add_argument('--videocrafter_config', type=str, default='configs/inference_t2v_512_v2.0.yaml',
                        help='비디오 Crafter 모델의 설정 파일 경로')

    # 학습 관련 인자
    parser.add_argument('--num_epochs', type=int, default=1000, help='학습할 에폭 수')
    parser.add_argument('--num_gpus', type=int, default=1, help='학습할 에폭 수')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 사이즈')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32, help='그래디언트 누적 스텝 수')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='학습률')
    parser.add_argument('--video_fps', type=float, default=12.5, help='비디오 FPS')
    parser.add_argument('--date', type=str, default="1223_MMG", help='체크포인트 및 wandb 이름에 사용될 날짜 또는 식별자')

    # 손실 가중치
    parser.add_argument('--audio_loss_weight', type=float, default=1.0, help='오디오 손실 가중치')
    parser.add_argument('--video_loss_weight', type=float, default=4.0, help='비디오 손실 가중치')

    # 기타 설정
    parser.add_argument('--dataset_name', type=str, default="vggsound_sparse", help='데이터셋 이름')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader의 num_workers')
    parser.add_argument('--wandb_project', type=str, default="MMG_auffusion_videocrafter",
                        help='WandB 프로젝트 이름')
    parser.add_argument('--checkpoint_dir', type=str, default="MMG_CHECKPOINTS_1223",
                        help='체크포인트가 저장될 디렉토리')

    args = parser.parse_args()
    
    # CSV 파일 경로 설정
    if args.csv_file is None:
        args.csv_file = os.path.join(args.root_dir, "dataset_info.csv")
    
    return args



def main():
    args = parse_args()
    # 데이터 로더 설정
    dataloader = get_dataloader(processed_data_dir=args.processed_data_dir, csv_file=args.csv_file,
                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    accelerator = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device


    # Audio UNet 로드 및 평가 모드 설정
    audio_unet = UNet2DConditionModel.from_pretrained(args.audio_model_name, subfolder="unet")
    audio_unet.eval()

    # Video UNet 로드
    video_config = OmegaConf.load(args.videocrafter_config)
    video_model = instantiate_from_config(video_config.model)
    state_dict = torch.load(args.videocrafter_ckpt)['state_dict']
    video_model.load_state_dict(state_dict, strict=True)
    video_model.to(device)
    video_unet = video_model.model.diffusion_model.eval()

    # Cross-modal config 설정
    cross_modal_config = {
        'layer_channels': args.layer_channels,
        'd_head': args.d_head,
        'device': device
    }

    # 모델 초기화
    model = CrossModalCoupledUNet(audio_unet, video_unet, cross_modal_config)
    noise_scheduler = DDPMScheduler.from_pretrained(args.audio_model_name, subfolder="scheduler")
    
    # 파라미터 수 출력
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))
    if accelerator.is_main_process:
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")

    # 옵티마이저 설정
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
    
    # 모델, 옵티마이저, 데이터로더 준비
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model.train()

    global_step = 0
    losses = []
    losses_video = []
    losses_audio = []

    # WandB 설정
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, 
                   name=f"{args.date}_lr_{args.learning_rate}_batch_{args.batch_size * args.num_gpus * args.gradient_accumulation_steps}_a_w_{args.audio_loss_weight}_v_w_{args.video_loss_weight}_{args.dataset_name}")
    else:
        os.environ["WANDB_MODE"] = "offline"


    video_fps=args.video_fps
    video_fps = torch.tensor([video_fps] * batch_size).long()



    if accelerator.is_main_process:
        wandb.init(project="MMG_auffusion_videocrafter", name=f"{date}_lr_{lr}_batch_{full_batch_size}_a_w_{audio_loss_weight}_v_w_{video_loss_weight}_{dataset_name}")
    else:
        os.environ["WANDB_MODE"] = "offline"

    for epoch in range(args.num_epochs):
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}", unit="batch") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                audio_latents = batch["audio_latent"]
                video_latents = batch["video_latent"]
                audio_text_embed = batch["audio_text_embed"]
                video_text_embed = batch["video_text_embed"]

                # Null conditioning with 0.1 probability
                audio_null_text_embed = torch.zeros(1, 1, audio_text_embed.shape[-1], device=audio_text_embed.device)
                video_null_text_embed = torch.zeros(1, 1, video_text_embed.shape[-1], device=video_text_embed.device)
                mask = (torch.rand(batch_size, 1, 1, device=audio_text_embed.device) < 0.1)
                audio_text_embed = torch.where(mask, audio_null_text_embed, audio_text_embed)
                video_text_embed = torch.where(mask, video_null_text_embed, video_text_embed)

                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (batch_size,), device=device).long()

                noise_audio = torch.randn_like(audio_latents).to(device)
                noise_video = torch.randn_like(video_latents).to(device)
                noised_audio_latents = noise_scheduler.add_noise(audio_latents, noise_audio, timesteps)
                noised_video_latents = video_model.q_sample(x_start=video_latents, t=timesteps, noise=noise_video)

                # Forward pass
                audio_output, video_output = model(
                    audio_latents=noised_audio_latents,
                    audio_timestep=timesteps,
                    audio_encoder_hidden_states=audio_text_embed,
                    video_latents=noised_video_latents,
                    video_timestep=timesteps,
                    video_context=video_text_embed,
                    video_fps=video_fps
                )

                # Weighted loss
                loss_audio = args.audio_loss_weight * F.mse_loss(audio_output, noise_audio)
                loss_video = args.video_loss_weight * F.mse_loss(video_output, noise_video)
                loss = loss_audio + loss_video

                losses.append(loss.item())
                losses_audio.append(loss_audio.item())
                losses_video.append(loss_video.item())

                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if accelerator.is_main_process and (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    avg_losses = sum(losses) / len(losses)
                    avg_losses_video = sum(losses_video) / len(losses_video)
                    avg_losses_audio = sum(losses_audio) / len(losses_audio)
                    losses = []
                    losses_video = []
                    losses_audio = []

                    wandb.log({
                        "train/loss": avg_losses,
                        "train/loss_audio": avg_losses_audio,
                        "train/loss_video": avg_losses_video,
                        "epoch": epoch,
                        "step": global_step
                    })

            if accelerator.is_main_process and (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.date}_lr_{args.learning_rate}_batch_{args.batch_size * 4 * args.gradient_accumulation_steps}_epoch_{epoch+1}_{args.dataset_name}")
                os.makedirs(args.checkpoint_dir, exist_ok=True)
                accelerator.save_state(checkpoint_path)

    if accelerator.is_main_process:
        wandb.finish()
    print("Training Done.")


if __name__ == "__main__":
    main()
