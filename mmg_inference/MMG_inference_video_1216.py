import argparse
import datetime
import os
import time
import numpy as np
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from scripts.evaluation.funcs import load_model_checkpoint, load_prompts, save_videos
from utils.utils import instantiate_from_config
from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps
from lvdm.common import noise_like


def initialize_ddim_sampler(model, schedule="linear"):
    # Initialize DDIM sampler state dict
    return {
        "model": model,
        "ddpm_num_timesteps": model.num_timesteps,
        "schedule": schedule,
        "counter": 0,
        "use_scale": model.use_scale
    }


def register_buffer(sampler_state, name, attr):
    # Store tensors on GPU
    if isinstance(attr, torch.Tensor) and attr.device != torch.device("cuda"):
        attr = attr.to(torch.device("cuda"))
    sampler_state[name] = attr


def make_ddim_schedule(sampler_state, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
    # Prepare DDIM schedules and related buffers
    model = sampler_state["model"]
    ddpm_num_timesteps = sampler_state["ddpm_num_timesteps"]
    use_scale = sampler_state["use_scale"]

    ddim_timesteps = make_ddim_timesteps(
        ddim_discr_method=ddim_discretize,
        num_ddim_timesteps=ddim_num_steps,
        num_ddpm_timesteps=ddpm_num_timesteps,
        verbose=verbose
    )

    alphas_cumprod = model.alphas_cumprod
    to_torch = lambda x: x.clone().detach().float().to(model.device)

    register_buffer(sampler_state, 'betas', to_torch(model.betas))
    register_buffer(sampler_state, 'alphas_cumprod', to_torch(alphas_cumprod))
    register_buffer(sampler_state, 'alphas_cumprod_prev', to_torch(model.alphas_cumprod_prev))

    if use_scale:
        register_buffer(sampler_state, 'scale_arr', to_torch(model.scale_arr))
        ddim_scale_arr = sampler_state['scale_arr'].cpu()[ddim_timesteps]
        register_buffer(sampler_state, 'ddim_scale_arr', ddim_scale_arr)
        ddim_scale_arr_prev = np.asarray([sampler_state['scale_arr'].cpu()[0]] +
                                         sampler_state['scale_arr'].cpu()[ddim_timesteps[:-1]].tolist())
        register_buffer(sampler_state, 'ddim_scale_arr_prev', ddim_scale_arr_prev)

    register_buffer(sampler_state, 'sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
    register_buffer(sampler_state, 'sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
    register_buffer(sampler_state, 'log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
    register_buffer(sampler_state, 'sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
    register_buffer(sampler_state, 'sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

    ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
        alphacums=alphas_cumprod.cpu(),
        ddim_timesteps=ddim_timesteps,
        eta=ddim_eta,
        verbose=verbose
    )
    register_buffer(sampler_state, 'ddim_sigmas', ddim_sigmas)
    register_buffer(sampler_state, 'ddim_alphas', ddim_alphas)
    register_buffer(sampler_state, 'ddim_alphas_prev', ddim_alphas_prev)
    register_buffer(sampler_state, 'ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))

    sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
        (1 - sampler_state['alphas_cumprod_prev']) / (1 - sampler_state['alphas_cumprod']) *
        (1 - sampler_state['alphas_cumprod'] / sampler_state['alphas_cumprod_prev'])
    )
    register_buffer(sampler_state, 'ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    sampler_state['ddim_timesteps'] = ddim_timesteps
    sampler_state['ddim_num_steps'] = ddim_num_steps


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed")
    parser.add_argument("--mode", default="base", type=str, help="inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default='scripts/evaluation/model.ckpt', help="checkpoint path")
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml", help="config path")
    parser.add_argument("--prompt_file", type=str, default="prompts/test_prompts.txt", help="prompt text file")
    parser.add_argument("--savedir", type=str, default="1216_output_000", help="output directory")
    parser.add_argument("--savefps", type=float, default=12.5, help="fps for saved video")
    parser.add_argument("--ddim_steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="DDIM eta")
    parser.add_argument("--bs", type=int, default=2, help="batch size")
    parser.add_argument("--height", type=int, default=256, help="image height")
    parser.add_argument("--width", type=int, default=256, help="image width")
    parser.add_argument("--frames", type=int, default=40, help="video frames")
    parser.add_argument("--fps", type=float, default=12.5, help="video fps")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=12.0, help="CFG scale")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None, help="temporal CFG scale")
    return parser


@torch.no_grad()
def run_inference(args, device):
    # Load model and config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config).cuda(device)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()

    # Check image size
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size must be multiples of 16!"
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    # Prepare output dir
    os.makedirs(args.savedir, exist_ok=True)

    # Load prompts
    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)

    start = time.time()
    n_rounds = (len(prompt_list) + args.bs - 1) // args.bs

    # Initialize sampler
    sampler_state = initialize_ddim_sampler(model)

    for idx in range(n_rounds):
        idx_s = idx * args.bs
        idx_e = min(idx_s + args.bs, len(prompt_list))
        batch_size = idx_e - idx_s
        filenames = prompt_list[idx_s:idx_e]
        noise_shape = [batch_size, channels, frames, h, w]
        fps_tensor = torch.tensor([args.fps] * batch_size).to(model.device).long()

        prompts = prompt_list[idx_s:idx_e]
        text_emb = model.get_learned_conditioning(prompts)

        # Conditioning
        if args.mode == 'base':
            cond = {"c_crossattn": [text_emb], "fps": fps_tensor}
        else:
            raise NotImplementedError

        ddim_steps = args.ddim_steps
        ddim_eta = args.ddim_eta
        cfg_scale = args.unconditional_guidance_scale
        uncond_type = model.uncond_type

        # Unconditional guidance
        if cfg_scale != 1.0:
            if uncond_type == "empty_seq":
                uc_emb = model.get_learned_conditioning(batch_size * [""])
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0]
                uc_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError("Unknown uncond_type")

            if hasattr(model, 'embedder'):
                uc_img = torch.zeros(noise_shape[0], 3, 224, 224).to(model.device)
                uc_img = model.get_image_embeds(uc_img)
                uc_emb = torch.cat([uc_emb, uc_img], dim=1)

            uc = {key: cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = None

        # Make DDIM schedule
        make_ddim_schedule(sampler_state, ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

        # Sampling
        shape = noise_shape
        model = sampler_state["model"]
        device = model.betas.device
        img = torch.randn(shape, device=device)
        timesteps = sampler_state['ddim_timesteps']
        total_steps = timesteps.shape[0]
        time_range = np.flip(timesteps)



        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            ts = torch.full((batch_size,), step, device=device, dtype=torch.long)




            # Apply model with CFG
            if uc is None or cfg_scale == 1.:
                e_t = model.apply_model(img, ts, cond)
            else:
                e_t = model.apply_model(img, ts, cond)
                e_t_uncond = model.apply_model(img, ts, uc)
                e_t = e_t_uncond + cfg_scale * (e_t - e_t_uncond)


            # Select parameters
            use_original_steps = False
            alphas = (model.alphas_cumprod if use_original_steps else sampler_state['ddim_alphas'])
            alphas_prev = (model.alphas_cumprod_prev if use_original_steps else sampler_state['ddim_alphas_prev'])
            sqrt_one_minus_alphas = (model.sqrt_one_minus_alphas_cumprod if use_original_steps
                                     else sampler_state['ddim_sqrt_one_minus_alphas'])
            sigmas = (sampler_state['ddim_sigmas_for_original_num_steps'] if use_original_steps
                      else sampler_state['ddim_sigmas'])

            is_video = (img.dim() == 5)
            size = (batch_size, 1, 1, 1, 1) if is_video else (batch_size, 1, 1, 1)
            a_t = torch.full(size, alphas[index], device=device)
            a_prev = torch.full(size, alphas_prev[index], device=device)
            sigma_t = torch.full(size, sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device)

            pred_x0 = (img - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
            noise = sigma_t * noise_like(img.shape, device, False)
            if sampler_state['use_scale']:
                scale_arr = (model.scale_arr if use_original_steps else sampler_state['ddim_scale_arr'])
                scale_t = torch.full(size, scale_arr[index], device=device)
                scale_arr_prev = (model.scale_arr_prev if use_original_steps
                                  else sampler_state['ddim_scale_arr_prev'])
                scale_t_prev = torch.full(size, scale_arr_prev[index], device=device)
                pred_x0 /= scale_t
                img = a_prev.sqrt() * scale_t_prev * pred_x0 + dir_xt + noise
            else:
                img = a_prev.sqrt() * pred_x0 + dir_xt + noise



        # Decode latent to pixel space
        batch_images = model.decode_first_stage_2DAE(img)
        batch_samples = batch_images.unsqueeze(1)
        save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Inference:", now)
    device = 7
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    run_inference(args, device)
