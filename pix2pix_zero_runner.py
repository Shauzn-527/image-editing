from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image

from diffusers import DDIMInverseScheduler, DDIMScheduler

from pipeline_stable_diffusion_pix2pix_zero import (
    Pix2PixZeroL2Loss,
    StableDiffusionPix2PixZeroPipeline,
    prepare_unet,
)


@dataclass(frozen=True)
class Pix2PixZeroConfig:
    model_id: str = "CompVis/stable-diffusion-v1-4"
    num_steps: int = 20
    guidance_scale: float = 7.5
    inversion_guidance_scale: float = 1.0
    cross_attention_guidance_amount: float = 0.1
    eta: float = 0.0


def load_pipeline(model_id: str, torch_dtype: torch.dtype, device: str | None = None) -> StableDiffusionPix2PixZeroPipeline:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionPix2PixZeroPipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    return pipe.to(device)


@torch.no_grad()
def compute_edit_direction(
    pipe: StableDiffusionPix2PixZeroPipeline,
    source_sentences: list[str],
    target_sentences: list[str],
) -> torch.Tensor:
    """根据 source/target 语义集合计算编辑方向。"""
    # ====== TODO 1 [必做]: 计算编辑方向 ======
    # 目标:
    # 1) 分别编码 source / target 句子集合
    # 2) 按均值差构造方向: Δc = E[c_target] - E[c_source]
    # 3) 输出 shape 为 [1, seq_len, hidden_dim]

    source_embeds = pipe.get_embeds(source_sentences)
    target_embeds = pipe.get_embeds(target_sentences)
    return (target_embeds.mean(0) - source_embeds.mean(0)).unsqueeze(0)
    # ====== END TODO 1 ======


@torch.no_grad()
def ddim_invert(
    pipe: StableDiffusionPix2PixZeroPipeline,
    image: Image.Image,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    inv_out = pipe.invert(
        prompt=prompt,
        image=image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        return_dict=True,
        output_type="pil",
    )
    return inv_out.latents


def _encode_prompt_for_cfg(
    pipe: StableDiffusionPix2PixZeroPipeline,
    prompt: str,
    negative_prompt: str | None,
    device: torch.device,
    guidance_scale: float,
):
    do_cfg = guidance_scale > 1.0
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
    )
    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    return prompt_embeds, do_cfg


@torch.no_grad()
def reference_denoise(
    pipe: StableDiffusionPix2PixZeroPipeline,
    latents: torch.Tensor,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    negative_prompt: str | None,
    eta: float,
    generator: torch.Generator | None = None,
) -> Image.Image:
    device = pipe._execution_device
    prompt_embeds, do_cfg = _encode_prompt_for_cfg(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        guidance_scale=guidance_scale,
    )

    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    latents_t = latents
    for t in timesteps:
        latent_model_input = torch.cat([latents_t] * 2) if do_cfg else latents_t
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
            cross_attention_kwargs={"timestep": t},
        ).sample

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents_t = pipe.scheduler.step(noise_pred, t, latents_t, **extra_step_kwargs).prev_sample

    image = pipe.vae.decode(latents_t / pipe.vae.config.scaling_factor, return_dict=False)[0]
    image = image.detach()
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def edit_denoise(
    pipe: StableDiffusionPix2PixZeroPipeline,
    latents_init: torch.Tensor,
    prompt: str,
    edit_direction: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
    negative_prompt: str | None,
    eta: float,
    cross_attention_guidance_amount: float,
    generator: torch.Generator | None = None,
) -> Image.Image:
    device = pipe._execution_device
    prompt_embeds, do_cfg = _encode_prompt_for_cfg(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        device=device,
        guidance_scale=guidance_scale,
    )

    prompt_embeds_edit = prompt_embeds.clone()
    if do_cfg:
        prompt_embeds_edit[1:2] = prompt_embeds_edit[1:2] + edit_direction.to(prompt_embeds_edit.device)
    else:
        prompt_embeds_edit = prompt_embeds_edit + edit_direction.to(prompt_embeds_edit.device)

    pipe.scheduler.set_timesteps(num_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    latents_t = latents_init
    for t in timesteps:
        latent_model_input = torch.cat([latents_t] * 2) if do_cfg else latents_t
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        x_in = latent_model_input.detach().clone()
        x_in.requires_grad_(True)
        opt = torch.optim.SGD([x_in], lr=cross_attention_guidance_amount)

        with torch.enable_grad():
            loss = Pix2PixZeroL2Loss()
            _ = pipe.unet(
                x_in,
                t,
                encoder_hidden_states=prompt_embeds_edit.detach(),
                cross_attention_kwargs={"timestep": t, "loss": loss},
            ).sample
            loss.loss.backward(retain_graph=False)
            opt.step()

        with torch.no_grad():
            noise_pred = pipe.unet(
                x_in.detach(),
                t,
                encoder_hidden_states=prompt_embeds_edit,
                cross_attention_kwargs={"timestep": None},
            ).sample

        latents_t = x_in.detach().chunk(2)[0] if do_cfg else x_in.detach()

        if do_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents_t = pipe.scheduler.step(noise_pred, t, latents_t, **extra_step_kwargs).prev_sample

    latents_t = latents_t.detach()
    with torch.no_grad():
        image = pipe.vae.decode(latents_t / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = image.detach()
        return pipe.image_processor.postprocess(image, output_type="pil")[0]


def run_demo(
    *,
    pipe: StableDiffusionPix2PixZeroPipeline,
    input_image: Image.Image,
    prompt: str,
    source_sentences: list[str],
    target_sentences: list[str],
    cfg: Pix2PixZeroConfig,
    seed: int | None = 0,
) -> tuple[Image.Image, Image.Image]:
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe._execution_device)
        generator.manual_seed(int(seed))

    edit_direction = compute_edit_direction(pipe, source_sentences, target_sentences)

    inverted_latents = ddim_invert(
        pipe,
        image=input_image,
        prompt=prompt,
        num_steps=cfg.num_steps,
        guidance_scale=cfg.inversion_guidance_scale,
        generator=generator,
    )

    # Install Pix2Pix-Zero attention processors exactly once:
    # reference pass caches maps, edit pass consumes them.
    pipe.unet = prepare_unet(pipe.unet)

    reconstruction = reference_denoise(
        pipe,
        latents=inverted_latents,
        prompt=prompt,
        num_steps=cfg.num_steps,
        guidance_scale=cfg.guidance_scale,
        negative_prompt=prompt,
        eta=cfg.eta,
        generator=generator,
    )

    edited = edit_denoise(
        pipe,
        latents_init=inverted_latents,
        prompt=prompt,
        edit_direction=edit_direction,
        num_steps=cfg.num_steps,
        guidance_scale=cfg.guidance_scale,
        negative_prompt=prompt,
        eta=cfg.eta,
        cross_attention_guidance_amount=cfg.cross_attention_guidance_amount,
        generator=generator,
    )

    return reconstruction, edited
