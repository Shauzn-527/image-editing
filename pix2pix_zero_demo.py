from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pix2pix_zero_prompts import create_sentences
from pix2pix_zero_runner import Pix2PixZeroConfig, load_pipeline, run_demo
from pix2pix_zero_utils import load_image_rgb, make_row_grid, save_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Pix2Pix-Zero")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of a cat",
        help="Prompt describing the input image (used for inversion & reference)",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source concept for the edit direction (e.g. cat, sunny, wooden)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target concept for the edit direction (e.g. dog, snowy, marble)",
    )
    parser.add_argument("--model", type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--xa-guidance", type=float, default=0.1, help="Cross-attention guidance step size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="outputs/pix2pix_zero_demo")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = load_pipeline(args.model, torch_dtype=torch.float16, device=device)

    cfg = Pix2PixZeroConfig(
        model_id=args.model,
        num_steps=int(args.steps),
        guidance_scale=float(args.guidance_scale),
        inversion_guidance_scale=1.0,
        cross_attention_guidance_amount=float(args.xa_guidance),
        eta=0.0,
    )

    input_image = load_image_rgb(args.image)
    source_sentences, target_sentences = create_sentences(args.source, args.target)

    reconstruction, edited = run_demo(
        pipe=pipe,
        input_image=input_image,
        prompt=args.prompt,
        source_sentences=source_sentences,
        target_sentences=target_sentences,
        cfg=cfg,
        seed=args.seed,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    save_image(reconstruction, outdir / "reference_reconstruction.png")
    save_image(edited, outdir / "edited.png")

    grid = make_row_grid([input_image, reconstruction, edited])
    save_image(grid, outdir / "grid_input_reference_edited.png")

    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
