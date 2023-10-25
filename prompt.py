from __future__ import annotations

import time
from collections import defaultdict
from typing import Union, Optional, Any

import torch
#from diffusers.utils import randn_tensor

StrOrTensorEmbed = Union[str, torch.FloatTensor]


class Offload:
    sequential = "sequential"
    model = "model"


class ModelNames:
    sd1_4 = "CompVis/stable-diffusion-v1-4"
    sd1_5 = "runwayml/stable-diffusion-v1-5"
    sd2_1 = "stabilityai/stable-diffusion-2-1"
    sdxl = "stabilityai/stable-diffusion-xl-base-1.0"


class Prompt:
    @staticmethod
    def load(cfg: dict, **kwargs: Optional[Any]) -> Prompt:
        """
        :param cfg: Dict of arguments [Prompt(**cfg)]
        :param kwargs: Overridden or additional arguments
        """
        for k, v in kwargs.items():
            cfg[k] = v

        return Prompt(**cfg)
        # return Prompt(cfg['prompt'], **{k: v for k, v in cfg.items() if k != 'prompt'})

    def reseed(self, seed: Optional[int] = None):
        """
        Set explicit new seed or use timestamp.
        :param seed: Optional explicit seed, uses timestamp if None.
        """
        self.seed = int(time.time()) if seed is None else seed
        self.seed_latents = None

    def __init__(self, prompt: StrOrTensorEmbed,
                 negative: Optional[StrOrTensorEmbed] = None,
                 prompt_2: Optional[str] = None,
                 negative_2: Optional[str] = None,
                 seed: Optional[int] = None,
                 height: int = 1024,
                 width: int = 1024,
                 steps: int = 25,
                 guidance_scale: float = 5.0,
                 guidance_rescale: float = 0.7,
                 eta: float = 0.0,

                 latents: Optional[torch.FloatTensor] = None,
                 seed_latents: Optional[torch.FloatTensor] = None,

                 crops_coords_top_left: tuple[int] = (0, 0),
                 negative_crops_coords_top_left: tuple[int] = (0, 0),
                 negative_original_size: tuple[int] = (1024, 1024),
                 negative_target_size: tuple[int] = (0, 0),

                 denoising_end: Optional[float] = None):

        # JSON serializes these as lists in chonkfile
        crops_coords_top_left = tuple(crops_coords_top_left)
        negative_crops_coords_top_left = tuple(negative_crops_coords_top_left)
        negative_original_size = tuple(negative_original_size)
        negative_target_size = tuple(negative_target_size)

        prompt_embed = negative_embed = None
        if type(prompt) == torch.FloatTensor:
            prompt_embed = prompt
            prompt = None

        if type(negative) == torch.FloatTensor:
            negative_embed = negative
            negative = None

        self.prompt = prompt
        self.negative = negative
        self.prompt_embed = prompt_embed
        self.negative_embed = negative_embed
        self.prompt_2 = prompt_2
        self.negative_2 = negative_2

        self.seed = None
        self.reseed(seed)
        self.seed_latents = seed_latents

        self.height = height
        self.width = width
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.eta = eta

        self.latents = latents

        self.crops_coords_top_left = crops_coords_top_left
        self.negative_original_size = negative_original_size
        self.negative_crops_coords_top_left = negative_crops_coords_top_left
        self.negative_target_size = negative_target_size

        self.denoising_end = denoising_end

    def save(self):
        return {
            "prompt"                        : self.prompt,
            "negative"                      : self.negative,
            "prompt_embed"                  : self.prompt_embed,
            "prompt_2"                      : self.prompt_2,
            "negative_2"                    : self.negative_2,
            "seed"                          : self.seed,
            # "seed_latents"                  : self.seed_latents,
            "height"                        : self.height,
            "width"                         : self.width,
            "steps"                         : self.steps,
            "guidance_scale"                : self.guidance_scale,
            "guidance_rescale"              : self.guidance_rescale,
            "eta"                           : self.eta,
            "crops_coords_top_left"         : self.crops_coords_top_left,
            "negative_original_size"        : self.negative_original_size,
            "negative_target_size"          : self.negative_target_size,
            "negative_crops_coords_top_left": self.negative_crops_coords_top_left,
            "denoising_end"                 : self.denoising_end
        }
