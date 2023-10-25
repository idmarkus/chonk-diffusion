from tqdm import trange

from diffusion_pipe import *
from chonkfile import *

from prompts.prompts import prompt

import torch
torch.cuda.empty_cache()

pipe = Diffusion(ModelNames.sdxl, vae_tiling=True, vae_slicing=True, offload=Offload.model, fix_vae=True, scheduler="EulerDiscreteScheduler")#, scheduler="EulerDiscreteScheduler")#scheduler="UniPCMultistepScheduler")

pipe.set_prompt(prompt)

for i in trange(250):
    chonk = pipe.explore(1)
    chonk.save("outputs/output")
    prompt.reseed()