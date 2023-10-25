from tqdm import tqdm

from utilities import *
from chonkfile import *
from diffusion_pipe import *

chonkpath = ""
chonk = Chonk.load(chonkpath)
pipe = Diffusion.load(chonk.model_cfg)
prompt = Prompt.load(chonk.prompt_cfg)

pipe.set_prompt(prompt)
rechonk = pipe.explore(1)
rechonk.save("foam")