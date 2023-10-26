from __future__ import annotations

from collections import defaultdict
from functools import singledispatch

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import pickle
import json
import codecs
import lzma
import bz2
from datetime import datetime

from utilities import *

import torch



class Chonk:


    def __init__(self, img: Imagelike, latent: Tensorlike, model_cfg: dict, prompt_cfg: dict, seed_latents: Optional[Tensorlike] = None):
        """
        Construct a Chonk for saving diffused image with embedded metadata
        :param img: Generated image
        :param latent: Latent image was generated from
        """
        self.path = None
        self.img = img
        self.latent = latent
        self.seed_latents = seed_latents
        self.model_cfg = model_cfg  # defaultdict(lambda: None, **model_cfg)
        self.prompt_cfg = prompt_cfg  # defaultdict(lambda: None, **prompt_cfg)

    @staticmethod
    def __timestamp_filename():
        return timestamp() + ".chonk.png"

    @staticmethod
    def __disambiguate_path_arg(path: Optional[Pathlike], exist_ok=True):
        path = Path(path)

        if path is None:
            path = Path(Chonk.__timestamp_filename())

        # Path is directory
        elif not path.suffixes:
            path.mkdir(parents=True, exist_ok=True)
            path = path / (Chonk.__timestamp_filename())

        # Path is filename
        else:
            # Unlink / mkdir
            if path.exists():
                if exist_ok:
                    path.unlink()
                else:
                    raise InvalidPath(f"Path exists and not exist_ok: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
            # path = path.parent / path.stem / ".chonk.png"
        return path

    @staticmethod
    def encode_tensor(tensor: torch.Tensor) -> str:
        enc = pickle.dumps(tensor)
        return codecs.encode(enc, "base64").decode()

    @staticmethod
    def decode_tensor(data: str) -> torch.Tensor:
        dec = codecs.decode(data.encode(), "base64")
        return pickle.loads(dec)

    def save(self, path: Optional[Pathlike], exist_ok=True, compress=True) -> Path:
        """
        Save Chonk to file.
        :param path: Exact filename or parent directory to use with generated name
        :param mkdir: Make necessary parent directories for path
        :param exist_ok: Overwrite existing file if present
        :param compress: Compress metadata and latents in written file
        """

        path = self.__disambiguate_path_arg(path, exist_ok=exist_ok)
        info = PngInfo()

        # Extract prompt fields to be visible in exif tools
        if self.prompt_cfg['prompt'] is not None:
            info.add_text("prompt", self.prompt_cfg['prompt'], zip=True)
            del self.prompt_cfg['prompt']
        if self.prompt_cfg['prompt_2'] is not None:
            info.add_text("prompt_2", self.prompt_cfg['prompt_2'], zip=True)
            del self.prompt_cfg['prompt_2']
        if self.prompt_cfg['negative'] is not None:
            info.add_text("negative", self.prompt_cfg['negative'], zip=True)
            del self.prompt_cfg['negative']
        if self.prompt_cfg['negative_2'] is not None:
            info.add_text("negative_2", self.prompt_cfg['negative_2'], zip=True)
            del self.prompt_cfg['negative_2']

        model_cfg = json.dumps({k: v for k, v in self.model_cfg.items() if v})
        prompt_cfg = json.dumps({k: v for k, v in self.prompt_cfg.items() if v})
        info.add_text("model_cfg", model_cfg, zip=True)
        info.add_text("prompt_cfg", prompt_cfg, zip=True)

        latent = self.encode_tensor(self.latent)
        info.add_text("latent", latent, zip=True)

        # latent = pickle.dumps(self.latent)
        # latent = codecs.encode(latent, "base64").decode()

        if self.seed_latents is not None:
            seed_latents = self.encode_tensor(self.seed_latents)
            info.add_text("seed_latents", seed_latents, zip=True)

        # Write file
        self.img.save(path, pnginfo=info)
        return path

    @staticmethod
    def load(path: Pathlike) -> Chonk:
        path = Path(path)
        if not path.exists():
            raise InvalidPath(f"File does not exist: {path}")
        elif not path.is_file():
            raise InvalidPath(f"Path is not a file: {path}")

        with Image.open(path, mode='r') as f:
            try:
                info = f.text
            except Exception as e:
                raise InvalidPath(f"Chonkfile is missing any text metadata: {path}")

            if "model_cfg" not in info:
                raise InvalidPath(f"Chonkfile metadata is missing 'model_cfg': {path}")
            if "prompt_cfg" not in info:
                raise InvalidPath(f"Chonkfile metadata is missing 'prompt_cfg': {path}")
            if "latent" not in info:
                raise InvalidPath(f"Chonkfile metadata is missing 'latent': {path}")

            model_cfg = json.loads(info['model_cfg'])
            prompt_cfg = json.loads(info['prompt_cfg'])

            if "prompt" in info:
                prompt_cfg["prompt"] = info["prompt"]
            if "prompt_2" in info:
                prompt_cfg["prompt_2"] = info["prompt_2"]
            if "negative" in info:
                prompt_cfg["negative"] = info["negative"]
            if "negative_2" in info:
                prompt_cfg["negative_2"] = info["negative_2"]

            latent = Chonk.decode_tensor(info['latent'])
            # latent = codecs.decode(info['latent'].encode(), "base64")
            # latent = pickle.loads(latent)

            if "seed_latents" in info:
                seed_latents = Chonk.decode_tensor(info['seed_latents'])
                # seed_latents = codecs.decode(info['seed_latents'].encode(), "base64")
                # seed_latents = pickle.loads(seed_latents)

            img = Image.fromarray(np.array(f))

            return Chonk(img, latent, model_cfg, prompt_cfg, seed_latents=seed_latents)







