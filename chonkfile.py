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
        return datetime.now().strftime("%y%m%d_%H%M%S") + ".chonk.png"

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

        # Extract prompt fields to be more visible in exif tools
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

        latent = pickle.dumps(self.latent)
        latent = codecs.encode(latent, "base64").decode()

        if self.seed_latents is not None:
            seed_latents = pickle.dumps(self.seed_latents)
            seed_latents = codecs.encode(seed_latents, "base64").decode()
            info.add_text("seed_latents", seed_latents, zip=True)

        info.add_text("model_cfg", model_cfg, zip=True)
        info.add_text("prompt_cfg", prompt_cfg, zip=True)
        info.add_text("latent", latent, zip=True)


        # # Pickle data into bytes
        # meta = {'latent': self.latent, 'cfg': self.cfg}
        # byte = pickle.dumps(meta)
        #
        # # Compress bytes
        # if compress:
        #     # byte = lzma.compress(byte, preset=9)
        #     byte = bz2.compress(byte)
        #
        # # Encode bytes as base64 string
        # text = codecs.encode(byte, "base64").decode()
        #
        # # Add meta field
        # info = PngInfo()
        # if compress:
        #     info.add_text("CHOMP", text, zip=True)
        # else:
        #     info.add_text("CHONK", text, zip=True)

        # Save
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

            latent = codecs.decode(info['latent'].encode(), "base64")
            latent = pickle.loads(latent)

            if "seed_latents" in info:
                seed_latents = codecs.decode(info['seed_latents'].encode(), "base64")
                seed_latents = pickle.loads(seed_latents)
                #print("latent == seed_latents: ", info['latent'] == info['seed_latents'])

            img = Image.fromarray(np.array(f))

            # if 'fp16' not in model_cfg.keys():
            #     model_cfg['fp16'] = False
            # dtype = torch.float16 if model_cfg['fp16'] else torch.float32
            # latent = torch.tensor(latent, dtype=dtype)

            return Chonk(img, latent, model_cfg, prompt_cfg, seed_latents=seed_latents)








def save_chonk(img: Image, data: dict, path: Optional[Pathlike] = None, compress: bool = True, unlink: bool = False):
    path = Path(path)
    if path is None:
        path = Path(datetime.now().strftime("%y%m%d_%H%M%S") + ".chonk.png")
    elif not path.suffixes:  # Path is output directory
        path.mkdir(parents=True, exist_ok=True)
        path = path / (datetime.now().strftime("%y%m%d_%H%M%S") + ".chonk.png")
    else:  # Path is output filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.parent / path.stem / ".chonk.png"

    if unlink:
        Path(path).unlink(missing_ok=True)

    # Pickle data into bytes
    byte = pickle.dumps(data)

    # Compress bytes
    if compress:
        # byte = lzma.compress(byte, preset=9)
        byte = bz2.compress(byte)

    # Encode bytes as base64 string
    text = codecs.encode(byte, "base64").decode()

    # Add meta field
    meta = PngInfo()
    if compress:
        meta.add_text("CHOMP", text, zip=True)
    else:
        meta.add_text("CHONK", text, zip=True)

    # Save
    img.save(path, pnginfo=meta)
    return path


def load_chonk(path: Pathlike, compression="bz2") -> dict | None:

    with Image.open(str(path), mode='r') as img:

        try:
            meta = img.text

            if "CHONK" in meta:
                byte = codecs.decode(meta['CHONK'].encode(), "base64")
            elif "CHOMP" in meta:
                byte = codecs.decode(meta['CHOMP'].encode(), "base64")
                if "bz2" in compression:
                    byte = bz2.decompress(byte)
                if "lzma" in compression:
                    byte = lzma.decompress(byte)
            ret = pickle.loads(byte)
            ret['image'] = Image.fromarray(np.array(img))
            return ret
        except Exception as e:
            print(e)
            return None
            # raise InvalidPath(f"CHONK iTEXt chunk not found in png file: {path}")


if __name__ == "__main__":
    import torch
    from datetime import datetime


    def generate_inputs():
        sample = torch.randn(2, 4, 64, 64).half()
        timestep = torch.rand(1).half() * 999
        encoder_hidden_states = torch.randn(2, 77, 768).half()
        return sample, timestep, encoder_hidden_states


    torch_data = generate_inputs()
    manifest = {
        "name": "chonk_test",
        "time": str(datetime.now()),
        "data": torch_data
    }

    testfile = Image.open("testout_xl.png")

    chonk_p = save_chonk(testfile, manifest, "./chonks", compress=True, unlink=True)

    data = load_chonk(chonk_p)
    if data is None:
        print("some error")
    else:
        print(data['time'])
