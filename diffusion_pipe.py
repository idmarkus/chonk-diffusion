from __future__ import annotations

import gc
import random
from collections import defaultdict
from datetime import datetime

from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from tqdm import trange

from chonkfile import Chonk
from utilities import *
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    # DPMSolverMultistepScheduler,
    # EulerDiscreteScheduler,
    AutoencoderKL
)
import torch

#from trace_unet import *

from prompt import *
import time


class DummyProgress:
    class DummyContext:
        def update(self):
            pass

    def __init__(self, *args, **kwargs):
        self.ctx = self.DummyContext()

    def __enter__(self):
        return self.ctx

    def __exit__(self, *args):
        pass

    def update(self):
        pass


class Diffusion:
    CFG = dict

    @staticmethod
    def load(cfg: dict, **kwargs: Optional[Any]) -> Diffusion:
        """
        :param cfg: Dict of keyword args -> Diffusion(**cfg)
        :param kwargs: Custom overrides or additional keyword arguments
        """
        # Key will be filtered in chonkfile if explicitly false
        if 'fp16' not in cfg.keys():
            cfg['fp16'] = False

        # Override and append explicit args
        for k, v in kwargs.items():
            cfg[k] = v

        return Diffusion(**cfg)
        # return Diffusion(cfg['model_name'], **{k: v for k, v in cfg.items() if k != 'model_name'})

    def list_schedulers(self) -> dict:
        """
        :return: dict[name:class] of all compatible schedulers of pipe
        """
        # Fixme: ? ? ?
        # pipe.scheduler.compatible returns a list of class objects
        # which have no __name__ unless instantiated I guess, so we
        # will just parse the string rep, which looks like:
        # <class 'diffusers.schedulers.scheduling_k_dpm_2_discrete.KDPM2DiscreteScheduler'>,
        fmt = lambda s: s.split('.')[-1][:-2]

        return {fmt(str(cls)): cls for cls in self.pipe.scheduler.compatibles}

    def set_scheduler(self, scheduler: Optional[str]):
        available = self.list_schedulers()
        if scheduler is None or scheduler not in available.keys():
            print(f"Warning: Scheduler {scheduler} is missing. Using default: 'PNDMScheduler'")
            scheduler = "PNDMScheduler"
        self.pipe.scheduler = available[scheduler].from_config(self.pipe.scheduler.config)
        self.cfg['scheduler'] = scheduler

    @torch.no_grad()
    def __init__(self, model_name: str,
                 fp16: bool = True,
                 offload: Optional[str] = None,
                 vae_tiling: bool = False,
                 vae_slicing: bool = False,
                 fix_vae: bool = True,
                 vae_model: Optional[str] = None,  # "madebyollin/sdxl-vae-fp16-fix",
                 attn_slicing: bool = False,
                 scheduler: Optional[str] = "EulerDiscreteScheduler",
                 lora_weights: Optional[list[str]] = None,
                 torch_compile: Optional[str] = None,
                 trace: bool = False,
                 nobar: bool = False,

                 extended: bool = False,  # Add im2im and inpaint models reusing components
                 cache_d: Optional[Pathlike] = None):
        self.prompt = None
        if cache_d:
            assert_isdir(cache_d, "Cache override is not a directory.")
            Path(cache_d).mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_d)

        if fix_vae and vae_model is None:
            vae_model = "madebyollin/sdxl-vae-fp16-fix"

        self.cfg = self.CFG(
            model_name=model_name,
            fp16=fp16,
            offload=offload,
            vae_tiling=vae_tiling,
            vae_slicing=vae_slicing,
            attn_slicing=attn_slicing,
            vae_model=vae_model,
            fix_vae=fix_vae,
            extended=extended
        )

        dtype = {"torch_dtype": torch.float16, "variant": "fp16"} if fp16 else {}

        if vae_model is not None:
            vae = AutoencoderKL.from_pretrained(
                vae_model,
                torch_dtype=torch.float16
            )
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_name, **dtype, vae=vae, use_safetensors=True)
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(model_name, **dtype, use_safetensors=True)

        if lora_weights is not None:
            if type(lora_weights) != list:
                lora_weights = [lora_weights]
            for lora_weight in lora_weights:
                self.pipe.load_lora_weights(lora_weight)
            self.cfg['lora_weights'] = lora_weights
        # self.pipe.load_lora_weights("minimaxir/sdxl-wrong-lora")
        # self.pipe.load_lora_weights("artificialguybr/analogredmond-v2")

        self.set_scheduler(scheduler)

        if torch_compile is not None:
            self.pipe.unet = torch.compile(self.pipe.unet, backend=torch_compile, mode="reduce-overhead",
                                           fullgraph=True)

        if vae_tiling:
            self.pipe.enable_vae_tiling()

        if vae_slicing:
            self.pipe.enable_vae_slicing()

        if attn_slicing:
            self.pipe.enable_attention_slicing()

        if offload:
            if offload == Offload.sequential:
                self.pipe.enable_sequential_cpu_offload()
            elif offload == Offload.model:
                self.pipe.enable_model_cpu_offload()
            else:
                print('Warning: Invalid argument to offload ["sequential" | "model"]: ' + offload)
                self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        # if torch_compile is None and trace:
        #     unet_p = trace_unet(self)
        #     replace_unet(self.pipe, unet_p)

        if nobar:
            self.pipe.progress_bar = DummyProgress

        # Set up image2image and inpaint pipes using the same components
        if extended:
            kind_im2im = StableDiffusionXLImg2ImgPipeline  # if "xl" in model_name else StableDiffusionImg2ImgPipeline
            kind_inpaint = StableDiffusionXLInpaintPipeline  # if "xl" in model_name else StableDiffusionInpaintPipeline
            self.im2im = kind_im2im(**self.pipe.components)
            self.inpaint = kind_inpaint(**self.pipe.components)

    @torch.no_grad()
    def set_prompt(self, prompt: Prompt):
        prompt.height = prompt.height or self.pipe.default_sample_size * self.pipe.vae_scale_factor
        prompt.width = prompt.width or self.pipe.default_sample_size * self.pipe.vae_scale_factor
        self.height = prompt.height
        self.width = prompt.width

        self.original_size = (self.height, self.width)
        self.target_size = (self.height, self.width)

        self.pipe.check_inputs(
            prompt.prompt,
            prompt.prompt_2,
            self.height,
            self.width,
            1,  # callback_steps,
            negative_prompt=prompt.negative,
            negative_prompt_2=prompt.negative_2,
            prompt_embeds=prompt.prompt_embed,
            negative_prompt_embeds=prompt.negative_embed,
        )

        self.device = self.pipe._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        self.do_classifier_free_guidance = prompt.guidance_scale > 1.0

        # 3. Encode input prompt
        # lora_scale = cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        lora_scale = None

        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = self.pipe.encode_prompt(
            prompt=prompt.prompt,
            prompt_2=prompt.prompt_2,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            negative_prompt=prompt.negative,
            negative_prompt_2=prompt.negative_2,
            prompt_embeds=prompt.prompt_embed,
            negative_prompt_embeds=prompt.negative_embed,
            pooled_prompt_embeds=None,  # pooled_prompt_embeds,
            negative_pooled_prompt_embeds=None,  # negative_pooled_prompt_embeds,
            lora_scale=lora_scale
        )

        self.batch_size = 1
        self.prompt = prompt

    @torch.no_grad()
    # @torch.inference_mode()
    def explore(self, n: int):
        # gc.collect()
        # torch.cuda.empty_cache()

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(self.prompt.steps, device=self.device)
        self.timesteps = self.pipe.scheduler.timesteps

        # self.pipe.scheduler.set_timesteps(self.prompt.steps)

        # 4.5 Prepare generator
        generator = torch.Generator(self.device).manual_seed(self.prompt.seed)

        # 5. Prepare latent variables
        if self.prompt.seed_latents is not None:
            seed_latents = self.prompt.seed_latents.detach().cpu()
            latents = self.prompt.seed_latents.to(self.device)
        else:
            num_channels_latents = self.pipe.unet.config.in_channels
            latents = self.pipe.prepare_latents(
                self.batch_size,
                num_channels_latents,
                self.height,
                self.width,
                self.prompt_embeds.dtype,
                self.device,
                generator
            )
            seed_latents = latents.detach().cpu()

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, self.prompt.eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = self.pooled_prompt_embeds
        if self.pipe.text_encoder_2 is None:
            text_encoder_projection_dim = int(self.pooled_prompt_embeds.shape[-1])
        else:
            text_encoder_projection_dim = self.pipe.text_encoder_2.config.projection_dim

        add_time_ids = self.pipe._get_add_time_ids(
            self.original_size,
            self.prompt.crops_coords_top_left,
            self.target_size,
            dtype=self.prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )

        if self.prompt.negative_original_size is not None and self.prompt.negative_target_size is not None:
            negative_add_time_ids = self.pipe._get_add_time_ids(
                self.prompt.negative_original_size,
                self.prompt.negative_crops_coords_top_left,
                self.prompt.negative_target_size,
                dtype=self.prompt_embeds.dtype,
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        else:
            negative_add_time_ids = add_time_ids

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.negative_prompt_embeds, self.prompt_embeds], dim=0)
            add_text_embeds = torch.cat([self.negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)
        else:
            prompt_embeds = self.prompt_embeds

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(self.batch_size, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(self.timesteps) - self.prompt.steps * self.pipe.scheduler.order, 0)

        # 7.1 Apply denoising_end
        denoising_end = self.prompt.denoising_end
        if denoising_end is not None and isinstance(denoising_end, float) and 0 < denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.pipe.scheduler.config.num_train_timesteps
                    - (self.prompt.denoising_end * self.pipe.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, self.timesteps)))
            timesteps = self.timesteps[:num_inference_steps]
        else:
            timesteps = self.timesteps

        with self.pipe.progress_bar(total=self.prompt.steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.pipe.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,  # cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.prompt.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.prompt.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text,
                                                   guidance_rescale=self.prompt.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(self.timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.pipe.scheduler.order == 0):
                    progress_bar.update()
                    # if callback is not None and i % callback_steps == 0:
                    #     callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if not self.cfg['fix_vae'] and self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast:
            self.pipe.upcast_vae()
            latents = latents.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

        image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type='pil')

        latents = latents.detach().cpu()

        # Offload last model to CPU
        if hasattr(self.pipe, "final_offload_hook") and self.pipe.final_offload_hook is not None:
            self.pipe.final_offload_hook.offload()

        # gc.collect()
        # torch.cuda.empty_cache()

        prompt_cfg = self.prompt.save()
        model_cfg = self.cfg
        return Chonk(image[0], latents, model_cfg, prompt_cfg, seed_latents=seed_latents)
        # return image[0], latents.cpu(), manifest

    @torch.no_grad()
    def __call__(self, prompt: Prompt | list[Prompt], n: int, path: Optional[Pathlike] = None):
        self.set_prompt(prompt)
        # prompt_embeds = None
        # negative_prompt_embeds = None
        # pooled_prompt_embeds = None
        # negative_pooled_prompt_embeds = None
        batch_size = 1

        if n > 1:
            self.pipe.progress_bar = DummyProgress

        for i in trange(n):
            generator = torch.Generator(self.device).manual_seed(prompt.seed)

            if prompt.seed_latents is not None:
                latents = prompt.seed_latents.to(self.device)
                seed_latents = latents.detach().cpu()
            else:
                num_channels_latents = self.pipe.unet.config.in_channels
                latents = self.pipe.prepare_latents(
                    batch_size,
                    num_channels_latents,
                    prompt.height,
                    prompt.width,
                    self.prompt_embeds.dtype,
                    self.device,
                    generator
                )
                seed_latents = latents.detach().cpu()

            image = self.pipe(None,  # prompt.prompt,
                              # prompt_2=prompt.prompt_2,
                              # negative_prompt=prompt.negative,
                              # negative_prompt_2=prompt.negative_2,

                              prompt_embeds=self.prompt_embeds,
                              negative_prompt_embeds=self.negative_prompt_embeds,
                              pooled_prompt_embeds=self.pooled_prompt_embeds,
                              negative_pooled_prompt_embeds=self.negative_pooled_prompt_embeds,

                              height=prompt.height,
                              width=prompt.width,
                              num_inference_steps=prompt.steps,
                              denoising_end=prompt.denoising_end,
                              guidance_scale=prompt.guidance_scale,
                              guidance_rescale=prompt.guidance_rescale,
                              eta=prompt.eta,

                              generator=generator,
                              latents=latents,

                              # cross_attention_kwargs={'scale'=prompt.lora_scale},
                              # original_size=prompt.original_size,
                              crops_coords_top_left=prompt.crops_coords_top_left,
                              negative_crops_coords_top_left=prompt.negative_crops_coords_top_left,
                              negative_original_size=prompt.negative_original_size,
                              negative_target_size=prompt.negative_target_size,

                              # output_type='latent'
                              ).images[0]

            prompt_cfg = prompt.save()
            chonk = Chonk(image, latents, self.cfg, prompt_cfg, seed_latents=seed_latents)
            chonk.save(path)
            prompt.reseed()

    def old_call(self,
                 prompts: Prompt | list[Prompt],
                 n: int = 1,
                 height: int = 1024, width: int = 1024,
                 steps: int = 25,
                 guidance_scale: float = 5.0,
                 guidance_rescale: float = 0.7,
                 eta: float = 0.0,

                 latents: Optional[torch.FloatTensor] = None,
                 ):

        if type(prompts) != list:
            prompts = [prompts]

        ps = []
        ns = []
        ps2 = []
        ns2 = []
        gens = []
        # print(prompts)
        for p in prompts:
            if not p.seed:
                p.seed = int(time.time())
            p.manifest = self.manifest
            gens.append(torch.Generator("cuda").manual_seed(p.seed))
            ps.append(p.prompt)
            ns.append(p.negative)
            ps2.append(p.prompt_2)
            ns2.append(p.negative_2)

        print(ps)
        with torch.inference_mode():
            imgs = self.pipe(ps,
                             # prompt_2=ps2,
                             negative_prompt=ns,
                             # negative_prompt_2=ns2,
                             height=height,
                             width=width,
                             num_inference_steps=steps,
                             guidance_scale=guidance_scale,
                             guidance_rescale=guidance_rescale,
                             eta=eta).images  # ,
            # output_type='latents')
            # imgs = []
            #
            # # make sure the VAE is in float32 mode, as it overflows in float16
            # needs_upcasting = self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast
            # if needs_upcasting:
            #     self.pipe.upcast_vae()
            #
            # for lat in lats:
            #     if needs_upcasting:
            #         lat = lat.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)
            #     imgs.append(self.pipe.vae.decode(lat / self.pipe.vae.config.scaling_factor, return_dict=False)[0])
            #
            # # cast back to fp16 if needed
            # if needs_upcasting:
            #     self.pipe.vae.to(dtype=torch.float16)

            return imgs, prompts

        # if batch == 1:
        #     with torch.inference_mode():
        #         return self.pipe(prompt,
        #                          prompt_2=prompt_2,
        #                          negative_prompt=negative,
        #                          negative_prompt_2=negative_2,
        #                          height=height,
        #                          width=width,
        #                          num_inference_steps=steps,
        #                          guidance_scale=guidance_scale,
        #                          eta=eta,
        #                          prompt_embeds=prompt_embed,
        #                          negative_prompt_embeds=negative_embed,
        #                          guidance_rescale=guidance_rescale)
        #
        # generators = []
        # for i in range(batch):

# def load_diffusion(model: str ,
#                    fp16: bool = True,      # use fp16 variant
#                    cpu_offload: str = "",  # "sequential" | "model"
#                    vae_tiling: bool = False,
#                    vae_slicing: bool = False,
#                    attention_slicing: bool = False,
#                    torch_compile: bool = False,
#                    cache_d: Pathlike = None):
#     if cache_d:
#         cache_d = Path(cache_d)
#         assert_isdir(cache_d, "Override cache is not a directory.")
#         if not cache_d.exists():
#             print("Making override cache directory: " + cache_d.sanestr(expand=True), flush=True)
#             cache_d.mkdir(parents=True, exist_ok=True)
#         os.environ["HF_DATASETS_CACHE"] = str(cache_d)
#
#     dtype = {"torch_dtype": torch.float16, "variant": "fp16"} if fp16 else {}
#
#     pipe = DiffusionPipeline.from_pretrained(model,
#                                              **dtype,
#                                              use_safetensor=True,
#                                              safety_checker=None,
#                                              requires_safety_checker=False)
#
#     if torch_compile:
#         pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
#
#     if vae_tiling:
#         pipe.enable_vae_tiling()
#
#     if vae_slicing:
#         pipe.enable_vae_slicing()
#
#     if attention_slicing:
#         pipe.enable_attention_slicing()
#
#     if cpu_offload:
#         if cpu_offload == "sequential":
#             pipe.enable_sequential_cpu_offload()
#         elif cpu_offload == "model":
#             pipe.enable_model_cpu_offload()
#         else:
#             raise Exception('Invalid argument to cpu_offload ["sequential" | "model"]: ' + cpu_offload)
#     else:
#         pipe.to("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     return pipe
#
#
# class DiffusionNames:
#     sd1_4 = "CompVis/stable-diffusion-v1-4"
#     sd1_5 = "runwayml/stable-diffusion-v1-5"
#     sd2_1 = "stabilityai/stable-diffusion-2-1"
#     sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
