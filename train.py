
from utils.load_save_utils import *
from train_utils import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
from models.processors import *
import torch.nn.functional as F
import seaborn as sns
import pickle as pkl
import html
import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.image_processor import PixArtImageProcessor
from diffusers.models import AutoencoderKL, PixArtTransformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from create_dataset_pixart_latent_noise import *
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
dataset = LatentNoiseDataset()
dataloader = DataLoader(dataset, batch_size=5, shuffle=True, drop_last=True)


device = 'cuda:0'
num_inference_steps  = 20
processor_name = 'processor_x'
model_name = 'pixart_x'
model = load_model(model_name=model_name, device=device)
sigmas = None #check
timesteps = None #check
do_classifier_free_guidance = True
max_sequence_length: int = 120
negative_prompt: str = ""
num_images_per_prompt: int = 1
clean_caption: bool = False
added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
batch_size = 5
guidance_scale = 7.5

for param in list(model.transformer.parameters())+list(model.text_encoder.parameters())+list(model.vae.parameters()):
    param.requires_grad = False



model.transformer.anchor_token = torch.nn.Parameter(torch.randn((num_inference_steps,1,1,1152),requires_grad=True, dtype=torch.float16).to(device))

criterion = nn.MSELoss()
optimizer =optim.Adam([model.transformer.anchor_token],lr = 0)

del model.vae# model.text_encoder.to('cpu')

# model.text_encoder.to('cuda:2')
timesteps, num_inference_steps = retrieve_timesteps(
    model.scheduler, num_inference_steps, device, timesteps, sigmas
)
# breakpoint()
eta = 0.0,
generator= None,
extra_step_kwargs = model.prepare_extra_step_kwargs(generator, eta)

for original_noise_pred, text_prompts in dataloader:
    original_noise_pred = original_noise_pred.to(device)
    
    with torch.no_grad():
        text_prompts = list(text_prompts)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = model.encode_prompt(
            prompt = text_prompts,
            do_classifier_free_guidance = True,
            device=device,
            max_sequence_length=max_sequence_length,
        )
    
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

    prompt_emebds = prompt_embeds.to(device)
    
    
    i = torch.randint(low = 1, high = num_inference_steps, size = (1,))[0]
    
    current_timestep = timesteps[i]
    t = current_timestep
    # breakpoint()
    
    # breakpoint()
    original_noise_input  = original_noise_pred[:,i-1,:,:,:]
    original_noise_output  = original_noise_pred[:,i,:,:,:]
    # breakpoint()
    # noise_pred_uncond, original_noise_pred_text = original_noise_pred.chunk(2)
    # original_noise_pred = noise_pred_uncond + guidance_scale * (original_noise_pred_text - noise_pred_uncond)
    # print('i',i)
    model.scheduler._step_index = i.clone()
    # print('ii',i)
    latents = model.scheduler.step(original_noise_output, t,latents, **extra_step_kwargs, return_dict=False)[0]
    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
    current_timestep = current_timestep.expand(latent_model_input.shape[0])

    # original_noise_pred = model.scheduler.scale_model_input(original_noise_pred, t)
    
    # anchor_token = model.transformer.anchor_token[i]
    # print(i)
    # breakpoint()
    noise_pred = model.transformer(
                latent_model_input.to(torch.float16),
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
                # cross_attention_kwargs={'kwargs':{"attn_scale": scale}}
                cross_attention_kwargs={'kwargs':{'i':i }}
            )[0]

    noise_pred_positive = noise_pred[5:,4:,:,:]
    print(torch.norm(noise_pred_positive))
    # breakpoint()
    optimizer.zero_grad()  # Clear the gradients
    loss = criterion(noise_pred_positive, original_noise_output) 
    print(loss.item())# Compute the loss
    loss.backward()  # Backpropagate the gradients
    optimizer.step()  # Update the model parameters
