import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import is_torch_xla_available, logging
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps, FluxPipeline
import pdb
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm # type: ignore
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_clip_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
):
    device = device or text_encoder.device
    dtype = text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=text_encoder.config.max_position_embeddings,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def _get_t5_prompt_embeds(
    tokenizer,
    text_encoder,
    prompt: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or text_encoder.device
    dtype = dtype or text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def encode_prompt(
    tokenizers,
    text_encoders,
    prompt: Union[str, List[str]],
    prompt_2: Union[str, List[str]] = None,
    num_images_per_prompt: int = 1,
    max_sequence_length: int = 512,
):

    tokenizer_1, tokenizer_2 = tokenizers
    text_encoder_1, text_encoder_2 = text_encoders
    device = text_encoder_1.device
    dtype = text_encoder_1.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_2 = prompt_2 or prompt
    prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

    # We only use the pooled prompt output from the CLIPTextModel
    pooled_prompt_embeds = _get_clip_prompt_embeds(
        tokenizer=tokenizer_1,
        text_encoder=text_encoder_1,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
    )
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer=tokenizer_2,
        text_encoder=text_encoder_2,
        prompt=prompt_2,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


class CustomFluxPipeline(FluxPipeline):

    @staticmethod
    def _prepare_latent_image_ids(height, width, list_layer_box, device, dtype):

        latent_image_ids_list = []
        for layer_idx in range(len(list_layer_box)):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                latent_image_ids = torch.zeros(height // 2, width // 2, 3) # [h/2, w/2, 3]
                latent_image_ids[..., 0] = layer_idx # use the first dimension for layer representation
                latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
                latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                latent_image_ids = latent_image_ids[y1:y2, x1:x2, :]

                latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
                latent_image_ids = latent_image_ids.reshape(
                    latent_image_id_height * latent_image_id_width, latent_image_id_channels
                )

                latent_image_ids_list.append(latent_image_ids)

        full_latent_image_ids = torch.cat(latent_image_ids_list, dim=0)

        return full_latent_image_ids.to(device=device, dtype=dtype)

    def prepare_latents(
        self,
        batch_size,
        num_layers,
        num_channels_latents,
        height,
        width,
        list_layer_box,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // self.vae_scale_factor)
        width = 2 * (int(width) // self.vae_scale_factor)

        shape = (batch_size, num_layers, num_channels_latents, height, width) 

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) # [bs, f, c_latent, h, w]

        latent_image_ids = self._prepare_latent_image_ids(height, width, list_layer_box, device, dtype)

        return latents, latent_image_ids

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        validation_box: List[tuple] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        num_layers: int = 5,
        sdxl_vae: nn.Module = None,
        transparent_decoder: nn.Module = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_layers,
            num_channels_latents,
            height,
            width,
            validation_box,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latent_image_ids.shape[0] # ???
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    list_layer_box=validation_box,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # create a grey latent
        bs, n_frames, channel_latent, height, width = latents.shape

        pixel_grey = torch.zeros(size=(bs*n_frames, 3, height*8, width*8), device=latents.device, dtype=latents.dtype)
        latent_grey = self.vae.encode(pixel_grey).latent_dist.sample()
        latent_grey = (latent_grey - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latent_grey = latent_grey.view(bs, n_frames, channel_latent, height, width)  # [bs, f, c_latent, h, w]
        
        # fill in the latents
        for layer_idx in range(latent_grey.shape[1]):
            x1, y1, x2, y2 = validation_box[layer_idx]
            x1, y1, x2, y2 = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            latent_grey[:, layer_idx, :, y1:y2, x1:x2] = latents[:, layer_idx, :, y1:y2, x1:x2]
        latents = latent_grey

        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.reshape(bs * n_frames, channel_latent, height, width)
            image = self.vae.decode(latents, return_dict=False)[0]
            if sdxl_vae is not None:
                sdxl_vae = sdxl_vae.to(dtype=image.dtype, device=image.device)
                sdxl_latents = sdxl_vae.encode(image).latent_dist.sample()
                transparent_decoder = transparent_decoder.to(dtype=image.dtype, device=image.device)
                result_list, vis_list = transparent_decoder(sdxl_vae, sdxl_latents)
            else:
                result_list, vis_list = None, None
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, result_list, vis_list)

        return FluxPipelineOutput(images=image), result_list, vis_list


class CustomFluxPipelineCfg(FluxPipeline):

    @staticmethod
    def _prepare_latent_image_ids(height, width, list_layer_box, device, dtype):

        latent_image_ids_list = []
        for layer_idx in range(len(list_layer_box)):
            if list_layer_box[layer_idx] == None:
                continue
            else:
                latent_image_ids = torch.zeros(height // 2, width // 2, 3) # [h/2, w/2, 3]
                latent_image_ids[..., 0] = layer_idx # use the first dimension for layer representation
                latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
                latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]

                x1, y1, x2, y2 = list_layer_box[layer_idx]
                x1, y1, x2, y2 = x1 // 16, y1 // 16, x2 // 16, y2 // 16
                latent_image_ids = latent_image_ids[y1:y2, x1:x2, :]

                latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
                latent_image_ids = latent_image_ids.reshape(
                    latent_image_id_height * latent_image_id_width, latent_image_id_channels
                )

                latent_image_ids_list.append(latent_image_ids)

        full_latent_image_ids = torch.cat(latent_image_ids_list, dim=0)

        return full_latent_image_ids.to(device=device, dtype=dtype)

    def prepare_latents(
        self,
        batch_size,
        num_layers,
        num_channels_latents,
        height,
        width,
        list_layer_box,
        dtype,
        device,
        generator,
        latents=None,
    ):
        #height = 2 * (int(height) // self.vae_scale_factor)
        #width = 2 * (int(width) // self.vae_scale_factor)
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_layers, num_channels_latents, height, width) 
        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height, width, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype) # [bs, n_layers, c_latent, h, w]

        latent_image_ids = self._prepare_latent_image_ids(height, width, list_layer_box, device, dtype)

        return latents, latent_image_ids

    @torch.no_grad()
    def __call__(
        self,
        full_image = torch.Tensor,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        validation_box: List[tuple] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        true_gs: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        num_layers: int = 5,
        transparent_decoder: nn.Module = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """
        
        #pdb.set_trace()
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        (
            neg_prompt_embeds,
            neg_pooled_prompt_embeds,
            neg_text_ids,
        ) = self.encode_prompt(
            prompt="",
            prompt_2=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        #pdb.set_trace()
        # 4. Prepare latent variables  
        full_image_unsqueezed = full_image.unsqueeze(1)
        b, n, c, h, w = full_image_unsqueezed.shape
        init_full = full_image_unsqueezed
        full_image_unsqueezed = full_image_unsqueezed.reshape(b * n, c, h, w)
        full_image_latent = self.vae.encode(full_image_unsqueezed).latent_dist.sample()
        full_image_latent = (full_image_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        _, latent_c, latent_h, latent_w = full_image_latent.shape
        full_image_latent = full_image_latent.view(b, 1, latent_c, latent_h, latent_w)
        # print("full_image_latent:",full_image_latent.shape)

        num_channels_latents = self.transformer.config.in_channels // 4
        # mask
        latents, latent_image_ids = self.prepare_latents(       # [1, n, 16, 128, 128], [8257, 3]
            batch_size * num_images_per_prompt,
            num_layers,
            num_channels_latents,
            height,
            width,
            validation_box,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # print("latents:",latents.shape)
        latents[:, 0, :, :, :] = full_image_latent
        
        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latent_image_ids.shape[0]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None



        ##Adapter data####
        num_bg_fgs = num_layers - 1
        init_images = init_full.expand(-1, num_bg_fgs, -1, -1, -1)
        all_images_for_vae = init_images.flatten(0, 1)

        split_sizes = []
        split_sizes.append((height//16) * (width//16))
        mask_pixel = torch.ones(b, num_bg_fgs, 1, h, w, device=device, dtype=latents.dtype)
        overlap_counter = torch.zeros((b, h, w), device=device)
        
        for fg_idx in range(2, len(validation_box)):
            if validation_box[fg_idx] is None: continue
            fg_box = validation_box[fg_idx] 
            bx1, by1, bx2, by2 = fg_box
            overlap_counter[:, by1:by2, bx1:bx2] += 1
            
            h_p = (by2 - by1) // 16
            w_p = (bx2 - bx1) // 16
            split_sizes.append(h_p * w_p)

        is_fg_union = (overlap_counter > 0)          
        is_overlap_region_global = (overlap_counter > 1) 

        mask_pixel[:, 0] = is_fg_union
        for i, fg_idx_in_bbox in enumerate(range(2, len(validation_box)), start=1):
            if validation_box[fg_idx_in_bbox] is None: continue
            x1, y1, x2, y2 = validation_box[fg_idx_in_bbox]
            roi_overlap = is_overlap_region_global[:, y1:y2, x1:x2]
            mask_pixel[:, i, :, y1:y2, x1:x2] = roi_overlap

        mask_image_flat = mask_pixel.view(b * num_bg_fgs, 1, h, w)
        masked_image = all_images_for_vae * (1 - mask_image_flat)
        masked_image = masked_image.to(device=device, dtype=latents.dtype)

        # Encode -> [B*n, 16, H_lat, W_lat]
        with torch.no_grad():
            masked_latents = self.vae.encode(masked_image).latent_dist.sample()
            masked_latents = (masked_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        
        # Patchify Clean Latents -> [B*n, H/16, W/16, 64]
        b_m, c_m, h_m, w_m = masked_latents.shape
        masked_latents = masked_latents.view(b_m, c_m, h_m // 2, 2, w_m // 2, 2)
        masked_latents = masked_latents.permute(0, 2, 4, 1, 3, 5) 
        masked_latents = masked_latents.reshape(b_m, h_m // 2, w_m // 2, c_m * 4) 
        vae_scale_factor = 8  # VAE scale
        b_sz, _, h_px, w_px = mask_image_flat.shape
        
        # VAE Latent dimensions (e.g. 128)
        h_lat = h_px // vae_scale_factor
        w_lat = w_px // vae_scale_factor

        # 1. Pixel -> Latent (8x8 blocking) -> [B*n, 128, 128, 64]
        mask_latents = mask_image_flat.view(
            b_sz, 1, h_lat, vae_scale_factor, w_lat, vae_scale_factor
        )
        mask_latents = mask_latents.permute(0, 2, 4, 1, 3, 5)
        mask_latents = mask_latents.reshape(
            b_sz, h_lat, w_lat, vae_scale_factor * vae_scale_factor
        )
        
        # 2. Downsample 2x (128 -> 64) -> [B*n, 64, 64, 64]
        mask_latents = mask_latents.permute(0, 3, 1, 2)
        mask_latents = F.max_pool2d(mask_latents, kernel_size=2, stride=2)
        # Permute back to (N, H, W, C)
        mask_latents = mask_latents.permute(0, 2, 3, 1)
        mask_latents = mask_latents.to(dtype=latents.dtype)
        masked_image_latents_flat = torch.cat([masked_latents, mask_latents], dim=-1)
        _, h_patch, w_patch, c_total = masked_image_latents_flat.shape
        masked_image_latents = masked_image_latents_flat.view(b, num_bg_fgs, h_patch, w_patch, c_total)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latents[:, :1, :, :, :] = full_image_latent
                noise_pred = self.transformer(
                    hidden_states=latents,
                    adapter_data=masked_image_latents,
                    split_sizes=split_sizes,
                    list_layer_box=validation_box,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                neg_noise_pred = self.transformer(
                    hidden_states=latents,
                    adapter_data=masked_image_latents,
                    split_sizes=split_sizes,
                    list_layer_box=validation_box,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=neg_pooled_prompt_embeds,
                    encoder_hidden_states=neg_prompt_embeds,
                    txt_ids=neg_text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                noise_pred = neg_noise_pred + true_gs * (noise_pred - neg_noise_pred)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # create a grey latent
        #pdb.set_trace()
        bs, n_layers, channel_latent, height, width = latents.shape

        pixel_grey = torch.zeros(size=(bs*n_layers, 3, height*8, width*8), device=latents.device, dtype=latents.dtype)
        latent_grey = self.vae.encode(pixel_grey).latent_dist.sample()
        latent_grey = (latent_grey - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        latent_grey = latent_grey.view(bs, n_layers, channel_latent, height, width)  # [bs, n_layers, c_latent, h, w]
        
        # fill in the latents
        for layer_idx in range(latent_grey.shape[1]):
            if validation_box[layer_idx] == None:
                continue
            x1, y1, x2, y2 = validation_box[layer_idx]
            x1, y1, x2, y2 = x1 // 8, y1 // 8, x2 // 8, y2 // 8
            latent_grey[:, layer_idx, :, y1:y2, x1:x2] = latents[:, layer_idx, :, y1:y2, x1:x2]
            # mask 0326

        latents = latent_grey

        #pdb.set_trace()
        if output_type == "latent":
            image = latents

        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            latents = latents.reshape(bs * n_layers, channel_latent, height, width)
            latents_segs = torch.split(latents, 16, dim=0) ### split latents by 16 to avoid odd purple output
            image_segs = [self.vae.decode(latents_seg, return_dict=False)[0] for latents_seg in latents_segs]
            image = torch.cat(image_segs, dim=0)
            print("images.shape:",image.shape)
            if transparent_decoder is not None:
                transparent_decoder = transparent_decoder.to(dtype=image.dtype, device=image.device)
                
                decoded_fg, decoded_alpha = transparent_decoder(latents, [validation_box])
                print(decoded_alpha)
                decoded_alpha = (decoded_alpha + 1.0) / 2.0
                decoded_alpha = torch.clamp(decoded_alpha, min=0.0, max=1.0).permute(0, 2, 3, 1)

                decoded_fg = (decoded_fg + 1.0) / 2.0
                decoded_fg = torch.clamp(decoded_fg, min=0.0, max=1.0).permute(0, 2, 3, 1)

                vis_list = None
                png = torch.cat([decoded_fg, decoded_alpha], dim=3)
                result_list = (png * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
                print("result_list.shape:",result_list.shape)
            else:
                result_list, vis_list = None, None
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, result_list, vis_list, latents)

        return FluxPipelineOutput(images=image), result_list, vis_list, latents
