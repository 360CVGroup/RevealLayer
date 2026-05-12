import os
import sys
import math
import random
import argparse
import json
import shutil
import glob

import cv2
import numpy as np
import torch
import torch.utils.checkpoint
from PIL import Image
from tqdm import tqdm
from mmengine.config import Config
from torchvision import transforms
from safetensors.torch import load_file

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from diffusers import FluxTransformer2DModel, AutoencoderKL
from diffusers.utils import check_min_version
from diffusers.configuration_utils import FrozenDict
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING

from models.custom_model_xvae import AutoencoderKLTransformerTraining as CustomVAE
from models.custom_pipeline import CustomFluxPipelineCfg
from models.custom_model_mmdit import CustomFluxTransformer2DModel

check_min_version("0.31.0.dev0")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_config(path):
    config = Config.fromfile(path)
    config.config_dir = path
    if "LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
    elif "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ:
        config.local_rank = int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])
    else:
        config.local_rank = -1
    return config


def _load_layer_pe(transformer, ckpt_dir: str):
    print("Loading 'layer_pe' weights...")

    corrected_path = os.path.join(ckpt_dir, "layer_pe_corrected.pt")
    fallback_path = os.path.join(ckpt_dir, "layer_pe.pt")

    layer_pe_path = corrected_path if os.path.exists(corrected_path) else fallback_path

    if not os.path.exists(layer_pe_path):
        print(f"[ERROR] 'layer_pe' not found in {ckpt_dir}. Abort.")
        return

    print(f"Using layer_pe file: {layer_pe_path}")

    try:
        layer_pe_tensor = torch.load(layer_pe_path, map_location="cpu")
        transformer.layer_pe.data.copy_(layer_pe_tensor)
        print("'layer_pe' loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load 'layer_pe': {e}")


def _load_refiner(transformer, ckpt_dir: str):
    module_path = os.path.join(ckpt_dir, "Refiner.pt")
    if not module_path or not os.path.exists(module_path):
        print(f"[WARNING] 'Refiner' path not found: {module_path}. Skipping.")
        return

    print(f"Loading 'refiner' from {module_path}...")
    try:
        state_dict = torch.load(module_path, map_location="cpu")
        corrected_state_dict = {f"refiner.{k}": v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = transformer.load_state_dict(corrected_state_dict, strict=False)
        print("'Refiner' loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load 'Refiner': {e}")


def _prepare_custom_transformer(config, args):
    print("STEP 1: Initializing base Transformer model...")
    transformer_orig = FluxTransformer2DModel.from_pretrained(
        config.transformer_varient if hasattr(config, "transformer_varient") else config.pretrained_model_name_or_path,
        subfolder="" if hasattr(config, "transformer_varient") else "transformer",
        revision=config.revision,
        variant=config.variant,
        torch_dtype=torch.bfloat16,
        cache_dir=config.get("cache_dir", None),
    )

    print("STEP 2: Preparing custom Transformer model...")
    mmdit_config = dict(transformer_orig.config)
    mmdit_config["_class_name"] = "CustomSD3Transformer2DModel"
    mmdit_config["max_layer_num"] = args.max_layer
    mmdit_config = FrozenDict(mmdit_config)

    transformer = CustomFluxTransformer2DModel.from_config(mmdit_config).to(dtype=torch.bfloat16)
    transformer.load_state_dict(transformer_orig.state_dict(), strict=False)

    _load_layer_pe(transformer, args.ckpt_dir)
    _load_refiner(transformer, args.ckpt_dir)

    del transformer_orig
    return transformer


def _load_vae_encoder(args):
    print("STEP 3: Loading VAE for full image encoding...")
    vae_encoder = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=None,
        variant=None,
        torch_dtype=torch.bfloat16,
    )
    return vae_encoder


def _load_loras(pipeline, args):
    print("STEP 5: Loading LoRA weights...")

    lora_path = os.path.join(args.ckpt_dir, "pytorch_lora_weights.safetensors")
    if not os.path.exists(lora_path):
        print(f"[ERROR] Main LoRA file not found at {lora_path}. Skipping.")
        return pipeline

    try:
        dit_state_dict = load_file(lora_path)
        corrected_state_dict = {
            k.replace("transformer.module.", "transformer."): v
            for k, v in dit_state_dict.items()
        }
        print("Corrected LoRA keys, first 5:", list(corrected_state_dict.keys())[:5])
        pipeline.load_lora_weights(corrected_state_dict, adapter_name="layer")
        print("Primary LoRA 'layer' loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load primary LoRA: {e}")
        return pipeline

    if hasattr(args, "extra_lora_dir") and args.extra_lora_dir:
        print(f"Loading extra LoRA from {args.extra_lora_dir}...")
        try:
            pipeline.load_lora_weights(args.extra_lora_dir, adapter_name="extra")
            pipeline.set_adapters(["layer", "extra"], adapter_weights=[1.0, 0.5])
            print("Extra LoRA 'extra' loaded with weight 0.5.")
        except Exception as e:
            print(f"[ERROR] Failed to load or set extra LoRA: {e}")

    return pipeline


def initialize_pipeline(config, args):
    print("--- Starting Pipeline Initialization ---")

    transformer = _prepare_custom_transformer(config, args)
    vae_encoder = _load_vae_encoder(args)

    print("STEP 4: Building pipeline and moving to GPU...")
    pipeline_type = CustomFluxPipelineCfg
    pipeline = pipeline_type.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae_encoder,
        revision=config.revision,
        variant=config.variant,
        torch_dtype=torch.bfloat16,
        cache_dir=config.get("cache_dir", None),
    ).to(torch.device("cuda", index=args.gpu_id))

    _load_loras(pipeline, args)

    print("--- Pipeline Initialized Successfully ---")
    return pipeline


def adjust_coordinate(value, floor_or_ceil, k=16, min_val=0, max_val=1024):
    if floor_or_ceil == "floor":
        rounded_value = math.floor(value / k) * k
    else:
        rounded_value = math.ceil(value / k) * k
    return max(min_val, min(rounded_value, max_val))


def _transform_image_consistent_crop(image: Image.Image, content_w, content_h, target_w, target_h, offset_x, offset_y):
    resized = image.resize((content_w, content_h), Image.Resampling.LANCZOS)
    crop_box = (offset_x, offset_y, offset_x + target_w, offset_y + target_h)
    final = resized.crop(crop_box)
    return final


def resize_with_bbox_edge(image: Image.Image, bbox, resolution=1024, k=16):
    orig_w, orig_h = image.size
    long_side = max(orig_w, orig_h)
    ideal_scale = resolution / long_side

    new_content_w = int(round(orig_w * ideal_scale))
    new_content_h = int(round(orig_h * ideal_scale))

    target_w = (new_content_w // k) * k
    target_h = (new_content_h // k) * k

    target_w = max(k, target_w)
    target_h = max(k, target_h)

    offset_x = (new_content_w - target_w) // 2
    offset_y = (new_content_h - target_h) // 2

    real_scale_w = new_content_w / orig_w if orig_w > 0 else 0
    real_scale_h = new_content_h / orig_h if orig_h > 0 else 0

    new_bbox = []
    for bb in bbox:
        x1, y1, x2, y2 = bb
        sx1 = x1 * real_scale_w
        sy1 = y1 * real_scale_h
        sx2 = x2 * real_scale_w
        sy2 = y2 * real_scale_h
        new_bbox.append([
            sx1 - offset_x,
            sy1 - offset_y,
            sx2 - offset_x,
            sy2 - offset_y,
        ])

    return (new_content_h, new_content_w), (target_h, target_w), (offset_y, offset_x), new_bbox


def filter_and_align_bboxes(
    raw_bbox_list,
    orig_w,
    orig_h,
    cont_w,
    cont_h,
    off_x,
    off_y,
    tgt_w,
    tgt_h,
):
    if orig_w <= 0 or orig_h <= 0:
        return []

    real_scale_w = cont_w / orig_w
    real_scale_h = cont_h / orig_h

    valid_obj_bbox = []

    for box in raw_bbox_list:
        x1, y1, x2, y2 = box

        sx1 = x1 * real_scale_w
        sy1 = y1 * real_scale_h
        sx2 = x2 * real_scale_w
        sy2 = y2 * real_scale_h

        scaled_w = sx2 - sx1
        scaled_h = sy2 - sy1

        cx1 = max(0, min(sx1 - off_x, tgt_w))
        cy1 = max(0, min(sy1 - off_y, tgt_h))
        cx2 = max(0, min(sx2 - off_x, tgt_w))
        cy2 = max(0, min(sy2 - off_y, tgt_h))

        visible_w = cx2 - cx1
        visible_h = cy2 - cy1

        if (
            visible_w > 1
            and visible_h > 1
            and scaled_w > 0
            and visible_w / scaled_w >= 0.5
            and scaled_h > 0
            and visible_h / scaled_h >= 0.5
        ):
            fx1 = adjust_coordinate(cx1, "floor", k=16, max_val=tgt_w)
            fy1 = adjust_coordinate(cy1, "floor", k=16, max_val=tgt_h)
            fx2 = adjust_coordinate(cx2, "ceil", k=16, max_val=tgt_w)
            fy2 = adjust_coordinate(cy2, "ceil", k=16, max_val=tgt_h)

            valid_obj_bbox.append([fx1, fy1, fx2, fy2])

    return valid_obj_bbox


def test_one_sample(args, test_sample, pipeline, transp_vae, device, shuffle=False):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    this_index = test_sample["index"]

    validation_boxes_raw = test_sample["layout"]

    print(f"Shuffle layer order: {shuffle}")
    if shuffle:
        random.shuffle(validation_boxes_raw)

    original_full_image_path = test_sample["full_image_path"]
    full_image = Image.open(original_full_image_path)
    orig_w, orig_h = full_image.size

    (cont_h, cont_w), (tgt_h, tgt_w), (off_y, off_x), _ = resize_with_bbox_edge(
        full_image,
        [],
        resolution=1024,
    )

    resized_boxes = filter_and_align_bboxes(
        validation_boxes_raw,
        orig_w,
        orig_h,
        cont_w,
        cont_h,
        off_x,
        off_y,
        tgt_w,
        tgt_h,
    )

    resized_image = _transform_image_consistent_crop(
        full_image,
        cont_w,
        cont_h,
        tgt_w,
        tgt_h,
        off_x,
        off_y,
    )

    validation_boxes_processed = [[0, 0, tgt_w, tgt_h], [0, 0, tgt_w, tgt_h]] + resized_boxes

    image_transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    full_img_tensor = image_transform(resized_image).to(memory_format=torch.contiguous_format)
    full_image = full_img_tensor.to(device=device, dtype=torch.bfloat16).unsqueeze(0)

    print(f"Full image tensor shape: {full_image.shape}")

    validation_boxes_final = [
        (
            adjust_coordinate(rect[0], "floor", max_val=tgt_w),
            adjust_coordinate(rect[1], "floor", max_val=tgt_h),
            adjust_coordinate(rect[2], "ceil", max_val=tgt_w),
            adjust_coordinate(rect[3], "ceil", max_val=tgt_h),
        )
        for rect in validation_boxes_processed
    ]

    if len(validation_boxes_final) > 12:
        validation_boxes_final = validation_boxes_final[:12]

    print(f"Running generation for sample: {this_index} with {len(validation_boxes_final)} layers...")

    output, rgba_output, _, _ = pipeline(
        prompt="Decompose the image into foreground and background.",
        full_image=full_image,
        validation_box=validation_boxes_final,
        generator=generator,
        height=tgt_h,
        width=tgt_w,
        num_layers=len(validation_boxes_final),
        guidance_scale=args.cfg,
        num_inference_steps=args.steps,
        transparent_decoder=transp_vae,
    )

    sample_output_dir = os.path.join(args.output_dir, this_index)
    os.makedirs(sample_output_dir, exist_ok=True)

    try:
        original_image_save_path = os.path.join(sample_output_dir, "original_full_image.png")
        shutil.copy2(original_full_image_path, original_image_save_path)
        print(f"Original input image saved to {original_image_save_path}")

        resized_image_save_path = os.path.join(sample_output_dir, "resized_full_image.png")
        resized_image.save(resized_image_save_path)
        print(f"Resized image saved to {resized_image_save_path}")
    except Exception as e:
        print(f"[ERROR] Failed to copy original image: {e}")

    rgba_images = [Image.fromarray(arr, "RGBA") for arr in rgba_output]

    merged_pil = rgba_images[0]
    for i in range(1, len(rgba_images)):
        merged_pil = Image.alpha_composite(merged_pil, rgba_images[i])

    merged_save_path = os.path.join(sample_output_dir, "merged_image.png")
    merged_pil.save(merged_save_path)

    for i, frame_pil in enumerate(rgba_images):
        if frame_pil.mode != "RGBA":
            print(f"Warning: Skipping non-RGBA image at index {i}. Mode: {frame_pil.mode}")
            continue

        if i == 0:
            base_name = "fg"
            layer_save_path = os.path.join(sample_output_dir, f"{base_name}_rgba.png")
            frame_pil.save(layer_save_path)
        elif i == 1:
            base_name = "bg"
            layer_save_path = os.path.join(sample_output_dir, f"{base_name}_rgba.png")
            frame_pil.save(layer_save_path)
        else:
            layer_index = i - 2
            base_name = f"layer_{layer_index}"

            layer_save_path = os.path.join(sample_output_dir, f"{base_name}_rgba.png")
            frame_pil.save(layer_save_path)

            alpha_channel = frame_pil.getchannel("A")
            mask_save_path = os.path.join(sample_output_dir, f"{base_name}_rgba_mask.png")
            alpha_channel.save(mask_save_path)

    print(f"All {len(rgba_images)} layers and merged image saved to {sample_output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run multi-layer image generation test from a JSON manifest in parallel."
    )

    parser.add_argument(
        "--input_json",
        type=str,
        default="./benchmark/RevealLayerBenchMark-200.json",
        help="Path to the input JSON manifest file containing a list of samples.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results/RevealLayerV1_bench200",
        help="Directory to save the output images.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/jovyan/liushanyuan-sh-ceph-new/tools/black-forest-labs/FLUX.1-dev",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="./configs/ld_resolution1024_test.py",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./ckpts/RevealLayer",
    )
    parser.add_argument(
        "--transp_vae_ckpt",
        type=str,
        default="./ckpts/xvae/transparent_decoder_ckpt.pth",
    )
    parser.add_argument("--cfg", type=float, default=1.0, help="Classifier-Free Guidance scale.")
    parser.add_argument("--steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=43, help="Random seed for reproducibility.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use.")
    parser.add_argument("--max_layer", type=int, default=12)

    parser.add_argument(
        "--max_samples",
        default=200,
        type=int,
        help="Maximum number of samples to process from the JSON. Use -1 for all.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="If set, shuffle the detection order for each sample before generation.",
    )

    parser.add_argument(
        "--tols",
        default=1,
        type=int,
        help="Total number of processes/slices to run in parallel.",
    )
    parser.add_argument(
        "--cid",
        default=0,
        type=int,
        help="Current process ID/slice index, from 0 to tols-1.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.seed is not None:
        seed_everything(args.seed)

    try:
        with open(args.input_json, "r") as f:
            all_data_entries = json.load(f)
        print(f"Loaded {len(all_data_entries)} samples from {args.input_json}")
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {args.input_json}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON file at {args.input_json}")
        sys.exit(1)

    if args.max_samples > 0 and len(all_data_entries) > args.max_samples:
        all_data_entries = all_data_entries[:args.max_samples]
        print(f"Processing the first {len(all_data_entries)} samples as per --max_samples.")

    total_samples = len(all_data_entries)
    samples_per_core = total_samples // args.tols
    remainder = total_samples % args.tols

    start_idx = samples_per_core * args.cid + min(args.cid, remainder)
    end_idx = start_idx + samples_per_core + (1 if args.cid < remainder else 0)

    assigned_samples = all_data_entries[start_idx:end_idx]
    print(
        f"Process {args.cid}/{args.tols} will process {len(assigned_samples)} samples "
        f"(from index {start_idx} to {end_idx - 1})."
    )

    config = parse_config(args.cfg_path)
    device = torch.device("cuda", index=args.gpu_id)

    pipeline = initialize_pipeline(config, args)

    transp_vae = CustomVAE()
    transp_vae.load_state_dict(torch.load(args.transp_vae_ckpt, map_location="cpu"), strict=False)
    transp_vae.to(device).eval()

    print("--- Model initialization complete. ---")

    for entry in tqdm(assigned_samples, desc=f"Processing slice {args.cid}"):
        try:
            full_image_path = entry.get("full_image", "")
            if not full_image_path:
                print(f"Warning: No 'full_image' path found for imgid {entry.get('imgid', 'N/A')}. Skipping.")
                continue

            layout_from_detections = [d["bbox"] for d in entry.get("detections", [])]

            sample = {
                "index": entry.get("imgid", "0000xxx"),
                "description": entry.get("description", "A detailed image."),
                "layout": layout_from_detections,
                "full_image_path": full_image_path,
            }

            output_path = os.path.join(args.output_dir, sample["index"])
            index = sample["index"]

            try:
                os.makedirs(output_path, exist_ok=False)
            except FileExistsError:
                image_files = (
                    glob.glob(os.path.join(output_path, "*.png"))
                    + glob.glob(os.path.join(output_path, "*.jpg"))
                    + glob.glob(os.path.join(output_path, "*.jpeg"))
                )
                if image_files:
                    print(f"imgid: {index}, output folder already exists with images. Skipping.")
                    continue
                else:
                    pass

            test_one_sample(args, sample, pipeline, transp_vae, device, args.shuffle)

        except Exception as e:
            print(f"An error occurred while processing entry with imgid {entry.get('imgid', 'N/A')}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nProcess {args.cid} finished successfully.")
    print(f"Results are saved in sub-folders under: {args.output_dir}")

    del pipeline, transp_vae

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()