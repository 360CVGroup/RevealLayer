#!/usr/bin/env python
"""RevealLayerBench evaluation.

The metrics are evaluated on the full benchmark without bbox cropping. RGBA
layers are composited onto a white background and resized to 1024 x 1024.
PSNR, LPIPS and FID use PyIQA; SoftIoU is computed from the full alpha maps.

Expected predictions:
  <output_dir>/<imgid>/bg_rgba.png
  <output_dir>/<imgid>/layer_<index>_rgba.png
"""

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyiqa
import torch
import torchvision
from PIL import Image
from tqdm import tqdm


SIZE = (1024, 1024)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gt_json', type=str, required=True)
    parser.add_argument('--save_results', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def resolve_path(path, json_path):
    path = Path(path)
    if path.is_absolute() or path.exists():
        return path
    return Path(json_path).resolve().parent / path


def load_rgb(path, device):
    image = Image.open(path)
    if image.mode == "RGBA":
        rgba = np.asarray(image, dtype=np.float32)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4] / 255.0
        image = Image.fromarray((rgb * alpha + 255 * (1 - alpha)).astype(np.uint8))
    else:
        image = image.convert("RGB")
    if image.size != SIZE:
        image = image.resize(SIZE, Image.Resampling.LANCZOS)
    return torchvision.transforms.functional.to_tensor(image).unsqueeze(0).to(device)


def load_alpha(path):
    image = Image.open(path).convert("RGBA")
    if image.size != SIZE:
        image = image.resize(SIZE, Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32)[..., 3] / 255.0


def soft_iou(pred_path, gt_path):
    pred = load_alpha(pred_path)
    gt = load_alpha(gt_path)
    union = np.maximum(pred, gt).sum()
    return float(np.minimum(pred, gt).sum() / union) if union > 0 else 1.0


def summarize(scores):
    return {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in scores.items()
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    with open(args.gt_json, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    metrics = {
        "psnr": pyiqa.create_metric("psnr", device=device),
        "lpips": pyiqa.create_metric("lpips", device=device),
    }
    fid = pyiqa.create_metric("fid", device=device)
    bg_scores, fg_scores = defaultdict(list), defaultdict(list)
    details = []

    with tempfile.TemporaryDirectory(prefix="reveallayer_eval_") as temp:
        temp = Path(temp)
        dirs = {name: temp / name for name in ("bg_pred", "bg_gt", "fg_pred", "fg_gt")}
        for directory in dirs.values():
            directory.mkdir(parents=True)

        with torch.no_grad():
            for item in tqdm(metadata, desc="Evaluating"):
                imgid = str(item["imgid"])
                sample_dir = output_dir / imgid
                record = {"imgid": imgid, "background": {}, "layers": []}

                bg_pred = sample_dir / "bg_rgba.png"
                bg_gt = resolve_path(item["background"], args.gt_json)
                if not bg_pred.exists() or not bg_gt.exists():
                    raise FileNotFoundError(f"Missing background pair for {imgid}")

                pred, gt = load_rgb(bg_pred, device), load_rgb(bg_gt, device)
                record["background"] = {
                    name: float(metric(pred, gt).item()) for name, metric in metrics.items()
                }
                for name, value in record["background"].items():
                    bg_scores[name].append(value)
                torchvision.utils.save_image(pred, dirs["bg_pred"] / f"{imgid}.png")
                torchvision.utils.save_image(gt, dirs["bg_gt"] / f"{imgid}.png")

                for index, gt_name in enumerate(item.get("LayerInfoRaw", [])):
                    pred_path = sample_dir / f"layer_{index}_rgba.png"
                    gt_path = resolve_path(gt_name, args.gt_json)
                    if not pred_path.exists() or not gt_path.exists():
                        raise FileNotFoundError(
                            f"Missing foreground pair: {imgid}, layer {index}"
                        )

                    pred, gt = load_rgb(pred_path, device), load_rgb(gt_path, device)
                    scores = {
                        name: float(metric(pred, gt).item())
                        for name, metric in metrics.items()
                    }
                    scores["softiou"] = soft_iou(pred_path, gt_path)
                    record["layers"].append({"layer_idx": index, **scores})
                    for name, value in scores.items():
                        fg_scores[name].append(value)

                    key = f"{imgid}_{index}.png"
                    torchvision.utils.save_image(pred, dirs["fg_pred"] / key)
                    torchvision.utils.save_image(gt, dirs["fg_gt"] / key)

                details.append(record)

        bg_summary = summarize(bg_scores)
        fg_summary = summarize(fg_scores)
        bg_summary["fid"] = float(
            fid(str(dirs["bg_pred"]), str(dirs["bg_gt"])).item()
        )
        fg_summary["fid"] = float(
            fid(str(dirs["fg_pred"]), str(dirs["fg_gt"])).item()
        )

    result = {
        "settings": {
            "dataset": "full RevealLayerBench",
            "resolution": list(SIZE),
            "bbox_crop": False,
            "rgba_background": "white",
            "metric_library": "PyIQA",
        },
        "num_images": len(metadata),
        "num_layers": len(fg_scores["psnr"]),
        "background_metrics": bg_summary,
        "foreground_metrics": fg_summary,
        "details": details,
    }

    save_path = Path(args.save_results)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print(
        json.dumps(
            {
                "background_metrics": bg_summary,
                "foreground_metrics": fg_summary,
            },
            indent=2,
        )
    )
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
