"""Run several DA3 models on the same frames and save depth maps + disagreement maps.

    uv run tools/compare.py                 # default scenes from footage/Photos-1-001
    uv run tools/compare.py --res 392

Each model sees the same single frame at the same resolution. Disagreement is measured
against the reference model (DA3-LARGE-1.1) after a least-squares scale+shift fit in
inverse depth, since these models predict relative depth. It is agreement, not accuracy:
there is no ground truth here.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")

import argparse
import json
import time
import warnings
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

warnings.filterwarnings("ignore")
from depth_anything_3.api import DepthAnything3  # noqa: E402

from depth import colorize, disparity_range, load_rgb  # noqa: E402

MODELS = [
    "depth-anything/DA3-SMALL",
    "depth-anything/DA3-BASE",
    "depth-anything/DA3MONO-LARGE",
    "depth-anything/DA3-LARGE-1.1",
]
REFERENCE = "depth-anything/DA3-LARGE-1.1"
SCENES = {  # name -> (path, frame index for videos)
    "lab-photo": ("footage/Photos-1-001/PXL_20261116_161723252.jpg", None),
    "tables-clip": ("footage/Photos-1-001/PXL_20261116_161821771.mp4", 270),
    "hallway-clip": ("footage/Photos-1-001/PXL_20261116_161842089.mp4", 228),
}
ERR_CLIP = 0.30  # relative disagreement that maps to full overlay strength (linear)


def load_scene(path: str, frame: int | None, width: int) -> np.ndarray:
    if frame is None:
        rgb = load_rgb(Path(path), 4096)
    else:
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, bgr = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"could not read frame {frame} of {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (width, round(h * width / w / 2) * 2), interpolation=cv2.INTER_AREA)


def short(model: str) -> str:
    return model.split("/")[-1]


def disagreement(disp: np.ndarray, ref: np.ndarray, mask: np.ndarray):
    """Fit s*disp+t to ref on masked pixels; return per-pixel relative error and summary stats."""
    a = np.stack([disp[mask], np.ones(mask.sum())], axis=1)
    s, t = np.linalg.lstsq(a, ref[mask], rcond=None)[0]
    aligned = np.clip(s * disp + t, 1e-6, None)
    err = np.abs(aligned - ref) / ref
    ratio = np.maximum(aligned / ref, ref / aligned)
    return err, float(err[mask].mean()), float((ratio[mask] < 1.25).mean())


def overlay(rgb: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Dimmed grayscale input with orange where the model disagrees with the reference."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None] / 255
    base = 0.12 + 0.30 * gray
    orange = np.array([0xEB, 0x68, 0x34], np.float32) / 255
    alpha = np.clip(err / ERR_CLIP, 0, 1)[..., None]
    out = base * (1 - alpha) + orange * alpha
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=int, default=504)
    p.add_argument("--width", type=int, default=960, help="width of saved images")
    p.add_argument("--out", type=Path, default=Path("outputs/compare"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    scenes = {name: load_scene(path, frame, args.width) for name, (path, frame) in SCENES.items()}
    for name, rgb in scenes.items():
        imageio.imwrite(args.out / f"{name}_input.jpg", rgb, quality=86)

    preds = {}  # (scene, model) -> (depth, conf, ms)
    for model_id in MODELS:
        print(f"{model_id}")
        model = DepthAnything3.from_pretrained(model_id).to("mps").eval()
        model.inference([next(iter(scenes.values()))], process_res=args.res)  # warm-up
        for name, rgb in scenes.items():
            torch.mps.synchronize()
            t = time.time()
            pred = model.inference([rgb], process_res=args.res)
            ms = (time.time() - t) * 1000
            preds[name, model_id] = (pred.depth[0], None if pred.conf is None else pred.conf[0], ms)
            print(f"  {name}: {ms:.0f} ms, depth {pred.depth.shape[1:]}")
        del model
        torch.mps.empty_cache()

    metrics = {"res": args.res, "reference": short(REFERENCE), "scenes": {}}
    for name, rgb in scenes.items():
        h, w = rgb.shape[:2]
        ref_depth, ref_conf, _ = preds[name, REFERENCE]
        ref_disp = 1.0 / np.clip(ref_depth, 1e-6, None)
        mask = np.ones_like(ref_disp, bool) if ref_conf is None else ref_conf > np.percentile(ref_conf, 20)
        metrics["scenes"][name] = {}
        for model_id in MODELS:
            depth, conf, ms = preds[name, model_id]
            lo, hi = disparity_range(depth[None], None if conf is None else conf[None])
            vis = cv2.resize(colorize(depth[None], lo, hi)[0], (w, h), interpolation=cv2.INTER_LINEAR)
            imageio.imwrite(args.out / f"{name}_{short(model_id)}_depth.jpg", vis, quality=88)
            entry = {"ms": round(ms, 1), "depth_hw": list(depth.shape)}
            if model_id != REFERENCE:
                disp = 1.0 / np.clip(depth, 1e-6, None)
                if disp.shape != ref_disp.shape:
                    disp = cv2.resize(disp, ref_disp.shape[::-1], interpolation=cv2.INTER_LINEAR)
                err, absrel, d1 = disagreement(disp, ref_disp, mask)
                err_full = cv2.resize(err, (w, h), interpolation=cv2.INTER_LINEAR)
                imageio.imwrite(args.out / f"{name}_{short(model_id)}_diff.jpg", overlay(rgb, err_full), quality=88)
                entry.update(absrel=round(absrel, 4), delta1=round(d1, 4))
            metrics["scenes"][name][short(model_id)] = entry
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
