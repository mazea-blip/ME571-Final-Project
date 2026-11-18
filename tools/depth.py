"""Run Depth Anything 3 on photos and videos (Apple Silicon / MPS friendly).

    uv run tools/depth.py footage/Photos-1-001/     # every image and video in a folder
    uv run tools/depth.py footage/clip.mp4 --fps 5 --glb   # depth video + 3D point cloud
    uv run tools/depth.py footage/photo.jpg --model depth-anything/DA3MONO-LARGE

Images are processed one at a time. A video is sampled at --fps and all sampled
frames go through the model in a single joint pass, so depth is on one shared
scale across the clip (no flicker) and DA3 also recovers a camera pose per frame.
"""

import os

# torch and pycolmap (a DA3 dependency, only used for COLMAP export) each bundle
# their own libomp.dylib on macOS; without this the process aborts on import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")  # DA3 logs every step at INFO

import argparse
import time
import warnings
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

warnings.filterwarnings("ignore")
from depth_anything_3.api import DepthAnything3  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
CMAP = matplotlib.colormaps["Spectral_r"]


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def colorize(depth: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Depth (N,H,W) -> RGB uint8, using inverse depth so near = warm, far = cool."""
    disp = 1.0 / np.clip(depth, 1e-6, None)
    norm = np.clip((disp - lo) / max(hi - lo, 1e-8), 0, 1)
    return (CMAP(norm)[..., :3] * 255).astype(np.uint8)


def disparity_range(depth: np.ndarray, conf: np.ndarray | None) -> tuple[float, float]:
    disp = 1.0 / np.clip(depth, 1e-6, None)
    if conf is not None:  # ignore the least confident pixels (sky, edges) when picking the range
        disp = disp[conf > np.percentile(conf, 20)]
    return float(np.percentile(disp, 2)), float(np.percentile(disp, 98))


def even(img: np.ndarray) -> np.ndarray:
    return img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]  # H.264 needs even dims


def match_size(depth_rgb: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Upscale a model-resolution depth map to the input frame's size so they line up."""
    h, w = rgb.shape[:2]
    return cv2.resize(depth_rgb, (w, h), interpolation=cv2.INTER_LINEAR)


def write_video(path: Path, frames, fps: float) -> None:
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for f in frames:
            w.append_data(even(f))


def load_rgb(path: Path, max_side: int) -> np.ndarray:
    from PIL import Image, ImageOps

    if path.suffix.lower() == ".heic":
        import pillow_heif

        pillow_heif.register_heif_opener()
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(img)


def read_video(path: Path, fps: float, max_frames: int, max_side: int):
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(src_fps / fps))
    wanted = list(range(0, total, step))
    if len(wanted) > max_frames:
        print(f"  {len(wanted)} frames at {fps} fps is over --max-frames {max_frames}; sampling evenly")
        wanted = np.linspace(0, total - 1, max_frames).round().astype(int).tolist()
    wanted_set, frames, idx = set(wanted), [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx in wanted_set:
            _, f = cap.retrieve()
            h, w = f.shape[:2]
            s = min(1.0, max_side / max(h, w))
            if s < 1:
                f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    out_fps = src_fps / (np.mean(np.diff(wanted)) if len(wanted) > 1 else 1)
    return frames, out_fps


def save_npz(path: Path, pred) -> None:
    np.savez_compressed(
        path,
        depth=pred.depth.astype(np.float16),
        conf=None if pred.conf is None else pred.conf.astype(np.float16),
        intrinsics=pred.intrinsics,
        extrinsics=pred.extrinsics,
    )


def run_image(model, path: Path, out_dir: Path, args) -> None:
    rgb = load_rgb(path, args.max_side)
    t = time.time()
    pred = model.inference([rgb], process_res=args.res)
    print(f"  inference {time.time() - t:.2f}s, depth {pred.depth.shape[1:]}")
    lo, hi = disparity_range(pred.depth, pred.conf)
    imageio.imwrite(out_dir / f"{path.stem}_depth.png", match_size(colorize(pred.depth, lo, hi)[0], rgb))
    save_npz(out_dir / f"{path.stem}_depth.npz", pred)
    print(f"  -> {out_dir / (path.stem + '_depth.png')}")


def run_video(model, path: Path, out_dir: Path, args) -> None:
    frames, out_fps = read_video(path, args.fps, args.max_frames, args.max_side)
    print(f"  {len(frames)} frames ({frames[0].shape[1]}x{frames[0].shape[0]}), one joint pass")
    clip_dir = out_dir / path.stem
    clip_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    pred = model.inference(
        frames,
        process_res=args.res,
        export_dir=str(clip_dir) if args.glb else None,
        export_format="glb",
        conf_thresh_percentile=args.conf_percentile,
        # DA3's GLB export also dumps per-frame side-by-side JPEGs by default; skip them.
        export_kwargs={"glb": {"export_depth_vis": False}},
    )
    print(f"  inference {time.time() - t:.1f}s ({(time.time() - t) / len(frames) * 1000:.0f} ms/frame)")
    lo, hi = disparity_range(pred.depth, pred.conf)
    depth_rgb = colorize(pred.depth, lo, hi)
    # Two separate videos with identical frames, fps and size, so they line up when compared.
    write_video(clip_dir / "depth.mp4", (match_size(d, rgb) for rgb, d in zip(frames, depth_rgb)), out_fps)
    write_video(clip_dir / "rgb.mp4", frames, out_fps)
    save_npz(clip_dir / "depth.npz", pred)
    for name in ["depth.mp4", "rgb.mp4"] + (["scene.glb"] if args.glb else []):
        print(f"  -> {clip_dir / name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", type=Path, help="image/video files or folders")
    p.add_argument("--model", default="depth-anything/DA3-LARGE-1.1",
                   help="Hugging Face model id (default: %(default)s)")
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--res", type=int, default=504, help="model input size, long side (default: %(default)s)")
    p.add_argument("--fps", type=float, default=5, help="video sampling rate (default: %(default)s)")
    p.add_argument("--max-frames", type=int, default=96,
                   help="cap on frames per video pass; ~14 GB at 504px on MPS (default: %(default)s)")
    p.add_argument("--max-side", type=int, default=1280, help="downscale inputs to this before inference")
    p.add_argument("--glb", action="store_true", help="also export a 3D point cloud + cameras (videos)")
    p.add_argument("--conf-percentile", type=float, default=40.0,
                   help="GLB: drop this percentile of least-confident points (default: %(default)s)")
    args = p.parse_args()

    files = []
    for inp in args.inputs:
        files += sorted(f for f in inp.iterdir() if f.is_file()) if inp.is_dir() else [inp]
    files = [f for f in files if f.suffix.lower() in IMAGE_EXTS | VIDEO_EXTS]
    if not files:
        raise SystemExit("no images or videos found")

    device = pick_device(args.device)
    print(f"loading {args.model} on {device}")
    model = DepthAnything3.from_pretrained(args.model).to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(f"{f}")
        if f.suffix.lower() in VIDEO_EXTS:
            run_video(model, f, args.out, args)
        else:
            run_image(model, f, args.out, args)


if __name__ == "__main__":
    main()
