# Haptic Peripheral Vision Assist

ME571 Medical Robotics final project: a wearable belt that warns people with peripheral vision loss about
obstacles at their sides through vibration. The design, and what has changed since the pre-proposal
(`preproposal.pdf`), is in [`docs/design.md`](docs/design.md).

This repo holds the vision side. [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
(DA3) turns chest-camera frames into depth maps that give the distance to obstacles in each peripheral
zone. YOLOv8n detection from the preproposal stays an option on top of it. A laptop runs the model; the
belt's Raspberry Pi handles the ToF sensors and the motors.

## Setup

```sh
uv sync
```

Python 3.12, managed by [uv](https://docs.astral.sh/uv/). DA3 is pinned to an upstream commit in
`pyproject.toml`. Three things make it run on macOS:

- `xformers` is limited to Linux. It has no macOS wheels, and DA3 falls back to plain PyTorch without it.
- `addict` is added. DA3 imports it but doesn't declare it.
- The scripts set `KMP_DUPLICATE_LIB_OK=TRUE`, because PyTorch and pycolmap each ship their own OpenMP
  library and the process aborts when both load.

Tested on Apple Silicon (MPS). The scripts use CUDA or the CPU when there is no MPS, but that is untested.

Models download from Hugging Face on first use. Put photos and clips in `footage/`; results go to
`outputs/`. Both folders are ignored by git, so the lab test footage that `compare.py` uses
(`footage/Photos-1-001/`, about 190 MB) has to be copied in by hand.

## Layout

| Path | What it is |
|---|---|
| `tools/depth.py` | Depth maps for photos; depth videos, point clouds and camera paths for short clips |
| `tools/compare.py` | Runs four DA3 models on the same frames and measures how much they disagree |
| `tools/live.py` + `web/live3d.html` | Webcam depth, live, with the belt's motor zones and a browser 3D view |

## Live webcam

```sh
uv run tools/live.py
uv run tools/live.py --source footage/Photos-1-001/PXL_20261116_161842089.mp4 --headless 120   # timing, no camera
```

Opens a camera window, a depth window and the 3D view at http://127.0.0.1:8766. Keys: `1`–`4` model,
`r` input size, `s` flicker smoothing, `m` 3D mode (live frame / build map), `c` clear the map, `z` belt
zones, `q` quit. `--source` takes a camera index (default `0`, the built-in camera) or a video file. On an
M4 Pro, DA3-BASE runs at about 18 fps at 392 px and DA3-SMALL at about 30.

Both windows show the belt's four motor zones (Left, Mid-Left, Mid-Right, Right) in a band at waist
height. Each zone covers a fifth of the image width, and the middle fifth is left out. A zone turns red
when its nearest 10% of pixels is closer than `--zone-near` (default 0.5) times the frame's median depth;
the number shown in each zone is that ratio. The depth is relative, so this is a placeholder until the
camera gives metres. Webcams are mirrored by default and the zones follow the camera's own left and
right; pass `--no-mirror` for a chest camera. `--headless` also prints how often each zone alerted.

## Depth maps and short clips

```sh
uv run tools/depth.py footage/photo.jpg
uv run tools/depth.py footage/clip.mp4 --fps 5 --glb
```

A clip's sampled frames all go through the model in one pass, so depth is on one scale across the clip
(no flicker) and DA3 estimates a camera position per frame. One pass holds about 96 frames with
DA3-LARGE-1.1 on 24 GB; longer clips are thinned to fit. Writes `depth.mp4`, the matching `rgb.mp4`,
`depth.npz` and, with `--glb`, `scene.glb`.

## Comparing models

```sh
uv run tools/compare.py
```

Runs DA3-SMALL, DA3-BASE, DA3MONO-LARGE and DA3-LARGE-1.1 on a lab photo and one frame each from the
tables and hallway clips. Each model is compared with DA3-LARGE-1.1 after a scale-and-shift fit, because
these models predict relative depth. That measures agreement, not accuracy: there is no ground truth.
Writes depth maps, disagreement overlays and `metrics.json` to `outputs/compare/`.

## Models

| Model | Params | Notes | License |
|---|---|---|---|
| DA3-SMALL | 0.08B | Fastest; real time on the webcam | Apache 2.0 |
| DA3-BASE | 0.12B | Closest to LARGE in our comparison at under half the cost | Apache 2.0 |
| DA3MONO-LARGE | 0.35B | Single image only, no camera estimate | Apache 2.0 |
| DA3-LARGE-1.1 | 0.35B | Default for `depth.py` | CC BY-NC 4.0 per the DA3 README (Hugging Face lists Apache 2.0) |
