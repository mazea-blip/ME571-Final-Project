"""Live depth and 3D point cloud from your webcam with Depth Anything 3.

    uv run tools/live.py           # MacBook camera, starts on DA3-BASE at 392 px
    uv run tools/live.py --model 1 --res 280

Windows: the camera, the depth map with a speed readout, and a browser tab with the
live 3D point cloud (http://127.0.0.1:8766). The 3D view has two modes:
    live frame   the current frame in 3D, replaced every frame
    build map    keyframes accumulate into a map of the room as you move the camera,
                 using DA3's camera-pose output on [last keyframe, current frame] pairs.
                 Needs a model with a camera head (SMALL, BASE, LARGE-1.1). It drifts.

Keys (with the camera or depth window focused):
    1 2 3 4   switch model: SMALL, BASE, MONO-LARGE, LARGE-1.1 (first switch loads it)
    r         cycle input resolution 280 / 392 / 504 px
    s         toggle flicker smoothing (depth window)
    m         toggle 3D mode: live frame / build map
    c         clear the map
    q / Esc   quit

Test without a camera:  uv run tools/live.py --source footage/clip.mp4 --headless 120
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # torch + pycolmap both ship libomp
os.environ.setdefault("DA3_LOG_LEVEL", "ERROR")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import struct
import threading
import time
import warnings
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")
from depth_anything_3.api import DepthAnything3  # noqa: E402

MODELS = {
    "1": "depth-anything/DA3-SMALL",
    "2": "depth-anything/DA3-BASE",
    "3": "depth-anything/DA3MONO-LARGE",
    "4": "depth-anything/DA3-LARGE-1.1",
}
HAS_POSE = {"1", "2", "4"}  # MONO-LARGE has no camera head
RESOLUTIONS = [280, 392, 504]
PATCH = 14
STRIDE = 2               # 3D: keep every 2nd pixel in each direction
FALLBACK_HFOV = 75.0     # degrees, for models that don't predict intrinsics
MAX_KEYFRAMES = 120
KEYFRAME_ANGLE = 10.0    # degrees of rotation from the last keyframe before adding a new one
KEYFRAME_DIST = 0.15     # translation (in units of the first frame's median depth)
# Spectral_r as an OpenCV lookup table (BGR): far = blue/purple, near = red
LUT = (matplotlib.colormaps["Spectral_r"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)[:, ::-1]
CAM_WIN, DEPTH_WIN = "DA3 camera", "DA3 depth"
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV camera axes (y down, z forward) -> three.js (y up, -z forward)


def nearest_multiple(x: int, p: int) -> int:
    down = (x // p) * p
    return max(p, down + p if abs(down + p - x) <= abs(x - down) else down)


@dataclass
class Result:
    depth: np.ndarray            # (N, h, w)
    conf: np.ndarray | None      # (N, h, w)
    ext: np.ndarray | None       # (N, 3, 4) world-to-camera, view 0 = identity
    K: np.ndarray | None         # (N, 3, 3) at model resolution


class DepthEngine:
    """Loads models on demand, runs them, and turns depth into colours and 3D points."""

    def __init__(self, device: str):
        self.device = device
        self.models: dict[str, DepthAnything3] = {}
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        self.grids: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self.reset_smoothing()

    def reset_smoothing(self):
        self.prev = None
        self.range = None

    def model(self, name: str) -> DepthAnything3:
        if name not in self.models:
            self.models[name] = DepthAnything3.from_pretrained(name).to(self.device).eval()
        return self.models[name]

    def prep(self, rgb: np.ndarray, res: int) -> tuple[torch.Tensor, np.ndarray]:
        """Same resize + normalisation as DA3's own preprocessing, done on the GPU.
        Returns the model input (3, H, W) and a strided RGB image for point colours."""
        h, w = rgb.shape[:2]
        scale = res / max(h, w)
        tw, th = nearest_multiple(round(w * scale), PATCH), nearest_multiple(round(h * scale), PATCH)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1)[None].float().div_(255)
        x = F.interpolate(x, size=(th, tw), mode="bilinear", antialias=True, align_corners=False)[0]
        colours = (x[:, ::STRIDE, ::STRIDE].clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        return (x - self.mean) / self.std, colours

    def run(self, name: str, xs: list[torch.Tensor]) -> Result:
        out = self.model(name).forward(torch.stack(xs)[None], None, None, [], False, False, "first")
        n, (h, w) = len(xs), xs[0].shape[-2:]
        get = lambda k, shape: out[k].float().reshape(shape).cpu().numpy() if k in out else None  # noqa: E731
        return Result(get("depth", (n, h, w)), get("depth_conf", (n, h, w)),
                      get("extrinsics", (n, 3, 4)), get("intrinsics", (n, 3, 3)))

    def depth(self, name: str, rgb: np.ndarray, res: int) -> np.ndarray:
        return self.run(name, [self.prep(rgb, res)[0]]).depth[0]

    def colourise(self, depth: np.ndarray, smooth: bool) -> np.ndarray:
        prev, prev_range = self.prev, self.range  # read once: the GUI thread may reset these
        disp = 1.0 / np.clip(depth, 1e-6, None)
        disp /= np.median(disp)
        if smooth and prev is not None and prev.shape == disp.shape:
            # Fit this frame's inverse depth to the previous one (scale + shift) so the
            # model's per-frame scale jumps don't show up as colour flicker.
            a = disp[::4, ::4].ravel()
            A = np.stack([a, np.ones_like(a)], axis=1)
            s, t = np.linalg.lstsq(A, prev[::4, ::4].ravel(), rcond=None)[0]
            if s > 0:
                disp = s * disp + t
        self.prev = disp
        lo, hi = np.percentile(disp, [2, 98])
        if smooth and prev_range is not None:
            lo, hi = 0.85 * prev_range[0] + 0.15 * lo, 0.85 * prev_range[1] + 0.15 * hi
        self.range = (lo, hi)
        norm = np.clip((disp - lo) / max(hi - lo, 1e-6), 0, 1)
        return LUT[(norm * 255).astype(np.uint8)]

    def unproject(self, depth: np.ndarray, K: np.ndarray, conf: np.ndarray | None,
                  colours: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Depth (h, w) -> camera-frame points (OpenCV axes) and their colours, every STRIDE-th
        pixel, minus low-confidence pixels and 'flying' pixels on depth edges."""
        h, w = depth.shape
        if (h, w) not in self.grids:
            ys, xs = np.mgrid[0:h:STRIDE, 0:w:STRIDE].astype(np.float32)
            self.grids[h, w] = (xs + 0.5, ys + 0.5)
        u, v = self.grids[h, w]
        d = depth[::STRIDE, ::STRIDE].astype(np.float32)
        keep = d < 8 * np.median(d)
        # flying pixels: big depth jump within the 3x3 neighbourhood
        k = np.ones((3, 3), np.uint8)
        keep &= cv2.dilate(d, k) / np.maximum(cv2.erode(d, k), 1e-6) < 1.08
        if conf is not None:
            c = conf[::STRIDE, ::STRIDE]
            keep &= c > np.percentile(c, 15)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        z = d[keep]
        pts = np.stack([(u[keep] - cx) / fx * z, (v[keep] - cy) / fy * z, z], axis=1)
        return pts.astype(np.float32), colours[keep]


def fallback_K(h: int, w: int) -> np.ndarray:
    f = (w / 2) / np.tan(np.radians(FALLBACK_HFOV / 2))
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)


def to4(ext: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :4] = ext
    return m


def transform(pts: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return (pts @ c2w[:3, :3].T + c2w[:3, 3]).astype(np.float32)


def to_gl(pts: np.ndarray) -> np.ndarray:
    return (pts * np.array([1, -1, -1], np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Streaming points to the browser: long-polled HTTP, no extra dependencies
# ---------------------------------------------------------------------------

def pad4(b: bytes) -> bytes:
    return b + b"\0" * (-len(b) % 4)


class Hub:
    """Latest 3D state, shared between the model thread and the HTTP server threads."""

    def __init__(self):
        self.cond = threading.Condition()
        self.frame_id = 0
        self.epoch = 0
        self.live = (b"", b"", 0)
        self.pose = None
        self.chunks: list[tuple[bytes, bytes, int]] = []
        self.path: list[list[float]] = []
        self.stats: dict = {}
        self.commands: list[dict] = []

    def command(self, cmd: dict) -> None:
        with self.cond:
            self.commands.append(cmd)

    def take_commands(self) -> list[dict]:
        with self.cond:
            cmds, self.commands = self.commands, []
            return cmds

    def clear_map(self) -> None:
        with self.cond:
            self.epoch += 1
            self.chunks, self.path = [], []

    def publish(self, pts_gl, cols, pose_gl, stats, chunk=None, path_point=None) -> None:
        live = (pts_gl.tobytes(), cols.tobytes(), len(pts_gl))
        with self.cond:
            self.live, self.pose, self.stats = live, pose_gl, stats
            if chunk is not None:
                self.chunks.append((chunk[0].tobytes(), chunk[1].tobytes(), len(chunk[0])))
            if path_point is not None:
                self.path.append(path_point)
            self.frame_id += 1
            self.cond.notify_all()

    def snapshot(self, after: int, epoch: int, have: int) -> bytes:
        with self.cond:
            self.cond.wait_for(lambda: self.frame_id > after, timeout=2.0)
            reset = epoch != self.epoch
            start = 0 if reset else min(have, len(self.chunks))
            chunks = self.chunks[start:]
            header = {
                "id": self.frame_id, "epoch": self.epoch, "reset": reset, "chunkStart": start,
                "chunks": [c[2] for c in chunks], "live": self.live[2], "pose": self.pose,
                "path": self.path, "stats": self.stats,
            }
            parts = [self.live[0], pad4(self.live[1])]
            for xyz, rgb, _ in chunks:
                parts += [xyz, pad4(rgb)]
        hdr = json.dumps(header).encode()
        hdr += b" " * (-len(hdr) % 4)
        return struct.pack("<I", len(hdr)) + hdr + b"".join(parts)


def serve(hub: Hub, port: int) -> ThreadingHTTPServer:
    page = Path(__file__).resolve().parent.parent / "web" / "live3d.html"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the terminal quiet
            pass

        def send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):  # tab closed or reloaded mid-reply
                pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path in ("/", "/live3d.html"):
                self.send(200, page.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/frame":
                q = {k: int(v[0]) for k, v in parse_qs(url.query).items() if v and v[0].lstrip("-").isdigit()}
                body = hub.snapshot(q.get("after", 0), q.get("epoch", -1), q.get("have", 0))
                self.send(200, body, "application/octet-stream")
            else:
                self.send(404, b"not found", "text/plain")

        def do_POST(self):
            if urlparse(self.path).path != "/cmd":
                return self.send(404, b"not found", "text/plain")
            try:
                hub.command(json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)))))
                self.send(200, b"ok", "text/plain")
            except (ValueError, json.JSONDecodeError):
                self.send(400, b"bad command", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------------------------------------------------------------------------
# Model thread
# ---------------------------------------------------------------------------

@dataclass
class Keyframe:
    x: torch.Tensor       # model input, kept on the GPU for the next pair
    c2w: np.ndarray       # camera-to-world (OpenCV axes), world = first keyframe's camera
    depth_w: np.ndarray   # depth in world units


class Worker(threading.Thread):
    """Runs the model on the newest camera frame, as fast as it can, off the GUI thread."""

    def __init__(self, engine: DepthEngine, hub: Hub, model_key: str, res: int, smooth: bool):
        super().__init__(daemon=True)
        self.engine, self.hub = engine, hub
        self.model_key, self.res, self.smooth = model_key, res, smooth
        self.mode = "live"
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.result = None         # (depth BGR image, frame_id)
        self.status = "starting"
        self.note = ""
        self.model_ms = None
        self.fps = None
        self.running = True
        self.reset_map()

    def reset_map(self):
        self.kf: Keyframe | None = None
        self.n_keyframes = 0
        self.K_ema = None
        self.hub.clear_map()

    def submit(self, rgb: np.ndarray) -> None:
        with self.lock:
            self.frame, self.frame_id = rgb, self.frame_id + 1

    def apply_commands(self) -> None:
        for cmd in self.hub.take_commands():
            if cmd.get("mode") in ("live", "map") and cmd["mode"] != self.mode:
                self.mode = cmd["mode"]
                self.reset_map()
            if cmd.get("toggle_mode"):
                self.mode = "map" if self.mode == "live" else "live"
                self.reset_map()
            if cmd.get("clear"):
                self.reset_map()

    def live_step(self, name, x, colours):
        r = self.engine.run(name, [x])
        depth = r.depth[0]
        h, w = depth.shape
        K = r.K[0] if r.K is not None else fallback_K(h, w)
        # intrinsics are predicted per frame; smooth the focal length so the cloud doesn't breathe
        self.K_ema = K if self.K_ema is None else 0.9 * self.K_ema + 0.1 * K
        d = depth / np.median(depth)
        pts, cols = self.engine.unproject(d, self.K_ema, None if r.conf is None else r.conf[0], colours)
        return depth, to_gl(pts), cols, np.eye(4).tolist(), None, None  # camera at the origin

    def map_step(self, name, x, colours):
        eng = self.engine
        if self.kf is None:  # first keyframe defines the world frame and unit scale
            r = eng.run(name, [x])
            depth_w = r.depth[0] / np.median(r.depth[0])
            c2w = np.eye(4)
            self.kf, self.n_keyframes = Keyframe(x, c2w, depth_w), 1
            pts, cols = eng.unproject(depth_w, r.K[0], r.conf[0], colours)
            pts_gl = to_gl(pts)
            return r.depth[0], pts_gl, cols, (CV_TO_GL @ c2w @ CV_TO_GL).tolist(), (pts_gl, cols), [0.0, 0.0, 0.0]

        r = eng.run(name, [self.kf.x, x])
        valid = r.conf[0] > np.percentile(r.conf[0], 30)
        s = float(np.median(self.kf.depth_w[valid] / np.maximum(r.depth[0][valid], 1e-6)))
        T = to4(r.ext[0]) @ np.linalg.inv(to4(r.ext[1]))  # current camera -> keyframe camera
        T[:3, 3] *= s
        if not np.isfinite(s) or not 0.2 < s < 5 or np.linalg.norm(T[:3, 3]) > 2.0:
            self.note = "lost track: hold still or press c"
            return r.depth[1], None, None, None, None, None
        self.note = ""
        c2w = self.kf.c2w @ T
        depth_w = r.depth[1] * s
        pts, cols = eng.unproject(depth_w, r.K[1], r.conf[1], colours)
        pts_w = to_gl(transform(pts, c2w))
        angle = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        chunk = path_point = None
        if (angle > KEYFRAME_ANGLE or np.linalg.norm(T[:3, 3]) > KEYFRAME_DIST):
            if self.n_keyframes < MAX_KEYFRAMES:
                self.kf = Keyframe(x, c2w, depth_w)
                self.n_keyframes += 1
                chunk = (pts_w, cols)
                path_point = to_gl(c2w[None, :3, 3])[0].tolist()
            else:
                self.note = f"map full ({MAX_KEYFRAMES} keyframes): press c to clear"
        return r.depth[1], pts_w, cols, (CV_TO_GL @ c2w @ CV_TO_GL).tolist(), chunk, path_point

    def run(self) -> None:
        last_id, last_done, active = 0, None, None
        while self.running:
            self.apply_commands()
            with self.lock:
                rgb, fid = self.frame, self.frame_id
            if rgb is None or fid == last_id:
                time.sleep(0.002)
                continue
            key, res, smooth = self.model_key, self.res, self.smooth
            name = MODELS[key]
            mode = self.mode if key in HAS_POSE else "live"
            try:
                if (key, res) != active:
                    if name not in self.engine.models:
                        self.status = f"loading {name.split('/')[-1]}..."
                    self.engine.model(name)
                    self.engine.reset_smoothing()
                    self.reset_map()
                    active, last_done, self.fps, self.model_ms = (key, res), None, None, None
                    self.status = "warming up..."
                t0 = time.perf_counter()
                x, colours = self.engine.prep(rgb, res)
                step = self.map_step if mode == "map" else self.live_step
                depth, pts, cols, pose, chunk, path_point = step(name, x, colours)
                ms = (time.perf_counter() - t0) * 1000
                vis = self.engine.colourise(depth, smooth)
            except Exception as e:  # show it in the window instead of silently freezing
                self.status = f"error: {type(e).__name__}: {e}"[:90]
                print(self.status, flush=True)
                last_id, active = fid, None
                time.sleep(0.5)
                continue
            now = time.perf_counter()
            if last_done is not None:
                inst = 1.0 / max(now - last_done, 1e-6)
                self.fps = inst if self.fps is None else 0.9 * self.fps + 0.1 * inst
                self.model_ms = ms if self.model_ms is None else 0.9 * self.model_ms + 0.1 * ms
            last_done, last_id = now, fid
            self.status = "running"
            if self.mode == "map" and key not in HAS_POSE:
                self.note = "map mode needs SMALL, BASE or LARGE-1.1 (keys 1, 2, 4)"
            if pts is not None:
                self.hub.publish(pts, cols, pose, {
                    "model": name.split("/")[-1], "res": res, "mode": mode, "requested": self.mode,
                    "fps": round(self.fps or 0, 1), "ms": round(self.model_ms or ms),
                    "keyframes": self.n_keyframes if mode == "map" else 0, "note": self.note,
                }, chunk, path_point)
            with self.lock:
                self.result = (vis, fid)


# ---------------------------------------------------------------------------
# Camera, GUI and main
# ---------------------------------------------------------------------------

def draw_hud(img: np.ndarray, lines: list[str], footer: str) -> None:
    font, scale, th = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
    pad, lh = 10, 24
    width = max(cv2.getTextSize(t, font, scale, th)[0][0] for t in lines) + 2 * pad
    box = img[:pad + lh * len(lines), :width]
    box[:] = (box * 0.35).astype(np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (pad, pad + 16 + i * lh), font, scale, (255, 255, 255), th, cv2.LINE_AA)
    (fw, fh), _ = cv2.getTextSize(footer, font, 0.5, 1)
    y0 = img.shape[0] - fh - 2 * pad
    strip = img[y0:, : fw + 2 * pad]
    strip[:] = (strip * 0.35).astype(np.uint8)
    cv2.putText(img, footer, (pad, img.shape[0] - pad), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


def open_source(src: str, width: int, height: int, wait_s: float = 30) -> cv2.VideoCapture:
    if not src.isdigit():
        cap = cv2.VideoCapture(src)
        ok, _ = cap.read() if cap.isOpened() else (False, None)
    else:
        # On first use macOS shows a camera permission dialog, but OpenCV fails immediately
        # instead of waiting for the answer. Keep retrying while the dialog is up.
        deadline, told = time.time() + wait_s, False
        while True:
            cap = cv2.VideoCapture(int(src), cv2.CAP_AVFOUNDATION)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read() if cap.isOpened() else (False, None)
            if ok or time.time() > deadline:
                break
            cap.release()
            if not told:
                print("Waiting for camera access: click Allow if macOS asks "
                      f"(retrying for {wait_s:.0f} s)...", flush=True)
                told = True
            time.sleep(1)
    if not ok:
        raise SystemExit(
            f"Could not read from camera/source {src!r}.\n"
            "If this is the webcam: allow camera access for the app you ran this from "
            "(System Settings > Privacy & Security > Camera), then run it again."
        )
    return cap


def headless(args, engine: DepthEngine) -> None:
    """Process N frames with no windows and print timing, for testing without a camera."""
    cap = open_source(args.source, args.width, args.height)
    hub = Hub()
    worker = Worker(engine, hub, args.model, args.res, args.smooth)
    worker.mode = args.mode
    name = MODELS[args.model]
    times, n_pts = [], []
    for i in range(args.headless):
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(bgr, (args.width, args.height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        x, colours = engine.prep(rgb, args.res)
        step = worker.map_step if args.mode == "map" and args.model in HAS_POSE else worker.live_step
        depth, pts, cols, pose, chunk, path_point = step(name, x, colours)
        engine.colourise(depth, args.smooth)
        if pts is not None:
            hub.publish(pts, cols, pose, {}, chunk, path_point)
            n_pts.append(len(pts))
        if i >= 5:
            times.append(time.perf_counter() - t0)
    payload = hub.snapshot(0, -1, 0)
    print(f"{name.split('/')[-1]} @ {args.res}px, 3D mode {args.mode}: median {np.median(times) * 1000:.1f} ms "
          f"= {1 / np.median(times):.1f} fps; {int(np.median(n_pts))} points/frame; "
          f"{worker.n_keyframes} keyframes, {len(hub.chunks)} map chunks; payload {len(payload) / 1e6:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index or video file (default: 0, the built-in camera)")
    p.add_argument("--model", default="2", choices=list(MODELS), help="1 SMALL, 2 BASE, 3 MONO-LARGE, 4 LARGE-1.1")
    p.add_argument("--res", type=int, default=392, choices=RESOLUTIONS)
    p.add_argument("--mode", default="live", choices=["live", "map"], help="starting 3D mode")
    p.add_argument("--no-smooth", dest="smooth", action="store_false", help="start with smoothing off")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", dest="mirror", action="store_false", help="don't mirror the camera image")
    p.add_argument("--no-3d", dest="web", action="store_false", help="don't start the browser 3D view")
    p.add_argument("--no-browser", dest="open_browser", action="store_false", help="serve the 3D view but don't open a tab")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--headless", type=int, metavar="N", help="process N frames without windows and print timing")
    args = p.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    engine = DepthEngine(device)
    print(f"loading {MODELS[args.model]} on {device}", flush=True)
    engine.model(MODELS[args.model])
    if args.headless:
        return headless(args, engine)

    cap = open_source(args.source, args.width, args.height)
    hub = Hub()
    worker = Worker(engine, hub, args.model, args.res, args.smooth)
    worker.mode = args.mode
    worker.start()
    if args.web:
        serve(hub, args.port)
        url = f"http://127.0.0.1:{args.port}/"
        print(f"3D view: {url}", flush=True)
        if args.open_browser:
            webbrowser.open(url)
    cv2.namedWindow(CAM_WIN, cv2.WINDOW_NORMAL)
    cv2.namedWindow(DEPTH_WIN, cv2.WINDOW_NORMAL)
    placed = False
    depth_img, shown_id = None, -1
    last_log = time.perf_counter()
    footer = "1-4 model  r res  s smooth  m 3D mode  c clear map  q quit"
    print("Keys" + __doc__.split("Keys")[1].split("Test without")[0].rstrip(), flush=True)

    while True:
        ok, bgr = cap.read()
        if not ok:
            if not args.source.isdigit():  # loop video files
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("camera stopped delivering frames")
            break
        if bgr.shape[1] != args.width:
            bgr = cv2.resize(bgr, (args.width, round(bgr.shape[0] * args.width / bgr.shape[1])), interpolation=cv2.INTER_AREA)
        if args.mirror and args.source.isdigit():
            bgr = cv2.flip(bgr, 1)
        worker.submit(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cv2.imshow(CAM_WIN, bgr)

        with worker.lock:
            result = worker.result
        if result is not None and result[1] != shown_id:
            depth_img = cv2.resize(result[0], (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
            shown_id = result[1]
        canvas = depth_img.copy() if depth_img is not None else np.zeros_like(bgr)
        name = MODELS[worker.model_key].split("/")[-1]
        speed = (f"{worker.fps:.1f} fps   model {worker.model_ms:.0f} ms"
                 if worker.fps is not None and worker.status == "running" else worker.status)
        mode = (f"3D: map, {worker.n_keyframes} keyframes" if worker.mode == "map" else "3D: live frame")
        lines = [f"{name}   {worker.res} px   smoothing {'on' if worker.smooth else 'off'}", speed, mode]
        if worker.note:
            lines.append(worker.note)
        draw_hud(canvas, lines, footer)
        cv2.imshow(DEPTH_WIN, canvas)
        if time.perf_counter() - last_log > 5:
            print(f"{name} {worker.res}px smoothing {'on' if worker.smooth else 'off'} {mode}: {speed}", flush=True)
            last_log = time.perf_counter()

        if not placed:  # put the windows next to each other once, then leave them where the user drags them
            w = bgr.shape[1] // 2
            for win, x in ((CAM_WIN, 0), (DEPTH_WIN, w + 12)):
                cv2.resizeWindow(win, w, bgr.shape[0] // 2)
                cv2.moveWindow(win, x, 40)
            placed = True

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if chr(key) in MODELS:
            worker.model_key = chr(key)
        elif key == ord("r"):
            worker.res = RESOLUTIONS[(RESOLUTIONS.index(worker.res) + 1) % len(RESOLUTIONS)]
        elif key == ord("s"):
            worker.smooth = not worker.smooth
            engine.reset_smoothing()
        elif key == ord("m"):
            hub.command({"toggle_mode": True})
        elif key == ord("c"):
            hub.command({"clear": True})
        try:
            if any(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in (CAM_WIN, DEPTH_WIN)):
                break
        except cv2.error:
            break

    worker.running = False
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
