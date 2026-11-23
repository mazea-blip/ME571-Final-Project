# Raspberry Pi

The belt's Raspberry Pi: how it is set up, how its camera reaches a laptop, and what was measured on
2026-11-23. The depth model runs on the laptop ([design](design.md#changes-since-the-pre-proposal)); the Pi
streams the camera and will handle the ToF sensors and the motors. Running DA3 on the Pi was tried and
dropped, see [DA3 on the Pi](#da3-on-the-pi-tried-and-dropped).

## The board

| | |
|---|---|
| Board | Raspberry Pi 5 Model B Rev 1.1, 16 GB |
| OS | Ubuntu Server 24.04.5 LTS (no desktop), kernel 6.8.0-1068-raspi |
| Bootloader | 2025-12-08 build |
| Storage | 58 GB microSD |
| Network | Wi-Fi, `raspberrypi.local` (mDNS), user `hben09` |
| Interfaces | I2C (`/dev/i2c-1`) and SPI (`/dev/spidev0.*`) on in `/boot/firmware/config.txt`; UART on |

All packages, snaps and the bootloader were updated on 2026-11-23 (265 packages upgraded, 22 new).
Ubuntu 26.04.1 LTS is offered by `do-release-upgrade` but was not installed.

`hben09` is in the `gpio`, `i2c`, `spi`, `dialout` and `video` groups. On this image `/dev/i2c-1`,
`/dev/spidev0.*` and `/dev/gpiochip4` (the 40-pin header's GPIO on a Pi 5) belong to `dialout`, so the
sensor and motor code can run without `sudo`. Other `gpiochip`s are root-only.

## Power

**The Pi needs a 5 V / 5 A USB-C supply** (the official 27 W one). With less it browns out:

- Powered from a laptop's USB-C port, it logged an undervoltage warning about every 25 s and then lost
  power: the journal ends at 11:38:12 on `hwmon hwmon3: Undervoltage detected!`, with no shutdown.
- On a wall phone adapter, the 5 V rail sagged to 4.73–4.88 V under load, with 6 undervoltage warnings in
  the first minutes, and the Pi throttled its CPU (`throttled=0x50005`).
- With a supply that doesn't negotiate 5 A, the firmware also limits the USB ports to 600 mA in total
  (`usb_max_current_enable=0`), and the camera draws from that.

The design's 5 V / 3 A power bank is below the Pi 5's rating, so check the belt's supply under full load
before relying on it. To check power:

```sh
vcgencmd get_throttled            # 0x0 = no problems since boot
vcgencmd pmic_read_adc EXT5V_V    # the 5 V rail: 5.08 V idle here; the warnings came with it at 4.7-4.9 V
journalctl -b -k | grep -c "Undervoltage detected"
journalctl -b -1 | tail           # the previous boot's last lines, after an unexpected restart
```

`get_throttled` bits: `0x1` undervoltage now, `0x2` frequency capped now, `0x4` throttled now, `0x8` soft
temperature limit now; `0x10000`, `0x20000`, `0x40000` and `0x80000` mean the same things happened since
boot.

## Getting in from a laptop

The Pi accepts SSH keys as well as its password. Give each laptop its own key:

```sh
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_pi -N "" -C mac-to-pi     # on the laptop
cat ~/.ssh/id_ed25519_pi.pub                                          # copy this line
# then on the Pi, with that line in place of <pubkey>:
mkdir -p ~/.ssh && echo '<pubkey>' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys
```

```sh
ssh -i ~/.ssh/id_ed25519_pi hben09@raspberrypi.local
```

The login password isn't kept in this repo. `raspberrypi.local` sometimes fails to resolve on the Mac;
then use the IP address (`hostname -I` on the Pi; it was `10.44.219.98`, assigned by DHCP so it can
change) and keep the host key check with `ssh -o HostKeyAlias=raspberrypi.local hben09@<ip>`.

If your terminal is Ghostty, the Pi doesn't know its terminal type (`'xterm-ghostty': unknown terminal
type`, which breaks `clear`, `htop` and `vim`). Install it once from the Mac:
`infocmp -x xterm-ghostty | ssh hben09@raspberrypi.local -- tic -x -`.

## Camera

The ELP chest camera (see [design](design.md#hardware)) shows up as `HD USB Camera`, USB ID `32e4:9601`,
on `/dev/video0` (UVC driver). `/dev/video19`–`37` are the Pi 5's own image and video blocks, not cameras.

| Format | Sizes and frame rates offered | Measured |
|---|---|---|
| MJPEG | 1280×720, 960×720 at 30/25/15 fps; 800×600 at 30; 640×360 at 60; 640×480, 352×288, 320×240 at 30/15 | 1280×720: 30.0 fps, 165–250 KB per frame. 640×360: 50–70 KB per frame |
| YUYV (raw) | 1280×720, 960×720 at 10/5 fps; 800×600 at 15; 640×360 at 31; 640×480, 352×288, 320×240 at 30/15 | 1280×720: 10.0 fps |

640×360 is the full 1280×720 view scaled down, not a crop, so `--hfov 93` holds at both sizes.

Two of the camera's defaults are wrong for this project. Both reset when the camera is replugged or the Pi
restarts, so set them each time:

```sh
v4l2-ctl -d /dev/video0 -c power_line_frequency=2        # 60 Hz mains (default 1 = 50 Hz) against light flicker
v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0  # hold 30 fps; on, it dropped to 21 fps indoors
```

`v4l2-ctl --list-formats-ext` and `--list-ctrls` (package `v4l-utils`) show the rest.

## Streaming the camera to the laptop

[ustreamer](https://github.com/pikvm/ustreamer) (`apt install ustreamer`) serves the camera's own MJPEG
frames over HTTP without re-encoding them. It listens only on the Pi itself, and an SSH tunnel carries it
to the laptop, so nobody else on the network can open the camera. Nothing starts at boot, so after every
restart:

```sh
# on the Pi
v4l2-ctl -d /dev/video0 -c power_line_frequency=2 -c exposure_dynamic_framerate=0
setsid nohup ustreamer --device=/dev/video0 --format=MJPEG --resolution=640x360 --desired-fps=30 \
  --encoder=HW --host=127.0.0.1 --port=8080 --persistent > /tmp/ustreamer.log 2>&1 < /dev/null &

# on the laptop: keep this running, then use http://127.0.0.1:8080/stream
ssh -N -i ~/.ssh/id_ed25519_pi -L 8080:127.0.0.1:8080 hben09@raspberrypi.local
```

`/stream` is the video, `/snapshot` one JPEG and `/state` a JSON status (`online`, `captured_fps`, number
of viewers). ustreamer renames its process to `main`, so find it with `pgrep -f '^ustreamer'`, not
`pgrep ustreamer`. A single `Persistent device timeout (unplugged)` line right after it starts is the
camera warming up and can be ignored.

### Depth on the laptop from the Pi's camera

```sh
uv run tools/live.py --source http://127.0.0.1:8080/stream --model 5 --hfov 93
```

A URL source goes straight to OpenCV and is never mirrored, so `--no-mirror` isn't needed. On the M4 Pro
it ran at 9.6 fps with DA3METRIC-LARGE at 392 px (99 ms per frame), about 11 fps with DA3-BASE in 3D map
mode, and about 20 fps with DA3-SMALL at 280 px. If the stream drops, `live.py` treats the URL like a
finished video file and keeps trying to rewind it, and the windows freeze: quit with `q` and start it again.

### Delay

How long a frame takes from the Pi grabbing it to the laptop having all of it. ustreamer stamps each
frame with the Pi's clock (`/stream?extra_headers=1`, header `X-Timestamp`); the laptop compared that
with its own receive time, after measuring the offset between the two clocks over SSH. It leaves out the
camera's own exposure and encoding before the frame reaches the Pi.

| Stream | Viewers | Data received | Frames received | Median delay | Trend over 8–15 s |
|---|---|---|---|---|---|
| 1280×720, SSH tunnel | 2 | 26 Mbit/s | 16 fps | 1.35 s | 0.66 s → 1.9 s, growing |
| 1280×720, SSH tunnel | 1 | 37.5 Mbit/s | 23 fps | 0.17 s | 55 ms → 1.5 s, growing |
| 640×360, SSH tunnel | 1 | 9–15 Mbit/s | 21–37 fps | 60–64 ms | steady (46–122 ms) |
| 640×360, SSH tunnel | 2 | 9.6 Mbit/s | 22 fps | 300 ms | 266 → 352 ms |
| 640×360, direct HTTP | 1 | 10.3 Mbit/s | 22 fps | 69 ms | steady |
| 640×360, direct HTTP | 2 | 7.0 Mbit/s | 16 fps | 329 ms | 188 → 291 ms |

- **Wi-Fi is the limit.** The link between the Pi and the laptop carried between about 10 and 40 Mbit/s,
  varying from minute to minute. 1280×720 MJPEG needs about 47 Mbit/s, so frames pile up in SSH's buffer
  (about 2 MB) and the delay grows without limit.
- **Use 640×360 and one viewer.** That holds about 65 ms. Every extra viewer, such as a browser tab left
  open next to `live.py`, shares the link and raises the delay for all of them.
- **The tunnel costs nothing measurable.** Serving the stream straight to the network wasn't faster, so
  the stream stays private behind SSH.

## DA3 on the Pi (tried and dropped)

DA3-SMALL at 280 px took about 0.93 s per frame on the Pi's CPU (1.1 fps), compared with 40 ms on the
M4 Pro. Even before counting the slowdown from undervoltage, that is too slow for walking-speed warnings,
so the model stays on the laptop and the ToF sensors cover close range.

What it took, for the record:

- **The project's `uv.lock` doesn't install on the Pi.** `pycolmap` 4.2.1 and `torchvision` 0.29.2 have no
  Linux arm64 wheels in it, and `torch` is locked to its CUDA build on all of Linux, which would pull
  several GB of NVIDIA libraries. A separate environment, `~/me571-venv` on the Pi (1.3 GB), outside the
  repo so `uv run` can't replace it, has the CPU build of `torch` 2.15.0 and `torchvision` 0.29.2 from
  `https://download.pytorch.org/whl/cpu`, DA3 at the pinned commit installed with `--no-deps`, and the
  packages DA3 imports at runtime, pinned to the Mac's versions.
- `moviepy` must be 1.0.3: DA3 imports `moviepy.editor`, which 2.x removed.
- `pycolmap` has no wheel for Python 3.12 on arm64. DA3 imports it only to export COLMAP files, so an empty
  placeholder `pycolmap.py` in the environment lets it load.
- **DA3 is slow on any CPU unless it runs in float32.** DA3's `forward()` always runs under
  `torch.autocast` with float16 (bfloat16 only on CUDA cards that support it). On the Pi's CPU that took
  5–26 s per frame; calling the network directly, `model.model(images, None, None, [], False, False,
  "first")` under `torch.inference_mode()`, brought it to 0.93 s. This matters to anyone who runs
  `live.py` on a laptop without a usable GPU.

## What's on the Pi

- `~/ME571-Final-Project`: a clone of this repo.
- `~/me571-venv`: the DA3 environment above, no longer used.
- `uv` in `~/.local/bin`; `ustreamer` and `v4l-utils` from apt.
- Nothing runs at boot apart from the stock Ubuntu services: the camera settings and ustreamer above have
  to be started by hand after each restart.
