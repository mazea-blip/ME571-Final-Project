# Design

Haptic Peripheral Vision Assist: a wearable obstacle-sensing belt for peripheral vision loss.
ME571 Medical Robotics, Boston University. Team: Michael Sherris, Ben Hsu, Thanu Jayakumar, Manuel Zea.

This is the design from the project pre-proposal ([`preproposal.pdf`](../preproposal.pdf)), plus what has
changed since. Where they differ, [Changes since the pre-proposal](#changes-since-the-pre-proposal) is
current.

## Problem

Peripheral vision loss from glaucoma, retinitis pigmentosa and some strokes leaves central vision intact
but removes awareness of the sides. People collide with poles, doorframes, and people stepping in from the
left or right. The white cane sweeps only the ground directly ahead, so it misses obstacles at hip or
shoulder height to the side. Glaucoma and retinitis pigmentosa are progressive and largely irreversible,
so patients live with the loss; side collisions cause injuries and falls and erode confidence in
travelling independently.

The belt restores awareness of the sides through touch, without occupying the user's hands or hearing,
for about $51 in purchased parts.

## Approach

Three ideas set it apart:

- **Peripheral-only feedback.** It signals only the side zones that peripheral loss removes. The user's
  central vision handles the area straight ahead.
- **Complementary sensing.** A wide-angle camera finds obstacles; laser time-of-flight (ToF) sensors give
  fast, direct distance at the sides, so close-range warnings stay responsive when the camera pipeline
  runs slowly.
- **Spatial haptic encoding.** Four belt motors map one-to-one to four zones: which motor vibrates gives
  the direction, and intensity and pulse rate give how close the obstacle is.

## Zones

Four zones flank the user's central vision across the camera's 100° view.

| Zone | Camera | ToF | Motor |
|---|---|---|---|
| Left | yes | left hip sensor | Left |
| Mid-Left | yes (only sensor) | none | Mid-Left |
| Mid-Right | yes (only sensor) | none | Mid-Right |
| Right | yes | right hip sensor | Right |

Straight ahead gets no feedback. When the camera and a ToF sensor both see an obstacle in a zone, the
nearer reading wins. Each zone's nearest obstacle sets its motor's duty cycle, which rises as the obstacle
gets closer, in a continuous sense-to-actuate loop: about 3 m gives a slow pulse, 0.5 m a rapid, strong
buzz.

## Hardware

| Part | Pre-proposal spec | Notes |
|---|---|---|
| Chest camera | 720p USB, 100° lens, angled ~10° down | ELP-USB100W07M-MHV100: OV9712 sensor, 1280×720, M7 lens sold as 100° but 93° horizontal and 98° diagonal per its spec sheet, so `--hfov 93` and a focal length of about 607 px at 1280 px wide. 30 fps at 1280×720 only in MJPEG (5 fps in YUY2). |
| ToF sensors | 2× VL53L1X at the hips, angled ~30° outward, 0–4 m | Shared I2C bus; separate XSHUT pins give each a unique address at startup. A third is kept as a spare. |
| Controller | Raspberry Pi (a Pi 5 in the risk estimates) | Pocket unit, with the battery |
| Power | 5 V / 3 A, 10,000 mAh bank in a vented pocket enclosure | About 3–4 hours of use |
| Motors | 4× 3 V coin vibration motors, two on each hip, none at the front | Driven through 2N2222 transistors with flyback diodes |

Signal chain (Fig. 1 of the pre-proposal): camera and ToF sensors → Raspberry Pi (object detection and
distance mapping) → transistor drivers → vibration motors.

## Validation

- Check camera and ToF distances against a tape measure.
- Bench-test the zone-to-motor mapping.
- Walk an obstacle course blindfolded or in peripheral-restricting goggles, with and without the belt,
  and compare collisions and reaction distance.

## Risks

From the pre-proposal:

- Detection runs at about 5–10 fps on a Pi 5, so the faster ToF loop has to cover near-range side hazards.
- Camera distance error is about ±10–20%.
- Sunlight reduces ToF range.
- A 10,000 mAh bank gives roughly 3–4 hours of use.

## Changes since the pre-proposal

| | Pre-proposal | Now |
|---|---|---|
| Camera distance | YOLOv8n boxes plus a resolution and aperture calculation | Depth Anything 3 (DA3) depth maps, from the tools in `tools/`. YOLOv8n stays an option. |
| Where the camera model runs | Raspberry Pi | A laptop. The Pi handles the ToF sensors and the motors. |

On an M4 Pro laptop, DA3-BASE runs at about 18 fps at 392 px (see the [README](../README.md)), compared
with the 5–10 fps the pre-proposal expected from YOLOv8n on a Pi 5.

## Open questions

- **Metres from the camera.** DA3-SMALL, BASE and LARGE give relative depth, but the motor mapping needs
  real distances. Options: `DA3METRIC-LARGE` (needs the camera's focal length; model `5` in
  `tools/live.py`, accuracy not yet checked against a tape measure), or scaling the relative depth with
  the ToF readings in the Left and Right zones.
- **Belt-to-laptop link.** How camera frames reach the laptop and zone distances get back to the Pi isn't
  decided. Its delay adds to every camera-based warning.
- **Which ToF sensor.** The pre-proposal text names the VL53L1X (one distance per sensor); its Fig. 1
  labels the VL53L5CX (an 8×8 grid of distances per sensor).
- **Zone boundaries.** The pre-proposal doesn't give the angle each zone covers or how wide the central
  gap is. `tools/live.py` starts with equal fifths of the image width (the middle fifth is the gap) in a
  band from 35% to 65% of the image height, measured in pixels rather than angles.
