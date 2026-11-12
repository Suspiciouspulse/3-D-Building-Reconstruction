"""Shared configuration and helpers for the scene3d pipeline.

Every number that is an ASSUMPTION rather than a measurement is defined here,
so it can be changed in one place and the whole pipeline rerun.
"""
import os
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent

# Which image the OUTLINES are taken from:
#   "original"  input.png, bicubic-upscaled x4 (adds no detail)
#   "enhanced"  input_enhanced.png, an AI x4 upscale of input.png. Reduced back to
#               292 x 301 px it matches the original (correlation 0.987, offset under
#               0.15 px), so it is geometrically faithful, but detail finer than one
#               original pixel was inferred by the upscaler and cannot be verified.
# Shadows and colours are always read from the original, whichever is chosen here.
SOURCE = os.environ.get("SCENE_SOURCE", "enhanced")
WORK = HERE / ("work" if SOURCE == "original" else "work_enhanced")
OUT = HERE / ("output" if SOURCE == "original" else "output_enhanced")
WORK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- assumptions
# Ground sample distance of input.png. ESTIMATED, not from metadata:
#   * parked cars measure about 2.5-3 px long (a car is about 4.5 m)  -> 1.5-1.8 m/px
#   * a 1024 px tile at 0.5 m/px shown at 292 px wide                 -> 1.75 m/px
GSD_M = 1.75
GSD_REL_ERR = 0.15          # +/- 15 % (guess from the spread of the car measurements)

# Sun elevation above the horizon. ASSUMED: there is no metadata, and a single
# nadir image only fixes height / tan(elevation), never the two separately.
# 72 deg is a calibration, not a measurement: over the roughly 200 roofs with open
# ground on the shadow side the mean shadow width is about 0.5 px (median about
# 0.25 px). At 60 deg that would make the average building about 1.5 m tall, which
# no building is, so the sun was higher. 72 deg is the elevation at which that
# mean shadow width corresponds to roughly one storey (2.7 m). Every
# shadow-derived height scales with tan(elevation): x0.56 at 60 deg, x1.84 at 80 deg.
SUN_ELEV_DEG = 72.0
SUN_ELEV_RANGE = (60.0, 80.0)

# Image "up" is ASSUMED to be north.
NORTH_UP_ASSUMED = True

STOREY_M = 3.0              # storey height used for floor counts (convention)
TILED_PITCH_DEG = 25.0      # default pitch for tiled roofs (convention from the brief)

UPSCALE = 4                 # working resolution = input x UPSCALE (interpolated, adds no detail)


def load_content():
    """Return (rgb_content, (x0, y0, x1, y1)) with the white screenshot margin removed."""
    bgr = cv2.imread(str(HERE / "input.png"), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = rgb.mean(axis=2)
    sat = rgb.max(axis=2).astype(int) - rgb.min(axis=2).astype(int)
    # A margin row/column is almost entirely near-white and colourless.
    white = (gray > 225) & (sat < 18)
    rows = white.mean(axis=1) < 0.85
    cols = white.mean(axis=0) < 0.85
    ys = np.where(rows)[0]
    xs = np.where(cols)[0]
    # inset by one pixel: the outermost row/column is blended with the white margin
    x0, x1, y0, y1 = xs[0] + 1, xs[-1], ys[0] + 1, ys[-1]
    return rgb[y0:y1, x0:x1].copy(), (int(x0), int(y0), int(x1), int(y1))


def upscale(rgb, k=UPSCALE):
    return cv2.resize(rgb, (rgb.shape[1] * k, rgb.shape[0] * k), interpolation=cv2.INTER_CUBIC)


def working_images():
    """Return (rgb, box, up_orig, up_geom) at working resolution.

    up_orig: the original, bicubic-upscaled (radiometry: shadows, colours).
    up_geom: the image outlines are traced on (the enhanced image when SOURCE says so).
    """
    rgb, box = load_content()
    up_orig = upscale(rgb)
    if SOURCE != "enhanced":
        return rgb, box, up_orig, up_orig
    enh = cv2.cvtColor(cv2.imread(str(HERE / "input_enhanced.png"), cv2.IMREAD_COLOR),
                       cv2.COLOR_BGR2RGB)
    x0, y0, x1, y1 = box
    k = UPSCALE
    up_geom = enh[y0 * k:y1 * k, x0 * k:x1 * k].copy()
    assert up_geom.shape == up_orig.shape, (up_geom.shape, up_orig.shape)
    return rgb, box, up_orig, up_geom


def px_to_m(pts_px, shape_hw):
    """Content-pixel coords (x right, y down) -> metres (x east, y north), origin at centre."""
    h, w = shape_hw
    pts = np.asarray(pts_px, dtype=float)
    out = np.empty_like(pts)
    out[..., 0] = (pts[..., 0] - w / 2.0) * GSD_M
    out[..., 1] = (h / 2.0 - pts[..., 1]) * GSD_M
    return out
