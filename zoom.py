"""Make an upscaled, gridded crop so footprints can be checked or digitised by eye.

Usage: python zoom.py x0 y0 x1 y1 [scale] [--overlay]
Grid labels are CONTENT pixel coordinates (the same frame as scene.json).
With --overlay, the current footprints from work/scene.json are drawn on top.
"""
import json
import sys

import cv2
import numpy as np

from common import UPSCALE as K, WORK, working_images

args = [a for a in sys.argv[1:] if not a.startswith("--")]
x0, y0, x1, y1 = map(int, args[:4])
scale = int(args[4]) if len(args) > 4 else 10
_, _, _, up = working_images()          # the image outlines are traced on
crop = cv2.resize(up[y0 * K:y1 * K, x0 * K:x1 * K], None, fx=scale / K, fy=scale / K,
                  interpolation=cv2.INTER_LANCZOS4)
if "--overlay" in sys.argv:
    scene = json.loads((WORK / "scene.json").read_text())
    for b in scene["buildings"]:
        p = ((np.array(b["poly_px"]) - [x0, y0]) * scale).round().astype(np.int32)
        col = (255, 0, 255) if b.get("source") == "manual" else (255, 255, 0)
        cv2.polylines(crop, [p], True, col, 1, cv2.LINE_AA)
        c = p.mean(axis=0).astype(int)
        cv2.putText(crop, b["id"][1:], (int(c[0]) - 10, int(c[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(crop, b["id"][1:], (int(c[0]) - 10, int(c[1]) + 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
step = 10 if (x1 - x0) <= 130 else 25
for gx in range(-(-x0 // step) * step, x1 + 1, step):
    X = (gx - x0) * scale
    cv2.line(crop, (X, 0), (X, crop.shape[0]), (255, 255, 0), 1)
    cv2.putText(crop, str(gx), (X + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
for gy in range(-(-y0 // step) * step, y1 + 1, step):
    Y = (gy - y0) * scale
    cv2.line(crop, (0, Y), (crop.shape[1], Y), (0, 255, 255), 1)
    cv2.putText(crop, str(gy), (2, Y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
out = WORK / f"zoom_{x0}_{y0}.png"
cv2.imwrite(str(out), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
print(out, crop.shape[1], "x", crop.shape[0])
