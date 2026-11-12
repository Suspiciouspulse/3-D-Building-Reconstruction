"""Diagnostic: brightness just outside the largest roofs, on each of 8 sides."""
import json
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
import classify as C
from common import UPSCALE as K, WORK, load_content, upscale

rgb, box = load_content()
up = upscale(rgb)
R, G, B, V, S, exg = C.features(up)
idmap = np.load(WORK / "idmap.npy")
roofs = idmap > 0
scene = json.loads((WORK / "scene.json").read_text())
blds = scene["buildings"]
order = sorted(range(len(blds)), key=lambda i: -blds[i]["area_m2"])[:24]
dirs = [0, 45, 90, 135, 180, 225, 270, 315]
print("mean V (and greenness) in the 0.5-2.5 px band outside the roof, by side")
print("id    class     area  " + "  ".join(f"{d:>9d}" for d in dirs))
agg = {d: [] for d in dirs}
for i in order:
    b = blds[i]
    m = idmap == i + 1
    row = []
    for a in dirs:
        ux, uy = math.sin(math.radians(a)), -math.cos(math.radians(a))
        def sh(d):
            M = np.float32([[1, 0, ux * d * K], [0, 1, uy * d * K]])
            return cv2.warpAffine(m.astype(np.uint8), M, (m.shape[1], m.shape[0]),
                                  flags=cv2.INTER_NEAREST) > 0
        band = sh(2.5) & ~sh(0.5) & ~roofs
        if band.sum() < 12:
            row.append("    --   ")
            continue
        v, g = V[band].mean(), exg[band].mean()
        agg[a].append(v)
        row.append(f"{v:4.0f}/{g:.2f}")
    print(f"{b['id']}  {b['class']:8s} {b['area_m2']:6.0f}  " + "  ".join(row))
print("mean over these roofs:  " + "  ".join(f"{np.mean(agg[d]):9.1f}" for d in dirs))
