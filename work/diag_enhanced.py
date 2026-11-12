"""Diagnostic: does the enhanced image agree with the original, and what did it add?"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).parent.parent
enh = np.array(Image.open(HERE / "input_enhanced.webp").convert("RGB"))
org = np.array(Image.open(HERE / "input.png").convert("RGB"))
print("enhanced", enh.shape[1], "x", enh.shape[0], " original", org.shape[1], "x", org.shape[0],
      " ratio", enh.shape[1] / org.shape[1], enh.shape[0] / org.shape[0])
Image.fromarray(enh).save(HERE / "input_enhanced.png")

# reduce the enhanced image back to the original size and compare
down = cv2.resize(enh, (org.shape[1], org.shape[0]), interpolation=cv2.INTER_AREA)
a = org[8:-8, 8:-8].astype(np.float32)
b = down[8:-8, 8:-8].astype(np.float32)
for i, c in enumerate("RGB"):
    corr = np.corrcoef(a[..., i].ravel(), b[..., i].ravel())[0, 1]
    print(f"{c}: correlation {corr:.3f}   mean original {a[..., i].mean():.1f}  "
          f"enhanced {b[..., i].mean():.1f}   mean abs diff {np.abs(a[..., i] - b[..., i]).mean():.1f}")
ga = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY).astype(np.float32)
gb = cv2.cvtColor(down, cv2.COLOR_RGB2GRAY).astype(np.float32)
(sx, sy), resp = cv2.phaseCorrelate(ga, gb)
print(f"offset of enhanced vs original: dx {sx:.2f} dy {sy:.2f} original px (response {resp:.2f})")

# local geometric agreement: offset in each cell of a 4 x 4 grid
H, W = ga.shape
offs = []
for gy in range(4):
    for gx in range(4):
        ys = slice(8 + gy * (H - 16) // 4, 8 + (gy + 1) * (H - 16) // 4)
        xs = slice(8 + gx * (W - 16) // 4, 8 + (gx + 1) * (W - 16) // 4)
        (dx, dy), r = cv2.phaseCorrelate(ga[ys, xs], gb[ys, xs])
        offs.append((dx, dy))
offs = np.array(offs)
print("local offsets (4 x 4 cells): max |dx| %.2f  max |dy| %.2f  rms %.2f original px"
      % (np.abs(offs[:, 0]).max(), np.abs(offs[:, 1]).max(), np.sqrt((offs ** 2).sum(1).mean())))

upb = cv2.resize(org, (enh.shape[1], enh.shape[0]), interpolation=cv2.INTER_CUBIC)
lap = lambda im: float(cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_32F).std())
print(f"fine-detail energy (Laplacian std): bicubic upscale {lap(upb):.1f}   enhanced {lap(enh):.1f}")
gv = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
print("brightness percentiles 1/5/50/95/99:  original",
      np.percentile(gv(org[8:-8, 8:-8]), [1, 5, 50, 95, 99]).round(0),
      " enhanced", np.percentile(gv(enh[32:-32, 32:-32]), [1, 5, 50, 95, 99]).round(0))
