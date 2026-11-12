"""Diagnostic: can cast shadow be told apart from canopy, and which way does it point?"""
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
lab = C.classify(up)
roofs = np.isin(lab, [C.RED, C.BRIGHT, C.GREYROOF])
roofs = cv2.morphologyEx(roofs.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0

print("dark pixels: exg/V distribution percentiles (5,25,50,75,95):",
      np.percentile(exg[V < 90], [5, 25, 50, 75, 95]).round(3))

for name, sel in [("dark, weakly green (exg<0.22)", (V < 92) & (exg < 0.22)),
                  ("dark, strongly green (exg>=0.26)", (V < 92) & (exg >= 0.26)),
                  ("all dark", V < 92),
                  ("brightness deficit (continuous)", None)]:
    out = []
    for a in range(0, 360, 15):
        tot = n = 0.0
        for d in (1.0, 2.0, 3.0):
            dx, dy = math.sin(math.radians(a)) * d * K, -math.cos(math.radians(a)) * d * K
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            sh = cv2.warpAffine(roofs.astype(np.uint8), M, (roofs.shape[1], roofs.shape[0]),
                                flags=cv2.INTER_NEAREST) > 0
            strip = sh & ~roofs
            if sel is None:
                tot += float(np.clip(110 - V[strip], 0, None).sum()) / 110
            else:
                tot += float(sel[strip].sum())
            n += float(strip.sum())
        out.append(tot / n)
    out = np.array(out)
    asym = out - np.roll(out, 12)          # D(a) - D(a+180)
    print("\n" + name)
    for a, o, s in zip(range(0, 360, 15), out, asym):
        bar = "#" * int(max(s, 0) * 400)
        print(f"  {a:3d}  frac={o:.3f}  asym={s:+.3f}  {bar}")

vis = up.copy()
vis[(V < 92) & (exg < 0.22)] = (255, 0, 255)
vis[(V < 92) & (exg >= 0.26)] = (0, 110, 0)
cv2.imwrite(str(WORK / "diag_shadow.png"), cv2.cvtColor(np.hstack([up, vis]), cv2.COLOR_RGB2BGR))
