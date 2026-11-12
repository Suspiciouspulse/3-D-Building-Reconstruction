"""Step 2a: per-pixel land-cover classes from colour.

The thumbnail is low-contrast with a yellow-green cast, so thresholds were tuned
on colours sampled from known surfaces in this image. Tree canopy and cast shadow have nearly the
same colour here (about RGB 76,78,60), so they share one class: DARK.

Classes (uint8 label image at working resolution):
  0 unclassified          4 grey roof candidate (light grey, colourless)
  1 dark: canopy/shadow   5 paved, mid-grey (roads, car parks, dark yards)
  2 red/terracotta roof   6 pale ground (bare soil, light paving, pale sheds)
  3 bright roof (white)   7 grass / low vegetation
"""
import cv2
import numpy as np

from common import WORK, load_content, upscale

DARK, RED, BRIGHT, GREYROOF, PAVED, PALE, GRASS = 1, 2, 3, 4, 5, 6, 7
NAMES = {0: "unclassified", DARK: "canopy_or_shadow", RED: "red_roof", BRIGHT: "bright_roof",
         GREYROOF: "grey_roof_candidate", PAVED: "paved", PALE: "pale_ground", GRASS: "grass"}
COLORS = {
    0: (90, 60, 110), DARK: (20, 90, 30), RED: (210, 70, 40), BRIGHT: (250, 250, 250),
    GREYROOF: (120, 170, 255), PAVED: (110, 110, 110), PALE: (215, 195, 150), GRASS: (120, 220, 90),
}
DARK_V = 90.0   # brightness below which a pixel counts as canopy/shadow


def features(rgb_up, sigma=1.0):
    f = cv2.GaussianBlur(rgb_up, (0, 0), sigma).astype(np.float32)
    R, G, B = f[..., 0], f[..., 1], f[..., 2]
    V = f.max(axis=2)
    S = (V - f.min(axis=2)) / np.maximum(V, 1)
    exg = (2 * G - R - B) / np.maximum(V, 1)
    return R, G, B, V, S, exg


def classify(rgb_up):
    R, G, B, V, S, exg = features(rgb_up)
    lab = np.zeros(V.shape, np.uint8)

    red = (R > 105) & (R > 1.17 * G) & (R > 1.38 * B)
    dark = (V < DARK_V) & ~red
    grass = (V >= DARK_V) & (V < 150) & (exg > 0.19) & ~red
    bright = (V > 195) & (S < 0.12)
    greyroof = (V >= 150) & (V <= 195) & (S < 0.115) & ~red
    paved = (V >= DARK_V) & (V < 135) & (exg <= 0.19) & ~red
    pale = (V >= 135) & ~red & ~bright & ~greyroof & ~grass

    lab[pale] = PALE
    lab[paved] = PAVED
    lab[greyroof] = GREYROOF
    lab[grass] = GRASS
    lab[bright] = BRIGHT
    lab[dark] = DARK
    lab[red] = RED
    return lab


def colorize(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for k, c in COLORS.items():
        out[lab == k] = c
    return out


if __name__ == "__main__":
    from common import SOURCE, working_images
    rgb, box, _, up = working_images()
    print("source:", SOURCE, " content box (x0,y0,x1,y1):", box, "size:", rgb.shape[1], "x",
          rgb.shape[0])
    lab = classify(up)
    for k in COLORS:
        print(f"{NAMES[k]:22s} {float((lab == k).mean()) * 100:5.1f} %")
    vis = np.hstack([up, colorize(lab)])
    cv2.imwrite(str(WORK / "classes.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
