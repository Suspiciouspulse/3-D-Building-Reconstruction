"""Compare the model traced on the original with the one traced on the enhanced image."""
import json

import cv2
import numpy as np

from common import HERE, UPSCALE as K

a = json.loads((HERE / "work" / "scene.json").read_text())["buildings"]
b = json.loads((HERE / "work_enhanced" / "scene.json").read_text())["buildings"]
va = json.loads((HERE / "output" / "verification.json").read_text())
vb = json.loads((HERE / "output_enhanced" / "verification.json").read_text())
size = json.loads((HERE / "work" / "scene.json").read_text())["content_size_px"]
W, H = size[0] * K, size[1] * K


def raster(blds):
    m = np.zeros((H, W), np.int32)
    for j, x in enumerate(blds, start=1):
        cv2.fillPoly(m, [np.round(np.array(x["poly_px"]) * K).astype(np.int32)], j)
    return m


ma, mb = raster(a), raster(b)
print(f"{'':34s}{'original':>10s}{'enhanced':>10s}")
rows = [
    ("buildings", len(a), len(b)),
    ("  red tile", sum(x["class"] == "red_tile" for x in a), sum(x["class"] == "red_tile" for x in b)),
    ("  outlined by hand", sum(x["source"] == "manual" for x in a), sum(x["source"] == "manual" for x in b)),
    ("outlines: rectangle", sum(x["shape"] == "rectangle" for x in a), sum(x["shape"] == "rectangle" for x in b)),
    ("outlines: right-angled", sum(x["shape"] == "rectilinear" for x in a), sum(x["shape"] == "rectilinear" for x in b)),
    ("outlines: free polygon", sum(x["shape"] == "polygon" for x in a), sum(x["shape"] == "polygon" for x in b)),
    ("median vertices, free polygons",
     int(np.median([len(x["poly_px"]) for x in a if x["shape"] == "polygon"])),
     int(np.median([len(x["poly_px"]) for x in b if x["shape"] == "polygon"]))),
    ("median footprint m2", round(float(np.median([x["area_m2"] for x in a]))),
     round(float(np.median([x["area_m2"] for x in b])))),
    ("largest footprint m2", round(max(x["area_m2"] for x in a)), round(max(x["area_m2"] for x in b))),
    ("heights measured from shadow", va["measured_heights_m"]["count"], vb["measured_heights_m"]["count"]),
    ("heights guessed", len(a) - va["measured_heights_m"]["count"], len(b) - vb["measured_heights_m"]["count"]),
    ("gabled roofs from shading", va["roof_shapes"].get("gabled (from shading)", 0),
     vb["roof_shapes"].get("gabled (from shading)", 0)),
    ("modelled footprint m2", va["footprint_check"]["modelled_footprint_total_m2"],
     vb["footprint_check"]["modelled_footprint_total_m2"]),
    ("shadow-side darkening (levels)",
     round(va["shadow_check"]["mean_brightness_in_same_region_mirrored_to_sun_side"]
           - va["shadow_check"]["mean_brightness_in_predicted_shadow"], 1),
     round(vb["shadow_check"]["mean_brightness_in_same_region_mirrored_to_sun_side"]
           - vb["shadow_check"]["mean_brightness_in_predicted_shadow"], 1)),
]
for name, x, y in rows:
    print(f"{name:34s}{x:>10}{y:>10}")

inter = float(((ma > 0) & (mb > 0)).sum())
print(f"\nfootprint area shared by both models: IoU {inter / float(((ma > 0) | (mb > 0)).sum()):.2f}")
# how the first model's buildings map onto the new one
split = same = gone = 0
for j in range(1, len(a) + 1):
    ids, cnt = np.unique(mb[ma == j], return_counts=True)
    area = cnt.sum()
    hit = [(i, c) for i, c in zip(ids, cnt) if i > 0 and c >= 0.15 * area]
    if not hit:
        gone += 1
    elif len(hit) == 1:
        same += 1
    else:
        split += 1
new = sum(1 for j in range(1, len(b) + 1) if (ma[mb == j] > 0).mean() < 0.15)
print(f"of the {len(a)} first-model buildings: {same} map to one new building, "
      f"{split} are split into several, {gone} have no counterpart")
print(f"new-model buildings with no counterpart in the first model: {new}")
