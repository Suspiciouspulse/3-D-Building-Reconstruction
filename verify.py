"""Step 7: check the model against the photo and fix heights the shadows contradict.

Reads work/scene.json, applies height corrections in place, and writes
output/verification.json plus two comparison images. Run build_scene.py afterwards.
"""
import json
import math
import shutil

import cv2
import numpy as np

import classify as C
from common import (GSD_M, GSD_REL_ERR, OUT, SOURCE, STOREY_M, SUN_ELEV_DEG, SUN_ELEV_RANGE,
                    UPSCALE, WORK, working_images)
from segment import ROOF_CLASSES, shift_mask, to_py

K = UPSCALE
TOL_PX = 1.0


def sweep(m, ux, uy, length_px):
    """Region covered when mask m is dragged length_px (original px) along (ux, uy)."""
    out = np.zeros(m.shape, bool)
    for d in range(1, int(round(length_px * K)) + 1):
        out |= shift_mask(m, ux * d, uy * d)
    return out & ~m


def main():
    scene = json.loads((WORK / "scene.json").read_text())
    blds = scene["buildings"]
    # up: image the outlines were traced on; up_orig: the original (all brightness readings)
    rgb, _, up_orig, up = working_images()
    lab = C.classify(up)
    _, _, _, V, _, exg = C.features(up_orig)
    H, W = lab.shape
    t = math.tan(math.radians(SUN_ELEV_DEG))
    az = scene["shadow"]["azimuth"]
    ux, uy = math.sin(math.radians(az)), -math.cos(math.radians(az))

    idmap = np.zeros((H, W), np.int32)
    for j, b in enumerate(blds, start=1):
        cv2.fillPoly(idmap, [np.round(np.array(b["poly_px"]) * K).astype(np.int32)], j)
    model = idmap > 0

    # ---------------------------------------------------------- 1. footprints
    roofish = cv2.morphologyEx(np.isin(lab, list(ROOF_CLASSES)).astype(np.uint8),
                               cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
    manual = np.zeros((H, W), bool)
    for j, b in enumerate(blds, start=1):
        if b["source"] == "manual":
            manual |= idmap == j
    auto = model & ~manual
    ref = roofish & ~manual
    inter = float((auto & ref).sum())
    fp = {
        "compared_against": "roof-coloured pixels from the colour classifier (NOT ground truth)",
        "iou": round(inter / float((auto | ref).sum()), 3),
        "model_area_that_is_roof_coloured": round(inter / float(auto.sum()), 3),
        "roof_coloured_area_covered_by_model": round(inter / float(ref.sum()), 3),
        "roof_coloured_area_not_modelled_m2": round(float((ref & ~model).sum()) * (GSD_M / K) ** 2),
        "modelled_footprint_total_m2": round(float(model.sum()) * (GSD_M / K) ** 2),
        "pale_or_unresolved_area_m2": round(float((np.isin(lab, [C.PALE, C.GREYROOF, C.BRIGHT])
                                                   & ~model).sum()) * (GSD_M / K) ** 2),
    }

    # top-down render of the model next to the photo
    flat = {C.DARK: (52, 92, 46), C.GRASS: (120, 160, 90), C.PAVED: (120, 120, 118),
            C.PALE: (196, 186, 164), C.GREYROOF: (196, 186, 164), C.BRIGHT: (196, 186, 164),
            C.RED: (196, 186, 164), 0: (150, 150, 150)}
    render = np.zeros((H, W, 3), np.uint8)
    for k, c in flat.items():
        render[lab == k] = c
    for x, y, r in scene["trees_px"]:
        cv2.circle(render, (int(x * K), int(y * K)), max(int(r * K), 1), (40, 78, 38), -1,
                   cv2.LINE_AA)
    for b in blds:
        p = np.round(np.array(b["poly_px"]) * K).astype(np.int32)
        cv2.fillPoly(render, [p], tuple(int(v) for v in b["roof_rgb"]), cv2.LINE_AA)
        cv2.polylines(render, [p], True, (40, 40, 40), 1, cv2.LINE_AA)
    gap = np.full((H, 12, 3), 255, np.uint8)
    cv2.imwrite(str(OUT / "verify_topdown.png"),
                cv2.cvtColor(np.hstack([up, gap, render]), cv2.COLOR_RGB2BGR))

    # ---------------------------------------------------------- 2. shadows
    changed = []
    disagree_before = 0
    readable = 0
    for b in blds:
        b["predicted_shadow_px"] = round(b["height_m"] / (t * GSD_M), 2)
        if b["shadow_px"] is None:
            continue
        readable += 1
        if abs(b["predicted_shadow_px"] - b["shadow_px"]) <= TOL_PX:
            continue
        disagree_before += 1
        if b["height_is_guess"] and b["predicted_shadow_px"] > b["shadow_px"]:
            floors = b["floors"]
            while floors > 1 and floors * STOREY_M / (t * GSD_M) - b["shadow_px"] > TOL_PX:
                floors -= 1
            if floors != b["floors"]:
                changed.append({"id": b["id"], "from_m": b["height_m"],
                                "to_m": floors * STOREY_M})
                b["height_m"] = round(floors * STOREY_M, 1)
                b["floors"] = floors
                b["height_method"] = "storey_guess_reduced_by_shadow_check"
                b["height_note"] += (f"; reduced to {floors} storey(s) because the image shows "
                                     f"a shadow of only {b['shadow_px']:.2f} px")
                b["predicted_shadow_px"] = round(b["height_m"] / (t * GSD_M), 2)
    disagree_after = sum(1 for b in blds if b["shadow_px"] is not None
                         and abs(b["predicted_shadow_px"] - b["shadow_px"]) > TOL_PX)

    # synthetic shadows from the final heights, and the same regions mirrored to the sun side
    pred = np.zeros((H, W), bool)
    mirror = np.zeros((H, W), bool)
    for j, b in enumerate(blds, start=1):
        ys, xs = np.nonzero(idmap == j)
        L = max(b["predicted_shadow_px"], 1.0)      # at least 1 px so there is something to sample
        pad = int(L * K) + 4
        sl = (slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)),
              slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)))
        m = idmap[sl] == j
        pred[sl] |= sweep(m, ux, uy, L)
        mirror[sl] |= sweep(m, -ux, -uy, L)
    pred &= ~model
    mirror &= ~model
    open_ground = exg < 0.20                         # leave out canopy on both sides
    v_pred = float(V[pred & open_ground].mean())
    v_mirror = float(V[mirror & open_ground].mean())
    v_lit = float(np.median(V[np.isin(lab, [C.PAVED, C.PALE]) & ~pred & ~mirror]))

    ov = up.copy()
    cnts, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    tint = ov.copy()
    tint[pred] = (255, 0, 255)
    ov = cv2.addWeighted(ov, 0.6, tint, 0.4, 0)
    for b in blds:
        p = np.round(np.array(b["poly_px"]) * K).astype(np.int32)
        col = (255, 160, 0) if b["height_is_guess"] else (0, 230, 200)
        cv2.polylines(ov, [p], True, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / "verify_shadows.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))
    shutil.copyfile(WORK / "overlay.png", OUT / "footprints_overlay.png")

    # ---------------------------------------------------------- 3. summary
    methods = {}
    for b in blds:
        methods[b["height_method"]] = methods.get(b["height_method"], 0) + 1
    meas = [b for b in blds if not b["height_is_guess"]]
    roof = {}
    for b in blds:
        key = b["roof_shape"] + (" (guess)" if b["roof_is_guess"] else " (from shading)")
        roof[key] = roof.get(key, 0) + 1
    sp = np.array([b["shadow_px"] for b in blds if b["shadow_px"] is not None])
    px_err_m = 0.5 * GSD_M * t
    lo, hi = (math.tan(math.radians(e)) / t for e in SUN_ELEV_RANGE)
    both = [(b["shadow_px"], b["shadow_px_enhanced"]) for b in blds
            if b["shadow_px"] is not None and b.get("shadow_px_enhanced") is not None]
    summary = {
        "outlines_traced_on": SOURCE,
        "buildings": len(blds),
        "outline_shapes": {s: sum(b["shape"] == s for b in blds)
                           for s in ("rectangle", "rectilinear", "polygon")},
        "outlines_not_supported_by_original": [b["id"] for b in blds
                                               if not b["supported_by_original"]],
        "shadow_width_original_vs_enhanced": None if len(both) < 6 else {
            "roofs": len(both),
            "correlation": round(float(np.corrcoef(*np.array(both).T)[0, 1]), 2),
            "mean_px_original": round(float(np.mean([o for o, _ in both])), 2),
            "mean_px_enhanced": round(float(np.mean([e for _, e in both])), 2),
            "note": "heights use the original reading only"},
        "by_roof_class": {c: sum(b["class"] == c for b in blds)
                          for c in ("red_tile", "bright", "grey", "pale")},
        "outlined_by_hand": sum(b["source"] == "manual" for b in blds),
        "height_methods": methods,
        "roof_shapes": roof,
        "measured_heights_m": {
            "count": len(meas),
            "median": round(float(np.median([b["height_m"] for b in meas])), 1) if meas else None,
            "min": min((b["height_m"] for b in meas), default=None),
            "max": max((b["height_m"] for b in meas), default=None)},
        "shadow_readings": {
            "roofs_with_open_ground_on_shadow_side": int(len(sp)),
            "median_px": round(float(np.median(sp)), 2), "mean_px": round(float(sp.mean()), 2),
            "at_least_1px": int((sp >= 1).sum()), "at_least_2px": int((sp >= 2).sum())},
        "footprint_check": fp,
        "shadow_check": {
            "tolerance_px": TOL_PX,
            "roofs_compared": readable,
            "disagreeing_by_more_than_tolerance_before": disagree_before,
            "heights_changed": changed,
            "disagreeing_after": disagree_after,
            "mean_brightness_in_predicted_shadow": round(v_pred, 1),
            "mean_brightness_in_same_region_mirrored_to_sun_side": round(v_mirror, 1),
            "median_brightness_of_lit_ground": round(v_lit, 1),
        },
        "estimated_height_error": {
            "measured_from_shadow": (f"+/- {px_err_m:.1f} m from a +/- 0.5 px shadow reading, "
                                     f"+/- {GSD_REL_ERR * 100:.0f} % from scale, and a factor "
                                     f"x{lo:.2f} to x{hi:.2f} if the sun was at "
                                     f"{SUN_ELEV_RANGE[0]:.0f} or {SUN_ELEV_RANGE[1]:.0f} deg "
                                     f"instead of {SUN_ELEV_DEG:.0f}"),
            "storey_guess": "not measured; at least +/- one storey (3 m)"},
        "shadow_azimuth": scene["shadow"],
    }
    (OUT / "verification.json").write_text(json.dumps(summary, indent=1, default=to_py))
    (WORK / "scene.json").write_text(json.dumps(scene, default=to_py))
    print(json.dumps(summary, indent=1, default=to_py))


if __name__ == "__main__":
    main()
