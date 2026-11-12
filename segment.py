"""Steps 2-4: footprints, sun azimuth, shadow heights, roof shapes, trees, land cover.

Writes work/scene.json (everything the mesh builder needs) and overlay images.
All pixel coordinates in scene.json are CONTENT pixels of the input (x right, y down).
"""
import json
import math

import cv2
import numpy as np
from scipy import ndimage as ndi

import classify as C
from common import (GSD_M, HERE, SOURCE, STOREY_M, SUN_ELEV_DEG, TILED_PITCH_DEG, UPSCALE,
                    WORK, working_images)

K = UPSCALE
PX_M = GSD_M / K                      # metres per working pixel
MIN_AREA_M2 = 30.0                    # smaller blobs are vehicles / noise at this resolution
ROOF_CLASSES = {C.RED: "red_tile", C.BRIGHT: "bright", C.GREYROOF: "grey"}

# shadow measurement
V_SHADOW = 78.0        # brightness of cast shadow on paving, sampled from the image
GREEN_OPEN = 0.20      # a strip greener than this is canopy: no shadow can be read there
NEAR_PX = 3.0          # shadow is looked for within this distance of the roof edge
FAR_PX = (3.0, 4.5)    # lit-ground reference band beyond it, on the same side
MIN_MEASURABLE_PX = 1.0
MAX_RUN_PX = 12.0


def to_py(o):
    """json.dumps fallback for numpy scalars."""
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


# ------------------------------------------------------------------ footprints
def split_components(mask, open_px=5, grow_px=5):
    """Label a class mask, cutting thin bridges between blobs."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    core = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k)
    n, markers = cv2.connectedComponents(core, connectivity=4)
    # give the pixels removed by the opening back to the nearest core blob
    dist, idx = ndi.distance_transform_edt(markers == 0, return_indices=True)
    nearest = markers[idx[0], idx[1]]
    lab = np.where(mask & (dist <= grow_px), nearest, 0)
    return lab, n - 1


def rectilinear(c, blob):
    """Right-angled outline along the blob's dominant edge direction, or None.

    Only used on the enhanced image, where roof edges are sharp enough to carry a
    direction. It is a modelling prior (walls meet at right angles), so footprints
    regularised this way are labelled "rectilinear".
    """
    pts = c.reshape(-1, 2).astype(np.float64)
    simp = cv2.approxPolyDP(c, 1.0 * K, True).reshape(-1, 2).astype(np.float64)
    n = len(simp)
    if n < 4:
        return None
    # dominant edge direction modulo 90 degrees (length-weighted)
    d = np.roll(simp, -1, axis=0) - simp
    ln = np.hypot(d[:, 0], d[:, 1])
    th = np.arctan2(d[:, 1], d[:, 0])
    a = math.atan2(float((ln * np.sin(4 * th)).sum()), float((ln * np.cos(4 * th)).sum())) / 4.0
    R = np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])   # rotate by -a
    ctr = pts.mean(axis=0)
    q = (simp - ctr) @ R.T
    dq = np.roll(q, -1, axis=0) - q
    types = ["H" if abs(e[0]) >= abs(e[1]) else "V" for e in dq]
    start = next((i for i in range(n) if types[i] != types[i - 1]), None)
    if start is None:
        return None
    groups, cur, acc_w, acc_c = [], None, 0.0, 0.0
    for i in [(start + k) % n for k in range(n)]:
        p0, p1 = q[i], q[(i + 1) % n]
        L = float(np.hypot(*(p1 - p0)))
        coord = (p0[1] + p1[1]) / 2 if types[i] == "H" else (p0[0] + p1[0]) / 2
        if types[i] == cur:
            acc_w += L
            acc_c += L * coord
        else:
            if cur is not None:
                groups.append((cur, acc_c / max(acc_w, 1e-9)))
            cur, acc_w, acc_c = types[i], L, L * coord
    groups.append((cur, acc_c / max(acc_w, 1e-9)))
    m = len(groups)
    if m < 4 or m % 2:
        return None
    verts = []
    for i in range(m):
        (t0, c0), (_, c1) = groups[i], groups[(i + 1) % m]
        verts.append((c1, c0) if t0 == "H" else (c0, c1))
    out = np.array(verts) @ R + ctr
    # accept only if it still covers the blob it came from
    x0, y0 = np.floor(out.min(axis=0)).astype(int) - 2
    x0, y0 = max(x0, 0), max(y0, 0)
    x1 = min(int(out[:, 0].max()) + 3, blob.shape[1])
    y1 = min(int(out[:, 1].max()) + 3, blob.shape[0])
    if x1 <= x0 or y1 <= y0:
        return None
    win = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(win, [np.round(out - [x0, y0]).astype(np.int32)], 1)
    ref = blob[y0:y1, x0:x1]
    union = float((win.astype(bool) | ref).sum())
    iou = float((win.astype(bool) & ref).sum()) / max(union, 1.0)
    covered = float((win.astype(bool) & ref).sum()) / max(float(blob.sum()), 1.0)
    return out if iou >= 0.72 and covered >= 0.8 else None


def polygon_of(blob):
    """Outer ring (working px) of a blob, regularised where a simple shape fits."""
    cnts, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    rect = cv2.minAreaRect(c)
    (cx, cy), (w, h), ang = rect
    rect_area = max(w * h, 1e-6)
    hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1e-6)
    info = {"rectangularity": round(area / rect_area, 3), "solidity": round(area / hull_area, 3)}
    pts = None
    if area / rect_area >= 0.78:
        # shrink the box so that it has the same area as the blob (keeps size unbiased)
        s = math.sqrt(area / rect_area)
        pts = cv2.boxPoints(((cx, cy), (w * s, h * s), ang))
        w, h = w * s, h * s
        info["shape"] = "rectangle"
    elif SOURCE == "enhanced":
        pts = rectilinear(c, blob)
        if pts is not None:
            info["shape"] = "rectilinear"
    if pts is None:
        pts = cv2.approxPolyDP(c, 0.6 * K, True).reshape(-1, 2).astype(np.float32)
        info["shape"] = "polygon"
        if len(pts) < 3:
            pts = cv2.boxPoints(rect)
            info["shape"] = "rectangle"
    info["rect"] = [float(cx), float(cy), float(w), float(h), float(ang)]
    return np.asarray(pts, float), info


def extract_buildings(lab, up):
    blds = []
    for cls, cname in ROOF_CLASSES.items():
        comp, n = split_components(lab == cls)
        for i in range(1, n + 1):
            blob = comp == i
            area_m2 = float(blob.sum()) * PX_M ** 2
            if area_m2 < MIN_AREA_M2:
                continue
            pts, info = polygon_of(blob)
            if cls == C.GREYROOF and (info["solidity"] < 0.72 or area_m2 < 45):
                continue          # ragged light-grey patches are usually paving
            holes = ndi.binary_fill_holes(blob) & ~blob
            col = up[blob].reshape(-1, 3).mean(axis=0)
            blds.append({
                "class": cname, "source": "automatic",
                "courtyard_m2": round(float(holes.sum()) * PX_M ** 2, 1),
                "roof_rgb": [int(round(v)) for v in col],
                "poly_up": pts.tolist(), **info,
            })
    # hand-drawn outlines for roofs the colour test cannot separate from pale ground
    manual = [mf for mf in json.loads((HERE / "manual_footprints.json").read_text())["footprints"]
              if mf["drawn_on"] == "original" or SOURCE == "enhanced"]
    manual_mask = np.zeros(lab.shape, np.uint8)
    for mf in manual:
        cv2.fillPoly(manual_mask, [np.round(np.array(mf["poly_px"]) * K).astype(np.int32)], 1)

    def mostly_inside_manual(b):
        m = np.zeros(lab.shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(b["poly_up"])).astype(np.int32)], 1)
        return (m & manual_mask).sum() >= 0.4 * max(int(m.sum()), 1)

    blds = [b for b in blds if not mostly_inside_manual(b)]
    for mf in manual:
        poly = np.array(mf["poly_px"], np.float32) * K
        m = np.zeros(lab.shape, np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        (cx, cy), (w, h), ang = cv2.minAreaRect(poly)
        area = cv2.contourArea(poly)
        blds.append({
            "class": mf["class"], "source": "manual", "outline_note": mf["note"],
            "drawn_on": mf["drawn_on"],
            "courtyard_m2": 0.0,
            "roof_rgb": [int(round(v)) for v in up[m > 0].reshape(-1, 3).mean(axis=0)],
            "poly_up": poly.astype(float).tolist(), "shape": "polygon",
            "rectangularity": round(area / max(w * h, 1e-6), 3),
            "solidity": round(area / max(cv2.contourArea(cv2.convexHull(poly)), 1e-6), 3),
            "rect": [float(cx), float(cy), float(w), float(h), float(ang)],
        })
    # ids run left to right within 25 px bands from the top
    blds.sort(key=lambda b: (round(np.mean([p[1] for p in b["poly_up"]]) / (25 * K)),
                             np.mean([p[0] for p in b["poly_up"]])))
    idmap = np.zeros(lab.shape, np.int32)
    for j, b in enumerate(blds, start=1):
        b["id"] = f"B{j:03d}"
        poly = np.array(b["poly_up"])
        cv2.fillPoly(idmap, [np.round(poly).astype(np.int32)], j)
        b["area_m2"] = round(float(cv2.contourArea(poly.astype(np.float32))) * PX_M ** 2, 1)
    return blds, idmap


# ------------------------------------------------------------------ sun / shadow
def shift_mask(mask, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask.astype(np.uint8), M, (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST) > 0


AZ_ANGLES = np.arange(0, 360, 15)


def directional_profile(roofs, value, region=None):
    """Mean of `value` in the 0.5-2.5 px band outside roofs, per image direction.

    Angles are degrees clockwise from image-up (0 = up, 90 = right).
    """
    inner = {}
    prof = np.zeros(len(AZ_ANGLES))
    for i, a in enumerate(AZ_ANGLES):
        ux, uy = math.sin(math.radians(a)), -math.cos(math.radians(a))
        band = shift_mask(roofs, ux * 2.5 * K, uy * 2.5 * K) & \
            ~shift_mask(roofs, ux * 0.5 * K, uy * 0.5 * K) & ~roofs
        if region is not None:
            band &= region
        prof[i] = float(value[band].mean()) if band.sum() > 50 else np.nan
    return prof


def first_harmonic(prof):
    """Direction (deg) and amplitude of the first harmonic of a directional profile."""
    p = np.where(np.isnan(prof), np.nanmean(prof), prof)
    sx = float((p * np.sin(np.radians(AZ_ANGLES))).sum())
    sy = float((p * np.cos(np.radians(AZ_ANGLES))).sum())
    return math.degrees(math.atan2(sx, sy)) % 360.0, 2.0 * math.hypot(sx, sy) / len(AZ_ANGLES)


def estimate_shadow_azimuth(roofs_raw, V, shadowish):
    """Two independent reads of the shadow direction, plus their spread over quadrants.

    (a) brightness: the side of roofs that is darkest on average
    (b) colour:     the side with the most dark, weakly green (shadow-coloured) pixels
    Tree canopy surrounds roofs on all sides, so both signals are faint.
    """
    darkness = 255.0 - V
    az_a, amp_a = first_harmonic(directional_profile(roofs_raw, darkness))
    az_b, amp_b = first_harmonic(directional_profile(roofs_raw, shadowish.astype(np.float32)))
    hh, ww = roofs_raw.shape
    quads = []
    for qy in (0, 1):
        for qx in (0, 1):
            reg = np.zeros(roofs_raw.shape, bool)
            reg[qy * hh // 2:(qy + 1) * hh // 2, qx * ww // 2:(qx + 1) * ww // 2] = True
            quads.append(first_harmonic(directional_profile(roofs_raw, darkness, reg))[0])
    # The brightness read is adopted: it repeats in all four quadrants. The colour
    # read is kept as a cross-check only (pixels blended between a red roof and
    # canopy also look "dark, weakly green", which contaminates it).
    az = az_a
    devs = [((q - az + 180) % 360) - 180 for q in quads]
    return {
        "azimuth": az, "brightness_based": az_a, "brightness_amplitude_V": amp_a,
        "colour_based": az_b, "colour_amplitude_fraction": amp_b,
        "quadrants_brightness_based": quads,
        "uncertainty_deg": float(np.sqrt(np.mean(np.square(devs)))),
    }


def side_profile(m, roofs, V, exg, ux, uy, dmax):
    """Mean V, mean greenness and visible fraction in rings behind mask m along (ux, uy)."""
    prev = m.copy()
    vs, gs, fr, rings = [], [], [], []
    for d in range(1, dmax + 1):
        sh = shift_mask(m, ux * d, uy * d)
        ring = sh & ~prev
        prev |= sh
        v = ring & ~roofs
        n, nv = int(ring.sum()), int(v.sum())
        rings.append(v)
        fr.append(nv / n if n else 0.0)
        vs.append(float(V[v].mean()) if nv >= 6 else np.nan)
        gs.append(float(exg[v].mean()) if nv >= 6 else np.nan)
    return np.array(vs), np.array(gs), np.array(fr), rings


def measure_shadows(idmap, V, exg, n_blds, az_deg):
    """Equivalent shadow width behind each roof; returns (list of readings, shadow mask).

    The thumbnail blurs a 1-2 px shadow into a faint dip, so a brightness threshold
    misses it. Blur preserves the integral, so the width is taken as the integrated
    darkening relative to lit ground further out on the same side:
        width = sum over d < NEAR_PX of clip((V_ref - V(d)) / (V_ref - V_SHADOW), 0, 1)
    Longer shadows are read as the length of the continuous dark run instead.
    Nothing is reported where the shadow side is canopy or another roof.
    """
    roofs = idmap > 0
    ux, uy = math.sin(math.radians(az_deg)), -math.cos(math.radians(az_deg))
    near, far0, far1 = int(NEAR_PX * K), int(FAR_PX[0] * K), int(FAR_PX[1] * K)
    dmax = int(MAX_RUN_PX * K)
    shadow_map = np.zeros(idmap.shape, bool)
    H, W = idmap.shape
    out = []
    for j in range(1, n_blds + 1):
        # work on a window around the roof: the search never reaches further than dmax
        ys, xs = np.nonzero(idmap == j)
        pad = dmax + 4
        sl = (slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)),
              slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)))
        m, roofs_w, V_w, exg_w = idmap[sl] == j, roofs[sl], V[sl], exg[sl]
        vs, gs, fr, rings = side_profile(m, roofs_w, V_w, exg_w, ux, uy, dmax)
        res = {"shadow_px": None}
        vis_near = float(np.mean(fr[:near]))
        green_near = float(np.nanmean(gs[:near])) if np.isfinite(gs[:near]).any() else 1.0
        res["shadow_side_visible"] = round(vis_near, 2)
        res["shadow_side_greenness"] = round(green_near, 3)
        if vis_near < 0.5:
            res["shadow_status"] = "occluded: another roof covers the shadow side"
        elif green_near >= GREEN_OPEN:
            res["shadow_status"] = "canopy: trees cover the shadow side"
        else:
            far = vs[far0:far1]
            far_g = gs[far0:far1]
            ok = np.isfinite(far) & (far_g < GREEN_OPEN) & (fr[far0:far1] > 0.5)
            if ok.sum() >= 3:
                v_ref = float(np.median(far[ok]))
                res["lit_reference"] = "ground 3-4.5 px beyond the roof, same side"
            else:
                # lit ground on the sun side of the same roof
                svs, sgs, sfr, _ = side_profile(m, roofs_w, V_w, exg_w, -ux, -uy, near)
                ok2 = np.isfinite(svs) & (sgs < GREEN_OPEN) & (sfr > 0.5)
                v_ref = float(np.median(svs[ok2])) if ok2.sum() >= 3 else float("nan")
                res["lit_reference"] = "ground on the sun side of the roof"
            if not np.isfinite(v_ref) or v_ref < V_SHADOW + 15:
                res["shadow_status"] = "no usable lit-ground reference next to the roof"
            else:
                res["lit_reference_V"] = round(v_ref, 1)
                dip = np.clip((v_ref - vs[:near]) / (v_ref - V_SHADOW), 0.0, 1.0)
                dip = np.where(np.isfinite(dip), dip, 0.0)
                w_eq = float(dip.sum()) / K
                # continuous dark run (for shadows longer than the near band)
                dark = np.nan_to_num(vs, nan=255.0) < (V_SHADOW + 12)
                run = 0
                while run < dmax and dark[run] and fr[run] > 0.4:
                    run += 1
                run_ok = run >= near and run < dmax and \
                    np.nanmean(gs[:run]) < GREEN_OPEN and fr[min(run, dmax - 1)] > 0.4
                res["shadow_px"] = round(run / K if run_ok else w_eq, 2)
                res["shadow_status"] = ("continuous dark run ending on lit ground" if run_ok
                                        else "integrated darkening next to the roof")
                n_mark = run if run_ok else int(round(res["shadow_px"] * K))
                for s in rings[:n_mark]:
                    shadow_map[sl] |= s
        out.append(res)
    return out, shadow_map


def storey_guess(b):
    if b["class"] == "red_tile":
        return 1 if b["area_m2"] < 80 else 2
    return 1 if b["area_m2"] < 100 else 2


def assign_heights(blds):
    t = math.tan(math.radians(SUN_ELEV_DEG))
    for b in blds:
        sp = b["shadow_px"]
        if sp is not None and sp >= MIN_MEASURABLE_PX:
            h = sp * GSD_M * t
            b["height_method"] = "shadow"
            b["height_is_guess"] = False
            b["confidence"] = "medium" if sp >= 2.0 else "low"
            b["height_note"] = (f"shadow {sp:.2f} px x {GSD_M} m/px x tan({SUN_ELEV_DEG:.0f} deg); "
                                "sun elevation and scale are assumed")
        else:
            h = storey_guess(b) * STOREY_M
            b["height_method"] = "storey_guess"
            b["height_is_guess"] = True
            b["confidence"] = "low"
            why = (f"shadow only {sp:.2f} px, below the 1 px that can be measured" if sp is not None
                   else b["shadow_status"])
            b["height_note"] = (f"GUESS from roof type and footprint size x {STOREY_M:.0f} m "
                                f"per storey ({why})")
        b["height_m"] = round(float(h), 1)
        b["floors"] = max(1, int(round(h / STOREY_M)))


# ------------------------------------------------------------------ roof shape
def roof_shapes(blds, up):
    V = cv2.GaussianBlur(up, (0, 0), 1.0).astype(np.float32).max(axis=2)
    for b in blds:
        cx, cy, w, h, ang = b["rect"]
        long_ang = ang if w >= h else ang + 90.0          # image angle of the long side
        b["ridge_azimuth_deg"] = None
        if b["class"] != "red_tile":
            b["roof_shape"] = "flat"
            b["roof_shape_method"] = "GUESS: not tiled, no ridge or pitch resolved, taken as flat"
            b["roof_is_guess"] = True
            continue
        if b["shape"] != "rectangle":
            b["roof_shape"] = "unknown"
            b["roof_shape_method"] = ("tiled, so pitched, but the outline is irregular and no "
                                      "ridge is resolved; modelled as a flat-topped prism")
            b["roof_is_guess"] = True
            continue
        poly = np.array(b["poly_up"])
        m = np.zeros(V.shape, np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        ys, xs = np.nonzero(m)
        lx, ly = math.cos(math.radians(long_ang)), math.sin(math.radians(long_ang))
        side = (xs - cx) * -ly + (ys - cy) * lx           # signed distance from the long axis
        half = min(w, h) / 2.0
        a = V[ys, xs][side > 0.25 * half]
        c = V[ys, xs][side < -0.25 * half]
        diff = float(a.mean() - c.mean()) if len(a) > 8 and len(c) > 8 else 0.0
        b["ridge_azimuth_deg"] = round((math.degrees(math.atan2(lx, -ly))) % 180.0, 1)
        b["roof_shape"] = "gabled"
        b["slope_brightness_diff"] = round(diff, 1)
        if abs(diff) >= 6.0:
            b["roof_shape_method"] = ("shading: the two halves either side of the long axis "
                                      f"differ by {abs(diff):.0f} brightness levels; "
                                      f"pitch {TILED_PITCH_DEG:.0f} deg is the default")
            b["roof_is_guess"] = False
        else:
            b["roof_shape_method"] = ("GUESS: tiled roof, ridge assumed along the long side at "
                                      f"{TILED_PITCH_DEG:.0f} deg; no ridge or shading visible")
            b["roof_is_guess"] = True


# ------------------------------------------------------------------ trees / cover
def place_trees(canopy, cap_m=5.0, min_m=1.6):
    """Representative crowns: greedy circle packing of the canopy mask."""
    dist = cv2.distanceTransform(canopy.astype(np.uint8), cv2.DIST_L2, 5)
    r_work = np.minimum(dist, cap_m / PX_M)
    rmin = min_m / PX_M
    trees = []
    yy, xx = np.mgrid[0:canopy.shape[0], 0:canopy.shape[1]]
    while True:
        i = int(np.argmax(r_work))
        y, x = divmod(i, canopy.shape[1])
        rad = float(r_work[y, x])
        if rad < rmin:
            break
        trees.append((x / K, y / K, rad / K))
        sup = int(rad * 1.7) + 1
        y0, y1, x0, x1 = max(y - sup, 0), y + sup + 1, max(x - sup, 0), x + sup + 1
        sub = (xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2 <= (rad * 1.7) ** 2
        r_work[y0:y1, x0:x1][sub] = 0
    return trees


def cover_polygons(mask, min_m2):
    out = []
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return out
    for i, c in enumerate(cnts):
        if hier[0][i][3] != -1:
            continue                                   # holes are attached to their parent
        if cv2.contourArea(c) * PX_M ** 2 < min_m2:
            continue
        rings = [cv2.approxPolyDP(c, 0.5 * K, True).reshape(-1, 2) / K]
        ch = hier[0][i][2]
        while ch != -1:
            if cv2.contourArea(cnts[ch]) * PX_M ** 2 >= min_m2 / 2:
                hr = cv2.approxPolyDP(cnts[ch], 0.5 * K, True).reshape(-1, 2) / K
                if len(hr) >= 3:
                    rings.append(hr)
            ch = hier[0][ch][0]
        if len(rings[0]) >= 3:
            out.append([r.tolist() for r in rings])
    return out


# ------------------------------------------------------------------ main
def main():
    # up: the image outlines are traced on. up_orig: the original, used for everything
    # that depends on brightness or colour (shadows, roof colour, roof shading).
    rgb, box, up_orig, up = working_images()
    lab = C.classify(up)
    lab_o = C.classify(up_orig)
    _, _, _, V, _, exg = C.features(up_orig)
    opened = lambda m: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                                        np.ones((5, 5), np.uint8)) > 0

    blds, idmap = extract_buildings(lab, up_orig)

    # is each outline also roof-coloured in the original?
    roofish_o = np.isin(lab_o, list(ROOF_CLASSES))
    for j, b in enumerate(blds, start=1):
        m = idmap == j
        frac = float(roofish_o[m].mean()) if m.any() else 0.0
        b["roof_coloured_in_original"] = round(frac, 2)
        if b["source"] != "manual":
            b["outline_source"] = "AI-enhanced image" if SOURCE == "enhanced" else "original"
            b["supported_by_original"] = bool(frac >= 0.5)
        elif b["drawn_on"] == "original":
            b["outline_source"] = "drawn by hand on the original"
            b["supported_by_original"] = True
        else:
            # traced on the enhanced image: it must also stand out in the original
            ring = cv2.dilate(m.astype(np.uint8), np.ones((2 * 2 * K + 1,) * 2, np.uint8)) > 0
            ring &= ~m & (idmap == 0)
            contrast = float(V[m].mean() - V[ring].mean()) if ring.sum() > 20 else 0.0
            b["outline_source"] = "drawn by hand on the AI-enhanced image"
            b["contrast_in_original"] = round(contrast, 1)
            b["supported_by_original"] = bool(contrast >= 8.0 or frac >= 0.5)

    sun = estimate_shadow_azimuth(opened(roofish_o), V, (lab_o == C.DARK) & (exg < 0.22))
    az = sun["azimuth"]
    print("shadow azimuth (deg clockwise from image-up), read from the original:")
    print(f"  brightness-based {sun['brightness_based']:.0f}  (amplitude "
          f"{sun['brightness_amplitude_V']:.1f} brightness levels)")
    print(f"  colour-based     {sun['colour_based']:.0f}")
    print(f"  by quadrant      {[round(q) for q in sun['quadrants_brightness_based']]}")
    print(f"  adopted {az:.0f} +/- {sun['uncertainty_deg']:.0f}")

    readings, shadow_map = measure_shadows(idmap, V, exg, len(blds), az)
    for b, r in zip(blds, readings):
        b.update(r)
    if SOURCE == "enhanced":
        # the same two readings taken from the enhanced image, kept for comparison only
        _, _, _, Ve, _, exge = C.features(up)
        sun_e = estimate_shadow_azimuth(opened(np.isin(lab, list(ROOF_CLASSES))), Ve,
                                        (lab == C.DARK) & (exge < 0.22))
        sun["enhanced_image_brightness_based"] = sun_e["brightness_based"]
        sun["enhanced_image_quadrants"] = sun_e["quadrants_brightness_based"]
        print(f"  enhanced image: {sun_e['brightness_based']:.0f}, quadrants "
              f"{[round(q) for q in sun_e['quadrants_brightness_based']]}")
        readings_e, _ = measure_shadows(idmap, Ve, exge, len(blds), az)
        for b, r in zip(blds, readings_e):
            b["shadow_px_enhanced"] = r["shadow_px"]
        both = [(b["shadow_px"], b["shadow_px_enhanced"]) for b in blds
                if b["shadow_px"] is not None and b["shadow_px_enhanced"] is not None]
        if len(both) > 5:
            o, e = np.array(both).T
            print(f"shadow width, original vs enhanced: n={len(o)}  correlation "
                  f"{np.corrcoef(o, e)[0, 1]:.2f}  mean {o.mean():.2f} vs {e.mean():.2f} px  "
                  f"median {np.median(o):.2f} vs {np.median(e):.2f} px")
    assign_heights(blds)
    roof_shapes(blds, up_orig)

    canopy = (lab == C.DARK) & ~shadow_map & (idmap == 0)
    trees = place_trees(canopy)

    blue = (up_orig[..., 2].astype(int) > up_orig[..., 0].astype(int) + 25) & \
           (up_orig[..., 2].astype(int) > up_orig[..., 1].astype(int) + 10)
    cover = {
        "tree_canopy_or_deep_shadow": cover_polygons(canopy, 60),
        "grass_or_low_vegetation": cover_polygons(lab == C.GRASS, 80),
        "paved_road_or_car_park": cover_polygons((lab == C.PAVED) & (idmap == 0), 80),
        "pale_ground_or_unresolved_roof": cover_polygons(
            np.isin(lab, [C.PALE, C.GREYROOF, C.BRIGHT]) & (idmap == 0), 80),
    }

    for b in blds:
        b["poly_px"] = (np.array(b.pop("poly_up")) / K).round(2).tolist()
        cx, cy, w, h, ang = b.pop("rect")
        b["rect_px"] = [round(cx / K, 2), round(cy / K, 2), round(w / K, 2), round(h / K, 2),
                        round(ang, 1)]

    scene = {
        "outline_source": SOURCE,
        "content_box_in_input": box,
        "content_size_px": [rgb.shape[1], rgb.shape[0]],
        "shadow": {k: (round(v, 2) if isinstance(v, float) else [round(q, 1) for q in v])
                   for k, v in sun.items()},
        "blue_pixel_fraction": round(float(blue.mean()), 5),
        "buildings": blds,
        "trees_px": [[round(x, 2), round(y, 2), round(r, 2)] for x, y, r in trees],
        "cover_px": cover,
    }
    (WORK / "scene.json").write_text(json.dumps(scene, default=to_py))
    np.save(WORK / "shadow_map.npy", shadow_map)
    np.save(WORK / "idmap.npy", idmap)

    # ---- overlay for inspection
    ov = up.copy()
    ov[shadow_map] = (0.5 * ov[shadow_map] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    colors = {"red_tile": (255, 255, 0), "bright": (0, 255, 255), "grey": (255, 140, 0),
              "pale": (255, 90, 200)}
    for b in blds:
        p = np.round(np.array(b["poly_px"]) * K).astype(np.int32)
        cv2.polylines(ov, [p], True, colors[b["class"]], 2 if b["source"] == "manual" else 1,
                      cv2.LINE_AA)
    for b in blds:
        c = (np.array(b["poly_px"]) * K).mean(axis=0)
        for col, th in (((0, 0, 0), 2), ((255, 255, 255), 1)):
            cv2.putText(ov, b["id"][1:], (int(c[0]) - 9, int(c[1]) + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, th, cv2.LINE_AA)
    a1 = (int(60 + 40 * math.sin(math.radians(az))), int(60 - 40 * math.cos(math.radians(az))))
    cv2.arrowedLine(ov, (60, 60), a1, (255, 0, 255), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.imwrite(str(WORK / "overlay.png"), cv2.cvtColor(ov, cv2.COLOR_RGB2BGR))

    methods, status = {}, {}
    for b in blds:
        methods[b["height_method"]] = methods.get(b["height_method"], 0) + 1
        s = b["shadow_status"].split(":")[0]
        status[s] = status.get(s, 0) + 1
    print("buildings:", len(blds), {c: sum(b["class"] == c for b in blds) for c in colors})
    print("outline shapes:", {s: sum(b["shape"] == s for b in blds)
                              for s in ("rectangle", "rectilinear", "polygon")})
    print("not roof-coloured in the original:",
          sum(not b["supported_by_original"] for b in blds))
    print("shadow status:", status)
    print("height methods:", methods)
    sp = np.array([b["shadow_px"] for b in blds if b["shadow_px"] is not None])
    if len(sp):
        print("shadow width where readable (px): n=%d  median %.2f  mean %.2f  90th pct %.2f"
              % (len(sp), np.median(sp), sp.mean(), np.percentile(sp, 90)))
    print("trees:", len(trees), " blue fraction:", scene["blue_pixel_fraction"])


if __name__ == "__main__":
    main()
