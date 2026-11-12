"""Step 6: work/scene.json -> output/scene.glb, buildings.geojson, landcover.geojson, viewer.html.

Coordinates: metres, Z up, X = image-right (east if the image is north-up),
Y = image-up, origin at the centre of the image content. glTF itself is Y-up by
specification, so the Z-up data sits under one root node that rotates it; Blender
and other importers undo that and show Z up.
"""
import base64
import json
import math
import struct

import cv2
import numpy as np

from common import (GSD_M, GSD_REL_ERR, HERE, NORTH_UP_ASSUMED, OUT, SOURCE, STOREY_M,
                    SUN_ELEV_DEG, SUN_ELEV_RANGE, TILED_PITCH_DEG, WORK, working_images)

WALL_RGB = (200, 198, 190)      # placeholder: facades are not visible in a nadir image
TREE_HEIGHT_PER_RADIUS = 2.5    # GUESS: tree height = 2.5 x crown radius, at least 4 m
TREE_MIN_HEIGHT_M = 4.0
COLOR_NAMES = {"red_tile": "terracotta red", "bright": "white", "grey": "light grey",
               "pale": "pale / mixed"}


# ------------------------------------------------------------------ geometry
def to_m(pts_px, w, h):
    p = np.asarray(pts_px, float)
    return np.stack([(p[:, 0] - w / 2.0) * GSD_M, (h / 2.0 - p[:, 1]) * GSD_M], axis=1)


def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def clean_ccw(p):
    keep = [p[0]]
    for q in p[1:]:
        if np.hypot(*(q - keep[-1])) > 1e-6:
            keep.append(q)
    if len(keep) > 1 and np.hypot(*(keep[0] - keep[-1])) < 1e-6:
        keep.pop()
    p = np.array(keep)
    return p[::-1].copy() if signed_area(p) < 0 else p


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def in_tri(p, a, b, c):
    d1, d2, d3 = cross2(b - a, p - a), cross2(c - b, p - b), cross2(a - c, p - c)
    return d1 > 1e-9 and d2 > 1e-9 and d3 > 1e-9


def triangulate(poly):
    """Ear clipping for a simple CCW polygon; returns index triples."""
    idx = list(range(len(poly)))
    tris = []
    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross2(b - a, c - b) <= 1e-9:
                continue
            if any(in_tri(poly[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            # no ear (self-touching outline): fall back to a fan over what is left
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
            return tris
    tris.append(tuple(idx))
    return tris


class Tris:
    """Flat-shaded triangle soup."""

    def __init__(self):
        self.pos, self.nrm = [], []

    def add(self, a, b, c):
        a, b, c = (np.asarray(v, float) for v in (a, b, c))
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-12:
            return
        n /= ln
        self.pos += [a, b, c]
        self.nrm += [n, n, n]

    def quad(self, a, b, c, d):
        self.add(a, b, c)
        self.add(a, c, d)

    def arrays(self):
        return np.array(self.pos, np.float32), np.array(self.nrm, np.float32)


def v3(p, z):
    return (p[0], p[1], z)


def building_mesh(poly, h, roof_shape):
    """Return (walls, roof, ridge_height). poly is CCW in metres."""
    walls, roof = Tris(), Tris()
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        walls.quad(v3(a, 0), v3(b, 0), v3(b, h), v3(a, h))
    if roof_shape == "gabled" and n == 4:
        p = poly
        if np.hypot(*(p[1] - p[0])) < np.hypot(*(p[2] - p[1])):
            p = np.roll(p, -1, axis=0)                 # make edge 0-1 the long side
        span = np.hypot(*(p[2] - p[1]))
        zr = h + 0.5 * span * math.tan(math.radians(TILED_PITCH_DEG))
        r0, r1 = (p[0] + p[3]) / 2.0, (p[1] + p[2]) / 2.0
        roof.quad(v3(p[0], h), v3(p[1], h), v3(r1, zr), v3(r0, zr))
        roof.quad(v3(p[2], h), v3(p[3], h), v3(r0, zr), v3(r1, zr))
        walls.add(v3(p[1], h), v3(p[2], h), v3(r1, zr))    # gable ends
        walls.add(v3(p[3], h), v3(p[0], h), v3(r0, zr))
        return walls, roof, zr
    for i0, i1, i2 in triangulate(poly):
        roof.add(v3(poly[i0], h), v3(poly[i1], h), v3(poly[i2], h))
    return walls, roof, h


def icosphere():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                  [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    v /= np.linalg.norm(v, axis=1)[:, None]
    f = np.array([[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
                  [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
                  [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
                  [8, 6, 7], [9, 8, 1]], np.uint32)
    return v, f


def tree_meshes(trees_m):
    """Indexed crowns (smooth) and trunks. trees_m: rows of x, y, crown radius, height."""
    sv, sf = icosphere()
    cp, cn, ci, tp, tn, ti = [], [], [], [], [], []
    ang = np.arange(6) * math.pi / 3
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    for x, y, r, hgt in trees_m:
        zc = hgt - r
        base = len(cp) * 12
        cp.append(sv * r + np.array([x, y, zc]))
        cn.append(sv)
        ci.append(sf + base)
        tr = max(0.15, 0.1 * r)
        ztop = max(zc - 0.6 * r, 0.5)
        base = len(tp) * 12
        bot = np.column_stack([x + ring[:, 0] * tr, y + ring[:, 1] * tr, np.zeros(6)])
        top = bot.copy()
        top[:, 2] = ztop
        tp.append(np.vstack([bot, top]))
        nn = np.column_stack([ring, np.zeros(6)])
        tn.append(np.vstack([nn, nn]))
        idx = []
        for k in range(6):
            a, b = k, (k + 1) % 6
            idx += [[a, b, b + 6], [a, b + 6, a + 6]]
        ti.append(np.array(idx, np.uint32) + base)
    cat = lambda L, t: np.concatenate(L).astype(t) if L else np.zeros((0, 3), t)
    return (cat(cp, np.float32), cat(cn, np.float32), cat(ci, np.uint32),
            cat(tp, np.float32), cat(tn, np.float32), cat(ti, np.uint32))


# ------------------------------------------------------------------ GLB writer
class Glb:
    def __init__(self):
        self.bin = bytearray()
        self.j = {"asset": {"version": "2.0", "generator": "scene3d build_scene.py"},
                  "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [], "meshes": [],
                  "materials": [], "accessors": [], "bufferViews": [], "buffers": [{}]}
        self._mats = {}

    def _view(self, data, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            v["target"] = target
        self.bin += data
        self.j["bufferViews"].append(v)
        return len(self.j["bufferViews"]) - 1

    def _acc(self, arr, ctype, atype, target, minmax=False):
        a = {"bufferView": self._view(arr.tobytes(), target), "componentType": ctype,
             "count": int(arr.shape[0]), "type": atype}
        if minmax:
            a["min"] = [float(x) for x in arr.min(axis=0)]
            a["max"] = [float(x) for x in arr.max(axis=0)]
        self.j["accessors"].append(a)
        return len(self.j["accessors"]) - 1

    def material(self, rgb, name, texture=None):
        key = (tuple(rgb), texture)
        if key in self._mats:
            return self._mats[key]
        lin = [((c / 255 + 0.055) / 1.055) ** 2.4 if c / 255 > 0.04045 else c / 255 / 12.92
               for c in rgb]
        pbr = {"baseColorFactor": [round(x, 4) for x in lin] + [1.0],
               "metallicFactor": 0.0, "roughnessFactor": 0.95}
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
        self.j["materials"].append({"name": name, "pbrMetallicRoughness": pbr,
                                    "doubleSided": True})
        self._mats[key] = len(self.j["materials"]) - 1
        return self._mats[key]

    def primitive(self, pos, nrm, material, uv=None, idx=None):
        attrs = {"POSITION": self._acc(pos, 5126, "VEC3", 34962, True),
                 "NORMAL": self._acc(nrm, 5126, "VEC3", 34962)}
        if uv is not None:
            attrs["TEXCOORD_0"] = self._acc(uv, 5126, "VEC2", 34962)
        p = {"attributes": attrs, "material": material, "mode": 4}
        if idx is not None:
            p["indices"] = self._acc(idx.reshape(-1), 5125, "SCALAR", 34963)
        return p

    def mesh(self, prims, name):
        self.j["meshes"].append({"name": name, "primitives": prims})
        return len(self.j["meshes"]) - 1

    def node(self, **kw):
        self.j["nodes"].append(kw)
        return len(self.j["nodes"]) - 1

    def texture(self, img_bytes, mime):
        iv = self._view(img_bytes)
        self.j["images"] = [{"bufferView": iv, "mimeType": mime, "name": "ground"}]
        self.j["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071,
                               "wrapT": 33071}]
        self.j["textures"] = [{"sampler": 0, "source": 0}]
        return 0

    def tobytes(self):
        while len(self.bin) % 4:
            self.bin.append(0)
        self.j["buffers"][0]["byteLength"] = len(self.bin)
        js = json.dumps(self.j, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        return (struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(js), b"JSON")
                + js + struct.pack("<I4s", len(self.bin), b"BIN\x00") + bytes(self.bin))


# ------------------------------------------------------------------ main
def public_props(b, ridge_h):
    """Attributes written to the GeoJSON and attached to each building in the GLB."""
    hexcol = "#%02x%02x%02x" % tuple(b["roof_rgb"])
    return {
        "id": b["id"],
        "footprint_area_m2": b["area_m2"],
        "footprint_shape": b["shape"],
        "height_m": b["height_m"],
        "ridge_height_m": round(ridge_h, 1),
        "floors": b["floors"],
        "roof_shape": b["roof_shape"],
        "roof_shape_method": b["roof_shape_method"],
        "roof_shape_is_guess": b["roof_is_guess"],
        "ridge_azimuth_deg": b["ridge_azimuth_deg"],
        "roof_color": f"{COLOR_NAMES[b['class']]} ({hexcol})",
        "damage": b.get("damage", "none visible"),
        "height_method": b["height_method"],
        "height_is_guess": b["height_is_guess"],
        "height_note": b["height_note"],
        "confidence": b["confidence"],
        "shadow_px": b["shadow_px"],
        "shadow_px_enhanced_image": b.get("shadow_px_enhanced"),
        "shadow_status": b["shadow_status"],
        "courtyard_m2": b["courtyard_m2"],
        "outline_source": b.get("outline_source", "original"),
        "outline_note": b.get("outline_note"),
        "supported_by_original": b.get("supported_by_original", True),
    }


def main():
    scene = json.loads((WORK / "scene.json").read_text())
    rgb, _, _, up = working_images()
    hpx, wpx = rgb.shape[:2]
    wm, hm = wpx * GSD_M, hpx * GSD_M
    shadow_az = scene["shadow"]["azimuth"]
    meta = {
        "outlines_traced_on": ("AI-enhanced x4 upscale of the original; detail finer than one "
                               "original pixel was inferred by the upscaler"
                               if SOURCE == "enhanced" else "original image"),
        "outline_source_short": "AI-enhanced image" if SOURCE == "enhanced" else "original image",
        "heights_and_colours_from": "original image",
        "ground_texture": "AI-enhanced image" if SOURCE == "enhanced" else "original image",
        "units": "metres, Z up, origin at image centre, X = image right, Y = image up",
        "georeferenced": False,
        "extent_m": [round(wm, 1), round(hm, 1)],
        "gsd_m_per_px": GSD_M, "gsd_is": "ESTIMATE from car lengths, +/- %d %%" % (GSD_REL_ERR * 100),
        "north": "ASSUMED image-up" if NORTH_UP_ASSUMED else "given",
        "shadow_azimuth_deg": round(shadow_az, 1),
        "shadow_azimuth_is": "MEASURED from the image, +/- about 30 deg",
        "sun_azimuth_deg": round((shadow_az + 180) % 360, 1),
        "sun_elevation_deg": SUN_ELEV_DEG,
        "sun_elevation_is": "ASSUMED (calibrated so the mean shadow equals one storey), "
                            "plausible range %d-%d deg" % SUN_ELEV_RANGE,
        "storey_m": STOREY_M, "tiled_pitch_deg": TILED_PITCH_DEG,
        "wall_colour": "placeholder; facades are not visible",
        "trees": "representative crowns packed into the canopy outline; heights are GUESSES",
    }

    g = Glb()
    g.j["asset"]["extras"] = meta
    root_children = []

    # ground, textured with the photo
    if SOURCE == "enhanced":
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(up, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 88])
        tex = g.texture(enc.tobytes(), "image/jpeg")
    else:
        ok, enc = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        tex = g.texture(enc.tobytes(), "image/png")
    gp = np.array([[-wm / 2, -hm / 2, 0], [wm / 2, -hm / 2, 0], [wm / 2, hm / 2, 0],
                   [-wm / 2, -hm / 2, 0], [wm / 2, hm / 2, 0], [-wm / 2, hm / 2, 0]], np.float32)
    gn = np.tile(np.array([0, 0, 1], np.float32), (6, 1))
    guv = np.array([[0, 1], [1, 1], [1, 0], [0, 1], [1, 0], [0, 0]], np.float32)
    gm = g.mesh([g.primitive(gp, gn, g.material((255, 255, 255), "ground_photo", tex), guv)],
                "ground")
    root_children.append(g.node(name="ground", mesh=gm,
                                extras={"kind": "ground", "note": "input image draped flat"}))

    wall_mat = g.material(WALL_RGB, "walls_placeholder")
    feats = []
    for b in scene["buildings"]:
        poly = clean_ccw(to_m(b["poly_px"], wpx, hpx))
        if len(poly) < 3:
            continue
        walls, roof, ridge_h = building_mesh(poly, b["height_m"], b["roof_shape"])
        props = public_props(b, ridge_h)
        wp, wn = walls.arrays()
        rp, rn = roof.arrays()
        roof_mat = g.material(tuple(b["roof_rgb"]), "roof_" + props["roof_color"])
        m = g.mesh([g.primitive(wp, wn, wall_mat), g.primitive(rp, rn, roof_mat)], b["id"])
        root_children.append(g.node(name=b["id"], mesh=m, extras=dict(props, kind="building")))
        ring = [[round(float(x), 2), round(float(y), 2)] for x, y in poly]
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}})

    # trees
    trees_m = []
    for x, y, r in scene["trees_px"]:
        (xm, ym), = to_m([[x, y]], wpx, hpx)
        rm = r * GSD_M
        trees_m.append((xm, ym, rm, max(TREE_MIN_HEIGHT_M, TREE_HEIGHT_PER_RADIUS * rm)))
    cp, cn, ci, tp, tn, ti = tree_meshes(trees_m)
    if len(cp):
        tm = g.mesh([g.primitive(cp, cn, g.material((62, 104, 52), "tree_crown"), idx=ci),
                     g.primitive(tp, tn, g.material((96, 74, 54), "tree_trunk"), idx=ti)], "trees")
        root_children.append(g.node(name="trees", mesh=tm, extras={
            "kind": "trees", "count": len(trees_m),
            "note": "GUESS: crowns packed into the canopy outline, height = 2.5 x crown "
                    "radius (min 4 m). Positions are representative, not individual trees."}))

    s = math.sqrt(0.5)
    # root goes first, so every child index moves up by one
    g.j["nodes"].insert(0, {"name": "scene_Zup_metres", "rotation": [-s, 0, 0, s],
                            "children": [c + 1 for c in root_children]})
    glb = g.tobytes()
    (OUT / "scene.glb").write_bytes(glb)

    crs_note = {"type": "local", "units": "metres", "origin": "image centre",
                "axes": "x = image right, y = image up", "georeferenced": False}
    (OUT / "buildings.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "name": "buildings", "local_crs": crs_note,
         "assumptions": meta, "features": feats}, indent=1))

    cover_feats = []
    for cls, polys in scene["cover_px"].items():
        for rings in polys:
            coords = []
            for r in rings:
                rm = to_m(r, wpx, hpx).round(2).tolist()
                coords.append(rm + [rm[0]])
            cover_feats.append({"type": "Feature", "properties": {"class": cls, "height_m": 0},
                                "geometry": {"type": "Polygon", "coordinates": coords}})
    for x, y, r, hgt in trees_m:
        cover_feats.append({"type": "Feature", "properties": {
            "class": "tree_representative", "crown_diameter_m": round(2 * r, 1),
            "height_m": round(hgt, 1), "height_is_guess": True},
            "geometry": {"type": "Point", "coordinates": [round(x, 2), round(y, 2)]}})
    (OUT / "landcover.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "name": "landcover", "local_crs": crs_note,
         "features": cover_feats}))

    html = (HERE / "viewer_template.html").read_text(encoding="utf-8")
    html = html.replace("__GLB_BASE64__", base64.b64encode(glb).decode("ascii"))
    html = html.replace("__META_JSON__", json.dumps(meta))
    (OUT / "viewer.html").write_text(html, encoding="utf-8")

    print(f"scene.glb {len(glb) / 1e6:.2f} MB, {len(feats)} buildings, {len(trees_m)} trees, "
          f"extent {wm:.0f} x {hm:.0f} m")


if __name__ == "__main__":
    main()
