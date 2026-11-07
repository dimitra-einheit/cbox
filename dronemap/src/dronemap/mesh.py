"""Mesh writers with no extra dependencies: GLB (glTF 2.0 binary), OBJ+MTL, binary STL.

Coordinates are passed in as local east/north/up metres. glTF and OBJ are Y-up,
so we store (east, up, -north), which Blender imports back to east/north/up.
"""
from __future__ import annotations

import io
import json
import struct

import numpy as np

FLOAT, UINT32 = 5126, 5125
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963


def heightfield(z: np.ndarray, res: float, x0: float, y0: float, origin=(0.0, 0.0, 0.0), step: int = 1,
                valid: np.ndarray | None = None):
    """Triangulate a north-up grid (row 0 = north). (x0, y0) = map coords of cell (0,0) centre.
    Returns verts (N,3) ENU relative to origin, faces (M,3), and the map x/y of each vertex."""
    zs = z[::step, ::step]
    h, w = zs.shape
    ok = np.isfinite(zs) if valid is None else (valid[::step, ::step] & np.isfinite(zs))
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    mx = x0 + cols * res * step
    my = y0 - rows * res * step
    idx = -np.ones((h, w), dtype=np.int64)
    idx[ok] = np.arange(ok.sum())
    verts = np.stack([mx[ok] - origin[0], my[ok] - origin[1], zs[ok] - origin[2]], axis=1)
    a = idx[:-1, :-1]
    b = idx[:-1, 1:]
    c = idx[1:, :-1]
    d = idx[1:, 1:]
    t1 = np.stack([a, c, b], axis=-1).reshape(-1, 3)  # counter-clockwise seen from above
    t2 = np.stack([b, c, d], axis=-1).reshape(-1, 3)
    faces = np.concatenate([t1, t2])
    faces = faces[(faces >= 0).all(axis=1)]
    return verts.astype(np.float64), faces.astype(np.int64), mx[ok], my[ok]


def _enu_to_yup(v: np.ndarray) -> np.ndarray:
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def _jpeg(rgb: np.ndarray, quality: int = 88) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8), "RGB").save(buf, format="JPEG",
                                                                                     quality=quality)
    return buf.getvalue()


class GlbBuilder:
    def __init__(self):
        self.bin = bytearray()
        self.g = {"asset": {"version": "2.0", "generator": "dronemap"}, "scene": 0, "scenes": [{"nodes": []}],
                  "nodes": [], "meshes": [], "materials": [], "accessors": [], "bufferViews": [],
                  "buffers": [], "images": [], "textures": [], "samplers": []}

    def _view(self, data: bytes, target=None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        off = len(self.bin)
        self.bin.extend(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        self.g["bufferViews"].append(v)
        return len(self.g["bufferViews"]) - 1

    def _accessor(self, arr: np.ndarray, comp: int, typ: str, target, minmax=False) -> int:
        view = self._view(arr.tobytes(), target)
        acc = {"bufferView": view, "componentType": comp, "count": int(arr.shape[0]), "type": typ}
        if minmax:
            acc["min"] = [float(x) for x in arr.min(axis=0)]
            acc["max"] = [float(x) for x in arr.max(axis=0)]
        self.g["accessors"].append(acc)
        return len(self.g["accessors"]) - 1

    def texture(self, rgb: np.ndarray) -> int:
        view = self._view(_jpeg(rgb))
        self.g["images"].append({"bufferView": view, "mimeType": "image/jpeg"})
        if not self.g["samplers"]:
            self.g["samplers"].append({"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071})
        self.g["textures"].append({"sampler": 0, "source": len(self.g["images"]) - 1})
        return len(self.g["textures"]) - 1

    def material(self, name: str, color=(1, 1, 1, 1), texture: int | None = None, blend: bool = False) -> int:
        pbr = {"baseColorFactor": [float(c) for c in color], "metallicFactor": 0.0, "roughnessFactor": 1.0}
        if texture is not None:
            pbr["baseColorTexture"] = {"index": texture}
        m = {"name": name, "pbrMetallicRoughness": pbr, "doubleSided": True}
        if blend:
            m["alphaMode"] = "BLEND"
        self.g["materials"].append(m)
        return len(self.g["materials"]) - 1

    def mesh(self, name: str, verts_enu: np.ndarray, faces: np.ndarray, material: int,
             uv: np.ndarray | None = None, colors: np.ndarray | None = None) -> None:
        if len(faces) == 0:
            return
        pos = _enu_to_yup(verts_enu).astype(np.float32)
        attrs = {"POSITION": self._accessor(pos, FLOAT, "VEC3", ARRAY_BUFFER, minmax=True)}
        if uv is not None:
            attrs["TEXCOORD_0"] = self._accessor(uv.astype(np.float32), FLOAT, "VEC2", ARRAY_BUFFER)
        if colors is not None:
            attrs["COLOR_0"] = self._accessor(colors.astype(np.float32), FLOAT,
                                              "VEC4" if colors.shape[1] == 4 else "VEC3", ARRAY_BUFFER)
        ind = self._accessor(faces.astype(np.uint32).reshape(-1), UINT32, "SCALAR", ELEMENT_ARRAY_BUFFER)
        self.g["meshes"].append({"name": name, "primitives": [{"attributes": attrs, "indices": ind,
                                                               "material": material, "mode": 4}]})
        self.g["nodes"].append({"name": name, "mesh": len(self.g["meshes"]) - 1})
        self.g["scenes"][0]["nodes"].append(len(self.g["nodes"]) - 1)

    def save(self, path: str) -> str:
        while len(self.bin) % 4:
            self.bin.append(0)
        self.g["buffers"] = [{"byteLength": len(self.bin)}]
        for k in ("images", "textures", "samplers"):
            if not self.g[k]:
                del self.g[k]
        js = json.dumps(self.g, separators=(",", ":")).encode("utf-8")
        while len(js) % 4:
            js += b" "
        total = 12 + 8 + len(js) + 8 + len(self.bin)
        with open(path, "wb") as f:
            f.write(struct.pack("<III", 0x46546C67, 2, total))
            f.write(struct.pack("<II", len(js), 0x4E4F534A))
            f.write(js)
            f.write(struct.pack("<II", len(self.bin), 0x004E4942))
            f.write(bytes(self.bin))
        return path


def uv_for(mx, my, tex_transform, tex_shape):
    """glTF UVs (origin top-left) of map coordinates on a north-up texture."""
    h, w = tex_shape[:2]
    u = (np.asarray(mx) - tex_transform.c) / (w * tex_transform.a)
    v = (tex_transform.f - np.asarray(my)) / (h * -tex_transform.e)
    return np.stack([np.clip(u, 0, 1), np.clip(v, 0, 1)], axis=1)


def write_obj(path: str, verts_enu: np.ndarray, faces: np.ndarray, uv: np.ndarray | None = None,
              texture_rgb: np.ndarray | None = None) -> str:
    """OBJ (Y-up) + MTL + JPEG texture, ready for Blender's default OBJ import."""
    import os

    base = os.path.splitext(path)[0]
    name = os.path.basename(base)
    v = _enu_to_yup(verts_enu)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# dronemap terrain (Y-up; east=X, up=Y, north=-Z), metres relative to the georef origin\n")
        if texture_rgb is not None:
            f.write(f"mtllib {name}.mtl\nusemtl terrain\n")
        np.savetxt(f, v, fmt="v %.4f %.4f %.4f")
        if uv is not None:
            np.savetxt(f, np.stack([uv[:, 0], 1.0 - uv[:, 1]], axis=1), fmt="vt %.6f %.6f")
            ff = faces + 1
            np.savetxt(f, np.stack([ff[:, 0], ff[:, 0], ff[:, 1], ff[:, 1], ff[:, 2], ff[:, 2]], axis=1),
                       fmt="f %d/%d %d/%d %d/%d")
        else:
            np.savetxt(f, faces + 1, fmt="f %d %d %d")
    if texture_rgb is not None:
        with open(base + ".mtl", "w", encoding="utf-8") as f:
            f.write(f"newmtl terrain\nKd 1 1 1\nmap_Kd {name}_texture.jpg\n")
        with open(base + "_texture.jpg", "wb") as f:
            f.write(_jpeg(texture_rgb, 90))
    return path


def read_obj_vertices(path: str) -> np.ndarray:
    """Vertices of an OBJ converted back to ENU (east, north, up)."""
    pts = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                p = line.split()
                pts.append((float(p[1]), float(p[2]), float(p[3])))
    if not pts:
        raise ValueError(f"No vertices in {path}")
    v = np.array(pts)
    return np.stack([v[:, 0], -v[:, 2], v[:, 1]], axis=1)


def write_stl_terrain(path: str, z: np.ndarray, res: float, size_mm: float = 180.0, exaggeration: float = 2.0,
                      base_mm: float = 4.0) -> dict:
    """Watertight 3D-printable block of a terrain grid (NaNs filled), scaled to fit size_mm."""
    from .raster import fill_nan_nearest

    zf = fill_nan_nearest(z)
    h, w = zf.shape
    span = max(h, w) * res
    s = size_mm / span  # mm per metre
    top = (zf - np.nanmin(zf)) * s * exaggeration + base_mm
    xs = np.arange(w) * res * s
    ys = (h - 1 - np.arange(h)) * res * s
    X, Y = np.meshgrid(xs, ys)
    P = np.stack([X, Y, top], axis=-1)
    # top surface (normals up)
    a, b, c, d = P[:-1, :-1], P[:-1, 1:], P[1:, 1:], P[1:, :-1]
    tops = np.concatenate([np.stack([a, d, c], axis=-2).reshape(-1, 3, 3),
                           np.stack([a, c, b], axis=-2).reshape(-1, 3, 3)])
    walls = []
    perim = []
    # edges walked clockwise seen from above: north (W->E), east (N->S), south (E->W), west (S->N)
    for line in (P[0, :], P[:, -1], P[-1, ::-1], P[::-1, 0]):
        for i in range(len(line) - 1):
            p, q = line[i], line[i + 1]
            pb, qb = p.copy(), q.copy()
            pb[2] = qb[2] = 0.0
            walls.append((p, qb, pb))  # outward normals
            walls.append((p, q, qb))
            perim.append(pb)
    # bottom: fan from the centre through every perimeter vertex (shares edges with the walls -> watertight)
    centre = np.array([(xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2, 0.0])
    bottom = [(centre, perim[i], perim[(i + 1) % len(perim)]) for i in range(len(perim))]  # normals down
    allt = np.concatenate([tops, np.array(walls).reshape(-1, 3, 3), np.array(bottom).reshape(-1, 3, 3)])
    n = np.cross(allt[:, 1] - allt[:, 0], allt[:, 2] - allt[:, 0])
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(nl == 0, 1, nl)
    rec = np.zeros(len(allt), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = allt
    with open(path, "wb") as f:
        f.write(b"dronemap terrain".ljust(80, b" "))
        f.write(struct.pack("<I", len(allt)))
        f.write(rec.tobytes())
    return {"triangles": int(len(allt)), "size_mm": (round(w * res * s, 1), round(h * res * s, 1)),
            "scale": f"1:{1000 / s:,.0f}", "height_mm": float(top.max())}
