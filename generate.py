"""Procedurally generate a labelled dataset of car-part meshes with Blender.

Each class is built from parametric primitives with randomised proportions,
tessellation density, orientation and noise, so the classifier has to learn
shape rather than memorise one model.

    python -m partclf.generate --per-class 150 --out data/parts.npz
"""
import argparse
import math
import random

import bpy  # noqa: F401  (loads bmesh/mathutils)
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Vector

CLASSES = ["body", "wheel", "glass", "mirror", "door", "seat", "bolt"]


def _box(bm, sx, sy, sz, cuts=0, center=(0, 0, 0)):
    g = bmesh.ops.create_cube(bm, size=1.0)
    vs = g["verts"]
    for v in vs:
        v.co = Vector((v.co.x * sx + center[0], v.co.y * sy + center[1], v.co.z * sz + center[2]))
    if cuts:
        es = list({e for v in vs for e in v.link_edges})
        bmesh.ops.subdivide_edges(bm, edges=es, cuts=cuts, use_grid_fill=True)


def body(r):
    bm = bmesh.new()
    L, W = r.uniform(3.5, 5.2), r.uniform(1.6, 2.0)
    _box(bm, L, W, r.uniform(0.6, 0.9), cuts=r.randint(2, 12))
    if r.random() < 0.7:  # cabin on top
        _box(bm, L * r.uniform(0.4, 0.6), W * 0.9, r.uniform(0.4, 0.6), cuts=r.randint(1, 6),
             center=(r.uniform(-0.4, 0.2), 0, r.uniform(0.55, 0.7)))
    if r.random() < 0.5:
        bmesh.ops.bevel(bm, geom=bm.edges[:], offset=r.uniform(0.02, 0.1), segments=2,
                        affect="EDGES", profile=0.5)
    return bm


def wheel(r):
    bm = bmesh.new()
    rad, wid = r.uniform(0.28, 0.45), r.uniform(0.18, 0.32)
    seg = r.choice([24, 32, 48, 64, 128])
    bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=rad, radius2=rad, depth=wid)
    if r.random() < 0.5:  # hub / rim detail
        bmesh.ops.create_cone(bm, cap_ends=True, segments=seg, radius1=rad * 0.6,
                              radius2=rad * 0.6, depth=wid * 1.15)
    if r.random() < 0.4:
        bmesh.ops.subdivide_edges(bm, edges=bm.edges[:], cuts=1)
    return bm


def glass(r):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=r.randint(2, 40), y_segments=r.randint(2, 30),
                          size=r.uniform(0.4, 0.9))
    sx = r.uniform(0.8, 2.0)
    bend = r.uniform(0, 0.25)
    for v in bm.verts:
        v.co.x *= sx
        v.co.z = bend * v.co.x ** 2  # curved windscreen
    return bm


def mirror(r):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=r.choice([12, 16, 24, 32]),
                              v_segments=r.choice([8, 12, 16]), radius=r.uniform(0.08, 0.15))
    a, b = r.uniform(0.4, 0.8), r.uniform(1.0, 1.6)
    for v in bm.verts:
        v.co.x *= a
        v.co.y *= b
        if v.co.x < 0:
            v.co.x *= 0.3  # flattened face where the glass sits
    return bm


def door(r):
    bm = bmesh.new()
    _box(bm, r.uniform(0.9, 1.3), r.uniform(0.04, 0.09), r.uniform(0.8, 1.1), cuts=r.randint(1, 8))
    bow = r.uniform(0.0, 0.08)
    for v in bm.verts:
        v.co.y += bow * (1 - (v.co.x / 0.65) ** 2)
    return bm


def seat(r):
    bm = bmesh.new()
    w = r.uniform(0.45, 0.6)
    _box(bm, w, r.uniform(0.45, 0.6), r.uniform(0.1, 0.18), cuts=r.randint(0, 3))
    _box(bm, w, r.uniform(0.08, 0.14), r.uniform(0.55, 0.8), cuts=r.randint(0, 3),
         center=(0, -0.28, 0.35))
    if r.random() < 0.6:  # headrest
        _box(bm, w * 0.5, 0.1, 0.18, center=(0, -0.28, 0.82))
    return bm


def bolt(r):
    bm = bmesh.new()
    rad = r.uniform(0.004, 0.012)
    bmesh.ops.create_cone(bm, cap_ends=True, segments=r.choice([8, 12, 16, 24]),
                          radius1=rad, radius2=rad, depth=rad * r.uniform(3, 8))
    hex_ = bmesh.ops.create_cone(bm, cap_ends=True, segments=6, radius1=rad * 1.8,
                                 radius2=rad * 1.8, depth=rad * 1.2)
    for v in hex_["verts"]:
        v.co.z += rad * 2.5
    return bm


BUILDERS = dict(body=body, wheel=wheel, glass=glass, mirror=mirror, door=door, seat=seat, bolt=bolt)


def to_arrays(bm, r):
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    rot = Euler([r.uniform(0, 2 * math.pi) for _ in range(3)]).to_matrix().to_4x4()
    s = r.uniform(0.9, 1.1)
    bm.transform(rot @ Matrix.Diagonal((s, s, s, 1)))
    jitter = r.uniform(0, 0.003)
    v = np.array([tuple(x.co) for x in bm.verts], dtype=np.float32)
    v += np.random.default_rng(r.randint(0, 10**9)).normal(0, jitter, v.shape).astype(np.float32) \
        * np.ptp(v, axis=0).max()
    bm.verts.index_update()
    f = np.array([[x.index for x in face.verts] for face in bm.faces], dtype=np.int32)
    bm.free()
    return v, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--out", default="data/parts.npz")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    r = random.Random(a.seed)
    verts, faces, labels = [], [], []
    for label in CLASSES:
        for _ in range(a.per_class):
            v, f = to_arrays(BUILDERS[label](r), r)
            verts.append(v)
            faces.append(f)
            labels.append(label)
    np.savez_compressed(a.out, verts=np.array(verts, dtype=object),
                        faces=np.array(faces, dtype=object), labels=np.array(labels))
    print(f"saved {len(labels)} meshes ({a.per_class} x {len(CLASSES)} classes) -> {a.out}")


if __name__ == "__main__":
    main()
