"""Rotation-invariant shape descriptor for a single mesh part.

Input is plain numpy (verts Nx3, faces Mx3 triangle indices), so it works on
meshes from Blender, trimesh, or any loader. 37 features in total:

  * PCA shape: eigenvalue ratios, normalised extents, log absolute size
  * surface/volume: area and enclosed-volume ratios vs. the oriented box
  * open-boundary ratio (sheets such as glass are open, solids are closed)
  * radial symmetry about the thinnest axis (wheels, knobs, bolts)
  * area-weighted histogram of |normal . axis_k| for each PCA axis
  * D2 shape distribution (Osada et al. 2002): histogram of distances
    between random surface point pairs
"""
import numpy as np

N_BINS_NORMAL = 4
N_BINS_D2 = 12


def _pca(points):
    c = points.mean(0)
    x = points - c
    w, v = np.linalg.eigh(np.cov(x.T) + 1e-12 * np.eye(3))
    order = np.argsort(w)[::-1]
    return c, w[order], v[:, order]


def _sample_surface(tri, area, n, rng):
    p = area / area.sum()
    idx = rng.choice(len(tri), size=n, p=p)
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
    return (1 - s)[:, None] * a + (s * (1 - r2))[:, None] * b + (s * r2)[:, None] * c


def describe(verts, faces, seed=0):
    verts = np.asarray(verts, float)
    faces = np.asarray(faces, int)
    rng = np.random.default_rng(seed)
    tri = verts[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    keep = area > 1e-12
    tri, cross, area = tri[keep], cross[keep], area[keep]
    normals = cross / (2 * area[:, None])

    pts = _sample_surface(tri, area, 2048, rng)
    c, w, axes = _pca(pts)
    local = (pts - c) @ axes
    ext = local.max(0) - local.min(0) + 1e-9
    size = ext.max()

    # surface & volume relative to oriented bounding box
    total_area = area.sum()
    box_area = 2 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2])
    vol = abs(np.einsum("ij,ij->i", tri[:, 0] - c, np.cross(tri[:, 1] - c, tri[:, 2] - c)).sum()) / 6
    fill = vol / np.prod(ext)

    # open boundary: edges used by exactly one face
    e = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    open_ratio = (counts == 1).sum() / max(len(counts), 1)

    # radial symmetry about the thinnest axis
    radial = np.linalg.norm(local[:, :2], axis=1)
    rad_cv = radial.std() / (radial.mean() + 1e-9)
    ang = np.arctan2(local[:, 1], local[:, 0])
    sector = np.histogram(ang, bins=8, range=(-np.pi, np.pi))[0] / len(ang)
    sector_unevenness = sector.std() * 8

    # normal orientation histograms
    nh = []
    for k in range(3):
        d = np.abs(normals @ axes[:, k])
        h = np.histogram(d, bins=N_BINS_NORMAL, range=(0, 1), weights=area)[0] / total_area
        nh.extend(h)

    # D2 distribution, scale-normalised
    i, j = rng.integers(0, len(pts), (2, 4096))
    d2 = np.linalg.norm(pts[i] - pts[j], axis=1) / size
    d2h = np.histogram(d2, bins=N_BINS_D2, range=(0, 1))[0] / len(d2)

    f = [
        w[1] / w[0], w[2] / w[0],
        ext[1] / size, ext[2] / size, np.log10(size),
        total_area / box_area, fill, open_ratio,
        rad_cv, sector_unevenness,
        np.log10(len(faces) + 1),
        d2.mean(), d2.std(),
    ]
    f += nh + list(d2h)
    return np.array(f, dtype=np.float32)


def feature_names():
    base = ["eig21", "eig31", "ext2", "ext3", "log_size", "area_ratio", "fill", "open_ratio",
            "radial_cv", "sector_uneven", "log_tris", "d2_mean", "d2_std"]
    base += [f"n{k}_{b}" for k in range(3) for b in range(N_BINS_NORMAL)]
    base += [f"d2_{b}" for b in range(N_BINS_D2)]
    return base
