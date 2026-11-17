"""PointNet-style deep classifier on raw surface point clouds (PyTorch).

Unlike the hand-designed descriptor in features.py, the network learns its
own shape features straight from (x, y, z, normal) points:
shared per-point MLP -> symmetric max-pool -> MLP head (Qi et al., PointNet, 2017).
Each cloud is PCA-aligned and scale-normalised. The log of the real-world
size is fed in alongside the points, because a 1 cm bolt and a 40 cm wheel
look alike once normalised.

    python -m partclf.pointnet --data data/parts.npz --out models/pointnet.pt
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from .stress import perturb

N_POINTS = 1024


def mesh_to_cloud(verts, faces, n=N_POINTS, rng=None):
    """Area-weighted surface samples with normals, PCA-aligned, unit-scaled.
    Returns (n,6) float32 array and log10 of the part's size."""
    rng = rng or np.random.default_rng(0)
    v, f = np.asarray(verts, np.float64), np.asarray(faces, np.int64)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    ok = area > 1e-14
    tri, cross, area = tri[ok], cross[ok], area[ok]
    nrm = cross / (2 * area[:, None])
    idx = rng.choice(len(tri), n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    a, b, c = tri[idx, 0], tri[idx, 1], tri[idx, 2]
    pts = (1 - s)[:, None] * a + (s * (1 - r2))[:, None] * b + (s * r2)[:, None] * c
    n_pts = nrm[idx]
    pts -= pts.mean(0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    vecs = vecs[:, ::-1]
    pts, n_pts = pts @ vecs, n_pts @ vecs
    size = np.ptp(pts, axis=0).max() + 1e-9
    pts /= size
    return np.concatenate([pts, n_pts], 1).astype(np.float32), np.float32(np.log10(size))


class PointNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Conv1d(6, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 + 1, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, pts, log_size):          # pts: (B, N, 6)
        x = self.point_mlp(pts.transpose(1, 2))
        x = x.max(dim=2).values                 # order-invariant global feature
        return self.head(torch.cat([x, log_size[:, None]], 1))


def random_flip(batch, rng):
    """PCA axes have arbitrary sign, so train with random sign flips."""
    sign = rng.choice([-1.0, 1.0], size=(len(batch), 1, 3)).astype(np.float32)
    out = batch.copy()
    out[:, :, :3] *= sign
    out[:, :, 3:] *= sign
    return out


def build_set(d, ids, rng, aug_copies=0):
    X, S, Y = [], [], []
    for i in ids:
        variants = [(d["verts"][i], d["faces"][i])]
        for _ in range(aug_copies):
            variants.append(perturb(d["verts"][i], d["faces"][i], rng,
                                    noise=rng.uniform(0, 0.05), hole=rng.uniform(0, 0.5)))
        for v, f in variants:
            c, s = mesh_to_cloud(v, f, rng=rng)
            X.append(c); S.append(s); Y.append(d["labels"][i])
    return np.stack(X), np.array(S), np.array(Y)


def predict(model, X, S, batch=128):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[k:k + batch]), torch.from_numpy(S[k:k + batch])))
    return torch.cat(out).softmax(1).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/parts.npz")
    ap.add_argument("--out", default="models/pointnet.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--augment", type=int, default=3)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    d = np.load(a.data, allow_pickle=True)
    labels = sorted(str(x) for x in set(d["labels"]))
    to_i = {l: k for k, l in enumerate(labels)}
    tr, te = train_test_split(np.arange(len(d["labels"])), test_size=0.25,
                              stratify=d["labels"], random_state=0)   # same split as stress.py
    t = time.time()
    Xtr, Str, Ytr = build_set(d, tr, rng, a.augment)
    ytr = np.array([to_i[l] for l in Ytr])
    print(f"train clouds: {Xtr.shape} ({time.time() - t:.0f}s)")

    model = PointNet(len(labels))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    t = time.time()
    for ep in range(a.epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tot = 0.0
        for k in range(0, len(perm), 64):
            b = perm[k:k + 64]
            xb = torch.from_numpy(random_flip(Xtr[b], rng))
            xb[:, :, :3] += 0.005 * torch.randn_like(xb[:, :, :3])
            loss = loss_fn(model(xb, torch.from_numpy(Str[b])), torch.from_numpy(ytr[b]))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if ep % 10 == 9 or ep == 0:
            print(f"epoch {ep + 1:3d}  loss {tot / len(perm):.3f}  ({time.time() - t:.0f}s)")

    # evaluate on the same perturbations as partclf.stress
    erng = np.random.default_rng(1)
    rows = []
    for name, kw in [("clean", {}), ("noise 1%", {"noise": 0.01}), ("noise 3%", {"noise": 0.03}),
                     ("noise 5%", {"noise": 0.05}), ("20% faces missing", {"hole": 0.2}),
                     ("50% faces missing", {"hole": 0.5})]:
        X, S = zip(*(mesh_to_cloud(*perturb(d["verts"][i], d["faces"][i], erng, **kw), rng=erng)
                     for i in te))
        p = predict(model, np.stack(X), np.array(S)).argmax(1)
        acc = float((np.array(labels)[p] == d["labels"][te]).mean())
        rows.append({"test": name, "acc_pointnet": round(acc, 3)})
        print(f"{name:20s} PointNet acc {acc:.3f}")
    torch.save({"state": model.state_dict(), "labels": labels}, a.out)
    json.dump(rows, open(a.out.replace(".pt", "_results.json"), "w"), indent=2)
    print("saved", a.out, f"({sum(p.numel() for p in model.parameters()):,} parameters)")


if __name__ == "__main__":
    main()
