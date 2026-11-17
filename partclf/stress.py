"""Robustness tests: how does accuracy hold up when meshes are noisier, have
holes, or when the model may not use absolute size?

    python -m partclf.stress --data data/parts.npz
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import describe, feature_names


def perturb(v, f, rng, noise=0.0, hole=0.0):
    v = v + rng.normal(0, noise * np.ptp(v, axis=0).max(), v.shape)
    if hole:
        f = f[rng.random(len(f)) > hole]
    return v, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/parts.npz")
    a = ap.parse_args()
    d = np.load(a.data, allow_pickle=True)
    idx = np.arange(len(d["labels"]))
    tr, te = train_test_split(idx, test_size=0.25, stratify=d["labels"], random_state=0)
    y = d["labels"]
    Xtr = np.stack([describe(d["verts"][i], d["faces"][i], seed=i) for i in tr])
    no_size = [k for k, n in enumerate(feature_names()) if n not in ("log_size", "log_tris")]

    def model(cols):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(Xtr[:, cols], y[tr])

    full, ablate = model(slice(None)), model(no_size)
    # augmented model: each training mesh also seen with random noise and holes
    arng = np.random.default_rng(2)
    Xaug, yaug = [Xtr], [y[tr]]
    for _ in range(3):
        Xaug.append(np.stack([describe(*perturb(d["verts"][i], d["faces"][i], arng,
                                                noise=arng.uniform(0, 0.05), hole=arng.uniform(0, 0.5)),
                                       seed=i) for i in tr]))
        yaug.append(y[tr])
    aug = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(
        np.concatenate(Xaug), np.concatenate(yaug))
    rng = np.random.default_rng(1)
    rows = []
    for name, kw in [("clean", {}), ("noise 1%", {"noise": 0.01}), ("noise 3%", {"noise": 0.03}),
                     ("noise 5%", {"noise": 0.05}), ("20% faces missing", {"hole": 0.2}),
                     ("50% faces missing", {"hole": 0.5})]:
        Xte = np.stack([describe(*perturb(d["verts"][i], d["faces"][i], rng, **kw), seed=i) for i in te])
        r = {"test": name,
             "acc_full": round(float((full.predict(Xte) == y[te]).mean()), 3),
             "acc_without_size": round(float((ablate.predict(Xte[:, no_size]) == y[te]).mean()), 3),
             "acc_augmented": round(float((aug.predict(Xte) == y[te]).mean()), 3)}
        rows.append(r)
        print(f"{name:20s} full={r['acc_full']:.3f}  without-size={r['acc_without_size']:.3f}  augmented={r['acc_augmented']:.3f}")
    json.dump(rows, open("models/stress_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
