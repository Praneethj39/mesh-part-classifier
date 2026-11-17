"""Train and evaluate part classifiers on the generated dataset.

    python -m partclf.train --data data/parts.npz --out models/part_clf.joblib
"""
import argparse
import json
import time

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import describe, feature_names


def featurise(path):
    d = np.load(path, allow_pickle=True)
    X = np.stack([describe(v, f, seed=i) for i, (v, f) in enumerate(zip(d["verts"], d["faces"]))])
    return X, d["labels"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/parts.npz")
    ap.add_argument("--out", default="models/part_clf.joblib")
    ap.add_argument("--augment", type=int, default=3, help="augmented copies per mesh")
    a = ap.parse_args()

    t = time.time()
    X, y = featurise(a.data)
    print(f"features: {X.shape} in {time.time() - t:.1f}s")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        "grad_boost": GradientBoostingClassifier(random_state=0),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    results = {}
    for name, m in models.items():
        s = cross_val_score(m, Xtr, ytr, cv=cv, n_jobs=-1)
        results[name] = {"cv_mean": round(s.mean(), 4), "cv_std": round(s.std(), 4)}
        print(f"{name:14s} 5-fold CV acc {s.mean():.3f} +/- {s.std():.3f}")

    best = max(results, key=lambda k: results[k]["cv_mean"])
    model = models[best].fit(Xtr, ytr)
    pred = model.predict(Xte)
    acc = (pred == yte).mean()
    results["selected"] = best
    results["test_acc"] = round(float(acc), 4)
    labels = sorted(set(y))
    print(f"\nselected {best}; held-out test accuracy {acc:.3f}\n")
    print(classification_report(yte, pred, digits=3))
    cm = confusion_matrix(yte, pred, labels=labels)
    print("confusion matrix (rows=true):", labels)
    print(cm)
    if hasattr(model, "feature_importances_"):
        top = np.argsort(model.feature_importances_)[::-1][:8]
        results["top_features"] = [feature_names()[i] for i in top]
        print("top features:", results["top_features"])

    # final model: all data + noise/hole-augmented copies (see partclf.stress for why)
    from .stress import perturb
    d = np.load(a.data, allow_pickle=True)
    rng = np.random.default_rng(2)
    Xs, ys = [X], [y]
    for _ in range(a.augment):
        Xs.append(np.stack([describe(*perturb(v, f, rng, noise=rng.uniform(0, 0.05),
                                              hole=rng.uniform(0, 0.5)), seed=i)
                            for i, (v, f) in enumerate(zip(d["verts"], d["faces"]))]))
        ys.append(y)
    joblib.dump({"model": model.fit(np.concatenate(Xs), np.concatenate(ys)), "labels": labels}, a.out)
    with open(a.out.replace(".joblib", "_metrics.json"), "w") as f:
        json.dump({**results, "confusion_matrix": cm.tolist(), "labels": labels}, f, indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
