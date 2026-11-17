"""Inference helpers. Works on numpy arrays, mesh files (via trimesh), or
Blender objects (used by blender-asset-optimizer to auto-name parts)."""
import sys

import joblib
import numpy as np

from .features import describe


class PartClassifier:
    def __init__(self, path):
        d = joblib.load(path)
        self.model, self.labels = d["model"], d["labels"]

    def predict_arrays(self, verts, faces):
        x = describe(verts, faces)[None]
        p = self.model.predict_proba(x)[0]
        i = int(np.argmax(p))
        return self.model.classes_[i], float(p[i])

    def predict_objects(self, objs, min_conf=0.5):
        """Blender mesh objects -> {object name: label}; low-confidence -> 'part'."""
        out = {}
        for o in objs:
            me = o.data
            me.calc_loop_triangles()
            v = np.empty(len(me.vertices) * 3, np.float32)
            me.vertices.foreach_get("co", v)
            f = np.empty(len(me.loop_triangles) * 3, np.int32)
            me.loop_triangles.foreach_get("vertices", f)
            if len(f) == 0:
                continue
            label, conf = self.predict_arrays(v.reshape(-1, 3), f.reshape(-1, 3))
            out[o.name] = label if conf >= min_conf else "part"
        return out


if __name__ == "__main__":  # python -m partclf.predict models/part_clf.joblib some_part.obj
    import trimesh
    clf = PartClassifier(sys.argv[1])
    for p in sys.argv[2:]:
        m = trimesh.load(p, force="mesh")
        print(p, *clf.predict_arrays(m.vertices, m.faces))


class PointNetClassifier(PartClassifier):
    """Same interface as PartClassifier, backed by the PyTorch PointNet model."""

    def __init__(self, path):
        import torch
        from .pointnet import PointNet
        ck = torch.load(path, map_location="cpu")
        self.labels = ck["labels"]
        self.net = PointNet(len(self.labels))
        self.net.load_state_dict(ck["state"])
        self.net.eval()

    def predict_arrays(self, verts, faces):
        import torch
        from .pointnet import mesh_to_cloud
        c, s = mesh_to_cloud(verts, faces)
        with torch.no_grad():
            p = self.net(torch.from_numpy(c)[None], torch.tensor([s])).softmax(1)[0].numpy()
        i = int(np.argmax(p))
        return self.labels[i], float(p[i])


def load(path):
    return PointNetClassifier(path) if str(path).endswith(".pt") else PartClassifier(path)
