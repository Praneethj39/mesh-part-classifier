# mesh-part-classifier

Labels 3D mesh parts (body, wheel, glass, mirror, door, seat, bolt) from
geometry alone, so assets with names like `Mesh.041` can be organised
automatically. It plugs into [blender-asset-optimizer](../blender-asset-optimizer)
for automatic part naming.

## How it works

1. **Data**: `partclf/generate.py` builds 1,050 labelled parts (150 per class) procedurally in Blender. Proportions, tessellation, orientation and noise are randomised.
2. **Features** (`partclf/features.py`, 37 per mesh, rotation-invariant):
   * PCA eigenvalue ratios and extents
   * area and volume ratios against the oriented bounding box
   * open-boundary ratio
   * radial symmetry about the thinnest axis
   * area-weighted normal-orientation histograms
   * D2 shape distribution (Osada et al., 2002)
3. **Feature models**: logistic regression, random forest and gradient boosting, compared with 5-fold CV. The final model is trained on clean data plus 3 copies of each mesh with random noise (0–5%) and missing faces (0–50%).
4. **Deep model** (`partclf/pointnet.py`): a PointNet-style network in PyTorch (77k parameters). It learns features straight from 1,024 surface points (position + normal) per part:
   * shared per-point MLP (6→64→128→256)
   * max-pool, so point order doesn't matter
   * MLP head
   * clouds are PCA-aligned and scale-normalised, with the log of real-world size fed in separately
   * trained 40 epochs on CPU (~15 min) with the same noise/hole augmentation, plus random axis flips

## Results

| Test | Accuracy |
|---|---|
| 5-fold CV, held-out split (synthetic) | 100% |
| Parts from 13 *separately generated, deliberately broken* car assets (via the optimizer) | **104 / 104** |

Synthetic shapes are easy, so the robustness tests (`python -m partclf.stress`) matter more:

| Perturbation | Feature model | Without size features | Feature model + augmentation | **PointNet** |
|---|---|---|---|---|
| clean | 100% | 100% | 99.2% | **100%** |
| 3% vertex noise | 73.0% | 71.9% | 99.2% | **100%** |
| 5% vertex noise | 60.1% | 62.7% | 98.1% | **99.6%** |
| 20% faces missing | 71.1% | 75.3% | 100% | **100%** |
| 50% faces missing | 16.3% | 17.5% | 99.2% | **100%** |

Both final models labelled 104/104 parts on the 13 unseen car assets. Pass either file to the optimizer: `--classifier models/part_clf.joblib` or `--classifier models/pointnet.pt`.

**Trade-off:** PointNet needs no hand-designed features and is the most robust, but it needs PyTorch and ~15 min of training. The feature model trains in seconds, and each of its features can be explained.

Lessons:
* The plain model collapsed on scan-like defects. Training with augmentation fixed this.
* In the optimizer, classifying **after** lossless polygon reduction misread a 2-triangle windscreen as a door. So classification now runs on the full-resolution mesh before reduction.

## Usage

```bash
pip install bpy==5.3.0 numpy scikit-learn joblib trimesh torch
python -m partclf.generate --per-class 150 --out data/parts.npz
python -m partclf.train  --data data/parts.npz --out models/part_clf.joblib
python -m partclf.stress --data data/parts.npz
python -m partclf.pointnet --data data/parts.npz --out models/pointnet.pt
python -m partclf.predict models/part_clf.joblib some_part.obj
```

## Limits and next steps

* All training data is synthetic. The next step is fine-tuning on real labelled parts (e.g. ShapeNet/PartNet car parts).
* PointNet looks at each point independently. Newer point-cloud models (DGCNN, Point Transformer) and pretrained 3D–language models (e.g. OpenShape, Uni3D) also capture local structure and could label parts zero-shot.
