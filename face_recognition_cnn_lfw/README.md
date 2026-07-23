# Face Recognition using CNN on LFW (Labeled Faces in the Wild)

A complete, from-scratch CNN pipeline for face **identification** (multi-class
classification of "who is this person") trained on the LFW dataset — a
benchmark of real-world, unposed photos ("in the wild": varying pose,
lighting, expression, occlusion, image quality).

## Project structure

```
face_recognition_cnn_lfw/
├── requirements.txt
├── README.md
├── src/
│   ├── data_loader.py   # downloads + preprocesses LFW via scikit-learn
│   ├── model.py         # CNN architecture (Keras Sequential)
│   ├── utils.py         # augmentation, plotting helpers
│   ├── train.py         # training entry point
│   ├── evaluate.py      # test-set metrics, confusion matrix, sample gallery
│   └── predict.py       # inference on a single new photo (+ face auto-crop)
└── models/              # created at train time: saved model, labels, plots
```

## How it works

1. **Data**: `sklearn.datasets.fetch_lfw_people` downloads and caches LFW,
   filtered to identities with at least `--min-faces` photos (default 70,
   giving 7 well-represented people — raise/lower this to trade off number
   of classes vs. difficulty). Images are grayscale, resized, normalized to
   `[0, 1]`, and split into stratified train/val/test sets.
2. **Model**: a 3-block CNN (Conv→BatchNorm→ReLU ×2 → MaxPool → Dropout per
   block, 32→64→128 filters) followed by global average pooling and a dense
   softmax head. Batch norm + dropout + L2 regularization are used because
   LFW subsets are small and prone to overfitting.
3. **Training**: light augmentation (rotation, shift, zoom, horizontal flip —
   no vertical flip, since faces are never upside down), class-balanced
   weighting (LFW is heavily imbalanced — e.g. far more George W. Bush photos
   than others), early stopping, and LR reduction on plateau.
4. **Evaluation**: accuracy, per-class precision/recall/F1, confusion matrix,
   and a visual gallery of correct (green) vs incorrect (red) predictions.
5. **Inference**: `predict.py` accepts an arbitrary photo, auto-detects and
   crops the face with OpenCV's Haar cascade, then classifies it.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Train (downloads LFW automatically on first run, ~200MB):

```bash
cd src
python train.py --min-faces 70 --epochs 40 --batch-size 32
```

Key `train.py` options:
| Flag | Default | Meaning |
|---|---|---|
| `--min-faces` | 70 | min photos/person to include as a class |
| `--resize` | 0.5 | LFW native-size scale factor |
| `--epochs` | 40 | training epochs (early stopping applies) |
| `--batch-size` | 32 | |
| `--lr` | 1e-3 | initial learning rate |
| `--output-dir` | `../models` | where model/plots/labels are saved |

Evaluate on the held-out test split:

```bash
python evaluate.py --model-dir ../models
```

Run inference on your own photo:

```bash
python predict.py --image /path/to/photo.jpg --model-dir ../models
```

## Notes on scaling up

- Increase `--min-faces` for an easier, higher-accuracy task (fewer, better-
  represented classes); decrease it for a harder, more realistic "in the
  wild" open-set-like problem (many classes, few examples each).
- For production-grade face recognition, an embedding approach (e.g.
  triplet loss / ArcFace producing a face embedding + nearest-neighbor
  matching) generalizes far better to *unseen* identities than this
  classification setup, which only recognizes identities seen during
  training. This project is deliberately structured as a plain CNN
  classifier for clarity and easy extension.
- GPU is strongly recommended for training; this will run on CPU but slower.
