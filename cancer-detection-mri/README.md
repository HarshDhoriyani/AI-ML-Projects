# 🧠 Brain Tumor Detection using MRI Images

A deep learning project that classifies brain MRI scans into **glioma**, **meningioma**, **pituitary tumor**, or **no tumor**, using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. Includes training, evaluation, single-image prediction, and an interactive Streamlit demo app.

> ⚠️ **Disclaimer:** This project is for educational and research purposes only. It is **not** a certified medical device and must **not** be used for real clinical diagnosis. Always consult a qualified radiologist/oncologist for medical decisions.

---

## 📁 Project Structure

```
cancer-detection-mri/
├── data/
│   └── README.md              # instructions for downloading the dataset
├── src/
│   ├── data_loader.py         # data loading & augmentation pipelines
│   ├── model.py                # CNN + transfer-learning model architectures
│   ├── train.py                 # training script
│   ├── evaluate.py              # evaluation: accuracy, confusion matrix, report
│   └── predict.py               # run inference on a single image
├── app/
│   └── app.py                   # Streamlit web demo
├── tests/
│   └── test_model.py            # basic unit tests
├── notebooks/
│   └── exploratory_analysis.md  # EDA notes/template
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset

This project uses the **Brain Tumor MRI Dataset** (publicly available on Kaggle), which contains ~7,000 MRI images across 4 classes:

| Class | Description |
|---|---|
| `glioma` | Tumor arising from glial cells |
| `meningioma` | Tumor arising from the meninges |
| `pituitary` | Tumor in the pituitary gland |
| `notumor` | Healthy brain scan |

**Download it here:** https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

After downloading, arrange it like this:

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── notumor/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── notumor/
```

See `data/README.md` for exact steps.

---

## ⚙️ Installation

```bash
git clone https://github.com/HarshDhoriyani/cancer-detection-mri.git
cd cancer-detection-mri
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the model
```bash
python src/train.py --data_dir data --epochs 25 --batch_size 32 --model_type cnn
```
Use `--model_type transfer` to fine-tune a pretrained EfficientNetB0 instead of the custom CNN.

Trained weights are saved to `models/best_model.h5` (created automatically).

### 2. Evaluate the model
```bash
python src/evaluate.py --data_dir data --model_path models/best_model.h5
```
Outputs accuracy, precision/recall/F1 per class, and a confusion matrix image saved to `assets/confusion_matrix.png`.

### 3. Predict on a single MRI image
```bash
python src/predict.py --image_path path/to/scan.jpg --model_path models/best_model.h5
```

### 4. Launch the interactive demo
```bash
streamlit run app/app.py
```
Upload an MRI image in the browser and get an instant prediction with confidence scores.

---

## 🧠 Model Architecture

Two options are provided in `src/model.py`:

1. **Custom CNN** — 4 convolutional blocks (Conv2D + BatchNorm + MaxPool) → Dense layers → Softmax(4). Lightweight, trains fast, good baseline.
2. **Transfer Learning** — EfficientNetB0 pretrained on ImageNet, with a custom classification head, fine-tuned on MRI data. Typically achieves higher accuracy with less data.


## 🛠️ Tech Stack

- Python 3.10+
- TensorFlow / Keras
- NumPy, Pandas
- scikit-learn (metrics)
- Matplotlib / Seaborn (visualization)
- Streamlit (demo app)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## 🙏 Acknowledgements

- Dataset: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (combines Figshare, SARTAJ, and Br35H datasets)
