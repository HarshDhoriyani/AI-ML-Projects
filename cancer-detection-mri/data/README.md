# Dataset Setup

This project does not ship the MRI images (they're large and licensed for redistribution via Kaggle only). Follow these steps:

## Option A: Manual download
1. Go to https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
2. Click "Download" (requires a free Kaggle account)
3. Unzip the archive into this `data/` folder so you end up with:
   ```
   data/Training/glioma/...
   data/Training/meningioma/...
   data/Training/pituitary/...
   data/Training/notumor/...
   data/Testing/glioma/...
   data/Testing/meningioma/...
   data/Testing/pituitary/...
   data/Testing/notumor/...
   ```

## Option B: Kaggle CLI
```bash
pip install kaggle
# place your kaggle.json API token in ~/.kaggle/kaggle.json first
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset -p data --unzip
```

## Note
This `data/` directory (except this README) is git-ignored — do not commit MRI images to the repo, both for size and licensing reasons.
