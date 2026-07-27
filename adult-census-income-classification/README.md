# Adult Census Income Classification

Predict whether an individual's annual income exceeds **$50K/year** based on U.S. Census data, using the classic [UCI Adult / Census Income dataset](https://archive.ics.uci.edu/dataset/2/adult).

This is an end-to-end binary classification project: data ingestion, cleaning, preprocessing, model training, evaluation, and inference — structured as a clean, reusable Python package rather than a single notebook.

---

## 📊 Problem Statement

Given demographic and employment attributes (age, education, occupation, hours worked per week, etc.), predict the binary target:

- `<=50K` — income at or below $50,000/year
- `>50K` — income above $50,000/year

## 📁 Project Structure

```
adult-census-income-classification/
├── data/                   # raw/processed data (gitignored, downloaded on demand)
├── models/                 # saved trained models (gitignored)
├── notebooks/
│   └── eda.ipynb           # exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── config.py           # paths, constants, column definitions
│   ├── data_loader.py      # downloads / loads the raw dataset
│   ├── preprocessing.py    # cleaning + feature engineering pipeline
│   ├── train.py            # trains and saves the model
│   ├── evaluate.py         # evaluation metrics + plots
│   └── predict.py          # run inference on new data
├── tests/
│   └── test_preprocessing.py
├── main.py                 # CLI entry point: train / evaluate / predict
├── requirements.txt
├── .gitignore
└── README.md
```

## 🧠 Dataset

The dataset is the **Adult Census Income** dataset (a.k.a. "Census Income"), originally extracted from the 1994 U.S. Census database. It contains ~48,842 rows and 14 features:

| Feature | Description |
|---|---|
| age | Age of the individual |
| workclass | Type of employer (Private, Self-emp, Government, etc.) |
| fnlwgt | Census sampling weight |
| education | Highest level of education |
| education-num | Education level, numeric |
| marital-status | Marital status |
| occupation | Occupation category |
| relationship | Relationship to household |
| race | Race |
| sex | Sex |
| capital-gain | Capital gains recorded |
| capital-loss | Capital losses recorded |
| hours-per-week | Hours worked per week |
| native-country | Country of origin |
| **income** (target) | `<=50K` or `>50K` |

The dataset is downloaded automatically the first time you run training (from the UCI Machine Learning Repository), so there's no need to manually download or commit large data files.

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/adult-census-income-classification.git
cd adult-census-income-classification
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 🚀 Usage

**Train a model** (downloads data automatically on first run):

```bash
python main.py train --model random_forest
```

Supported `--model` options: `logistic_regression`, `random_forest`, `gradient_boosting`.

**Evaluate the saved model** on the held-out test set:

```bash
python main.py evaluate
```

**Predict** on new data (CSV with the same 14 feature columns, no target column):

```bash
python main.py predict --input path/to/new_data.csv --output predictions.csv
```

## 📈 Results

The training script prints and saves a classification report + confusion matrix + ROC-AUC. Example (Random Forest, default settings):

| Metric | Score |
|---|---|
| Accuracy | ~0.86 |
| Precision (>50K) | ~0.75 |
| Recall (>50K) | ~0.63 |
| ROC-AUC | ~0.91 |

*(Exact numbers depend on train/test split and hyperparameters — rerun `evaluate` to reproduce.)*

## 🛠️ Tech Stack

- Python 3.9+
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- joblib (model persistence)

## 🧪 Tests

```bash
pytest tests/
```

## 🙌 Acknowledgements

- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository, Adult Data Set. University of California, Irvine, School of Information and Computer Sciences.
