# CIFAR-10 Image Classification using CNN

A convolutional neural network (CNN) built with TensorFlow/Keras to classify
images from the CIFAR-10 dataset into 10 categories: airplane, automobile,
bird, cat, deer, dog, frog, horse, ship, truck.

## Project Structure

```
cifar10-cnn/
├── model.py          # CNN architecture definition
├── train.py           # Training script (loads data, trains, saves model)
├── evaluate.py        # Evaluation script (accuracy, confusion matrix, sample predictions)
├── predict.py          # Run inference on a single custom image
├── requirements.txt    # Python dependencies
├── saved_models/       # Trained models are saved here
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model
```bash
python train.py --epochs 25 --batch-size 64
```
This will:
- Download CIFAR-10 automatically (via `tf.keras.datasets`)
- Normalize the data and apply data augmentation
- Train the CNN with early stopping and learning-rate reduction
- Save the best model to `saved_models/cifar10_cnn.keras`
- Save training curves to `saved_models/training_history.png`

### 2. Evaluate the model
```bash
python evaluate.py
```
This will:
- Load the saved model
- Report test accuracy / loss
- Print a classification report (precision/recall/F1 per class)
- Save a confusion matrix plot
- Save a grid of sample predictions

### 3. Predict on your own image
```bash
python predict.py --image path/to/your_image.jpg
```

## Model Architecture

The CNN uses a stack of 3 convolutional blocks (Conv2D + BatchNorm + Conv2D +
BatchNorm + MaxPool + Dropout), increasing filters (32 → 64 → 128), followed
by a dense classification head with dropout for regularization.

## Expected Performance

With the default settings, this architecture typically reaches **~82-87%
test accuracy** after 25-40 epochs on CIFAR-10. Accuracy can be improved
further with deeper architectures (e.g., ResNet-style blocks), longer
training, or more aggressive data augmentation.

## Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy, Matplotlib, scikit-learn
