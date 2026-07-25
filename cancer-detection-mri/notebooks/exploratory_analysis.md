# Exploratory Data Analysis — Template

Use this as a starting point for an `exploratory_analysis.ipynb` notebook once you have the dataset downloaded.

## Suggested sections

1. **Class distribution** — count images per class in Training/ and Testing/, check for imbalance.
2. **Sample visualization** — plot a grid of sample images from each class.
3. **Image properties** — check original resolutions, aspect ratios, color modes (grayscale vs RGB).
4. **Pixel intensity distributions** — histogram of pixel values per class, useful for spotting preprocessing needs.
5. **Duplicate/near-duplicate detection** — MRI datasets sometimes contain near-duplicate slices; consider checking with perceptual hashing.

## Example snippet to get started

```python
import os
import matplotlib.pyplot as plt
from PIL import Image

data_dir = "data/Training"
classes = os.listdir(data_dir)

fig, axes = plt.subplots(1, len(classes), figsize=(16, 4))
for ax, cls in zip(axes, classes):
    cls_dir = os.path.join(data_dir, cls)
    sample_file = os.listdir(cls_dir)[0]
    img = Image.open(os.path.join(cls_dir, sample_file))
    ax.imshow(img, cmap="gray")
    ax.set_title(cls)
    ax.axis("off")
plt.tight_layout()
plt.show()
```
