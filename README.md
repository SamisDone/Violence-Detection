# Violence Detection with AnomLite

A lightweight deep learning model for **multiclass crime detection** in video surveillance using the UCF-Crime dataset.

## Overview

**AnomLite** is a hybrid neural network that combines:

- **MobileNetV2** for efficient spatial feature extraction
- **LSTM** for temporal pattern recognition

This architecture achieves strong performance with only **~11 million parameters**, making it suitable for real-time deployment on resource-constrained devices.

## Performance

| Metric       | Score |
| ------------ | ----- |
| **Accuracy** | 79%   |
| **ROC AUC**  | 0.97  |
| **F1-Macro** | 0.75  |

## Crime Categories (14 Classes)

- Abuse, Arrest, Arson, Assault, Burglary
- Explosion, Fighting, Road Accidents, Robbery
- Shooting, Shoplifting, Stealing, Vandalism
- Normal Videos

## Dataset

Uses the [UCF-Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) containing real-world surveillance videos.

## Quick Start

### Option 1: Run on Kaggle (Recommended)

1. Open a new Kaggle notebook
2. Add the [UCF-Crime Dataset](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) to your notebook
3. Upload `AnomLite_Multiclass_Clean.ipynb`
4. Run all cells - paths are auto-detected

### Option 2: Run Locally

1. Clone this repository
2. Download the dataset (~11 GB):
   ```bash
   pip install kaggle
   kaggle datasets download -d odins0n/ucf-crime-dataset --unzip
   ```
3. Place `Train/` and `Test/` folders in the project directory
4. Run with Jupyter Notebook

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- matplotlib
- numpy
- PIL

## Architecture

```
Input Frames → MobileNetV2 (Spatial) → LSTM (Temporal) → FC → 14-Class Output
```

## License

This project uses the UCF-Crime dataset which is under CC0-1.0 license.

## Acknowledgments

- UCF-Crime Dataset creators
- PyTorch and torchvision teams
