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

### Run on Kaggle

1. Open `AnomLite_Multiclass_Clean.ipynb` in Kaggle
2. Add the UCF-Crime dataset to your notebook
3. Run all cells

### Run Locally

1. Clone this repository
2. Download the dataset and update paths in the notebook
3. Run with Jupyter Notebook

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
