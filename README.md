# Convolutional Neural Networks (CNN) for Medical Image Classification

A comparative study of four convolutional neural network architectures for classifying chest X-rays as **NORMAL** or **PNEUMONIA**, with a strong focus on what actually matters in a clinical setting: not just accuracy, but the trade-off between sensitivity (recall) and precision.

## What this project does

Using a public chest X-ray dataset (anteroposterior views, labeled by expert physicians, pre-split into train/validation/test), four progressively different models are trained and evaluated under a shared protocol (Adam optimizer, cross-entropy loss, 5 epochs, batch size 32):

1. **CNN C3D16 (simple)** — a single 3×3 convolutional layer with 16 filters, ReLU, half-padding, 2×2 max-pooling, and dropout (p=0.5). Learns only basic patterns; shows overfitting from the second epoch onward.
2. **CNN C3D16 with two convolutional layers + Batch Normalization** — normalizing activations before the activation function. Achieves a striking 100% recall on the NORMAL class (never misses a healthy patient), at the cost of missing more pneumonia cases.
3. **Deeper CNN (32 + 64 filters, no Batch Normalization, lower learning rate)** — captures more complex patterns but trains less stably; ends up classifying almost everything as PNEUMONIA — useful if the priority is never missing a sick patient, but too imbalanced for general use.
4. **VGG-style model trained from scratch** — three convolutional blocks feeding into a dense classification head. With a tuned decision threshold (0.4), this model achieves the **best overall balance: 93% recall on PNEUMONIA** without excessive false positives — the strongest model of the four for real-world use.

## Why recall, not just accuracy

In a medical screening context, missing a real pneumonia case (a false negative) is far more costly than a false alarm. The analysis explicitly foregrounds **recall on the PNEUMONIA class** as the primary metric, tracing how each architectural choice (filter depth, batch normalization, learning rate, decision threshold) shifts the trade-off between catching every sick patient and avoiding excessive false positives — and shows that the "best" model isn't the one with the highest raw accuracy, but the one with the most clinically appropriate error profile.

## Key results

| Model | Key characteristic | Behavior |
|---|---|---|
| Simple CNN (16 filters) | Minimal capacity | Overfits from epoch 2; limited pattern learning |
| CNN + Batch Normalization | Two conv layers, normalized activations | 100% recall on NORMAL, but misses more PNEUMONIA cases |
| Deeper CNN (32+64 filters) | No batch norm, low learning rate | Classifies nearly everything as PNEUMONIA — high sensitivity, low specificity |
| **VGG (from scratch, threshold 0.4)** | Three conv blocks + dense head | **Best balance: 93% recall on PNEUMONIA** with controlled false positives |

## Tech stack

- Python
- TensorFlow / Keras
- Kaggle API (dataset download)
- matplotlib (training curves, confusion matrices)

## Data

- **Chest X-Ray Images (Pneumonia)** dataset (Kaggle: `paultimothymooney/chest-xray-pneumonia`), pre-split into train/validation/test sets.

## Notes

This project began as a group activity, where each member completed their own individual implementation before the group met to compare approaches and refine conclusions together. The code, analysis, and interpretation in this notebook are my own individual work. The original assignment brief is not included here as it belongs to the issuing institution.
