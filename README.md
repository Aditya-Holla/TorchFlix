# TorchFlix – Binary Classification Model

TorchFlix is a movie preference prediction system built using PyTorch. It predicts whether a user would like or dislike a given movie based on content features like genres, keywords, and actors. The model is trained using metadata from the MovieLens and TMDB datasets and achieves over 65% accuracy on a held-out test set.

---

## Overview

- **Type**: Binary classification (like/dislike)
- **Framework**: PyTorch
- **Dataset**: MovieLens + TMDB
- **Features**: Genres, plot keywords, top-billed actors
- **Output**: 1 (like) or 0 (dislike)

---

## Features Used

- Multi-hot encoded genres from `movies_metadata.csv`
- Extracted keywords from `keywords.csv` (stringified JSON)
- Actor names from `credits.csv` (stringified JSON)
- Ratings from `ratings_small.csv` used to generate binary labels  
  (e.g. rating ≥ 4.0 = like, rating < 4.0 = dislike)

---

## Model Architecture

- PyTorch-based feedforward neural network
- Input: concatenated multi-hot feature vector for each movie
- Loss: Binary Cross Entropy
- Optimizer: Adam
- Performance: ~65% accuracy on test set

---

## File Structure

- `torchflix_model.py` – model architecture and training loop
- `data_preprocessing.ipynb` – feature engineering and merging datasets
- `inference_demo.ipynb` – prediction examples using trained model
- `data/` – directory for input CSVs (not included in repo due to size)

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/torchflix.git
   cd torchflix
