# 🎵 Spotify Song Popularity Prediction using PyTorch

A machine learning project to **predict the popularity score (0–100)** of a song based on its audio features using **PyTorch**.  
This project is built for hands-on practice with **deep learning on tabular data**, including data cleaning, exploratory analysis, model training, and performance evaluation.

---

## 🚀 Project Overview

The aim of this project is to:

- Load and preprocess the **Ultimate Spotify Tracks DB** dataset.
- Explore patterns between audio features and song popularity.
- Build a **PyTorch regression model** to predict popularity.
- Experiment with neural network architectures and hyperparameters.
- Evaluate the model’s performance using RMSE and MAE.
- Provide explanations for predictions via feature importance and sensitivity analysis.

This work demonstrates skills in **data preprocessing**, **EDA**, **model building**, and **machine learning evaluation**.

---

## 📦 Features

- Cleaned & preprocessed Spotify dataset.
- Train/validation/test split with feature scaling.
- PyTorch **MLP regressor** with:
  - Batch Normalization
  - Dropout
  - Early stopping & LR scheduling
- Performance metrics: **RMSE** and **MAE**.
- Correlation & feature importance analysis (Permutation + Gradients).
- Single-row prediction function with local explanation.
- Optional genre-wise performance breakdown.

---

## 🧠 Technologies

- **Python 3**
- **PyTorch**
- **pandas**, **NumPy**, **scikit-learn**
- **Matplotlib**, **Seaborn** (EDA & charts)
- **Jupyter Notebook** for interactive development

---

## 📂 Project Structure

PREDICTING_SPOTIFY_SONG_POPULARITY/
├── dataset/ SpotifyFeatures.csv
├── notebooks/
│ ├── 01_eda.ipynb # Data cleaning & exploratory analysis
│ └── 02_model_pytorch.ipynb # Model building & evaluation
├── artifacts/ # Saved models, scalers, configs
├── src/
│ ├── data_utils.py # Data loading, cleaning, splitting
│ ├── model.py # PyTorch model definitions
│ ├── train.py # Training & evaluation loops
├── requirements.txt # Dependencies
├── README.md # Project documentation

---

## 📊 Dataset

**Source:** [Ultimate Spotify Tracks DB — Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)  
**Rows:** ~232k  
**Columns:** Audio features, track metadata, and popularity score.

**Features used for prediction:**

- `danceability`
- `energy`
- `acousticness`
- `instrumentalness`
- `liveness`
- `loudness`
- `speechiness`
- `tempo`
- `valence`
- `duration_min`

**Target:**

- `popularity` (integer, 0–100)

---

## 📈 Workflow

1. **Data Preparation**

   - Load CSV data into Pandas
   - Drop unnecessary ID and categorical columns (e.g., artist name, track ID)
   - Convert `duration_ms` to minutes
   - Handle outliers and scale numeric features

2. **EDA**

   - Distribution of popularity
   - Correlation heatmap
   - Popularity by genre
   - Feature distributions by popularity tier

3. **Modeling**

   - PyTorch MLP with multiple hidden layers
   - Train with MSE loss
   - Evaluate using RMSE & MAE
   - Early stopping to avoid overfitting

4. **Evaluation**

   - Test set performance
   - Predicted vs Actual plot
   - Residuals analysis
   - Feature importance and sensitivity

5. **Prediction Function**
   - Pass new song feature values to get a popularity prediction
   - Local explanation for prediction

---

## 📊 Example Results

| Metric | Validation | Test   |
| ------ | ---------- | ------ |
| RMSE   | XX.XXX     | XX.XXX |
| MAE    | XX.XXX     | XX.XXX |

_(Exact numbers depend on final tuned model.)_

---

## ⚙️ Installation & Usage

1. **Clone the repository**

````bash
git clone https://github.com/yourusername/spotify-popularity.git
cd spotify-popularity

2. **Install the dependencies**
```bash
pip install -r requirements.txt

3. ** Run Notebooks**
jupyter notebook notebook.ipynb

📝 Notes

Popularity is influenced by non-audio factors (artist fame, release timing).

Predictions are based solely on audio features.

Model performance can be improved with additional metadata (e.g., artist popularity, playlist count).




````
