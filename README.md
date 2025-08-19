# 🎵 Spotify Song Popularity Prediction with PyTorch

A machine learning project to **predict the popularity score (0–100)** of a song from its audio features using **PyTorch**.  
This project demonstrates hands-on deep learning for tabular data: cleaning, exploratory analysis, model training, and evaluation.

---

## 🚀 Project Overview

The aim of this project is to:

- Load and preprocess the **Ultimate Spotify Tracks DB** dataset
- Explore patterns between audio features and song popularity
- Build a **PyTorch regression model** (MLP) to predict popularity
- Experiment with neural network architectures and hyperparameters
- Evaluate model performance with **RMSE** and **MAE**
- Provide a helper function for predicting new songs

This work demonstrates end-to-end skills in **data preprocessing**, **EDA**, **model building**, and **machine learning evaluation**.

---

## 📦 Features

- Cleaned & preprocessed Spotify dataset
- Train/validation/test split with feature scaling
- Flexible **PyTorch MLP regressor** with:
  - Dropout for regularization
  - Early stopping for generalization
- Performance metrics: **RMSE** and **MAE**
- Correlation analysis and feature engineering (`energy_danceability`)
- Prediction helper for single or batch inputs

---

## 🧠 Technologies

- **Python 3**
- **PyTorch**
- **pandas**, **NumPy**, **scikit-learn**
- **Matplotlib**, **Seaborn** for EDA
- **Jupyter Notebook**

---

## 📂 Project Structure

````plaintext
PREDICTING_SPOTIFY_SONG_POPULARITY/
├── dataset/
│   └── SpotifyFeatures.csv
├── notebooks/
│   ├── final.ipynb
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation

````

## 📊 Dataset

- **Source:** [Ultimate Spotify Tracks DB (Kaggle)](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
- **Rows:** ~232k
- **Columns:** Audio features, track metadata, and popularity score

**Features used:**
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
- `mode`
- `time_signature`
- `energy_danceability`

**Target:**
- `popularity` (integer, 0–100)

---

## 📈 Workflow

1. **Data Preparation**
   - Drop text/high-cardinality columns (IDs, artist name, genre)
   - Convert `duration_ms` → minutes (apply log transform if skewed)
   - Scale numeric features

2. **Exploratory Data Analysis (EDA)**
   - Plot feature distributions
   - Correlation heatmap (e.g., `loudness` vs `energy` correlation = 0.82)
   - Popularity vs audio features

3. **Modeling**
   - Flexible PyTorch MLP
   - Randomized hyperparameter search
   - Early stopping for robust training

4. **Evaluation**
   - Compare against **baseline mean predictor**
   - Compute **Train / Validation / Test RMSE**
   - Plot learning curves to check for over/underfitting

5. **Prediction Function**
   - Input: Pandas DataFrame or Python dict of features
   - Output: Predicted popularity score(s)

---

## 📊 Example Results

| Metric | Validation | Test   |
| ------ | ---------- | ------ |
| RMSE   | ~12.5      | ~12.9  |
| MAE    | ~9.8       | ~10.1  |

_(Exact values may vary slightly with random seeds and tuning.)_

---

## ⚙️ Installation & Usage

1. **Clone the repository**
```bash
git clone https://github.com/spotify-pytorch-project.git
cd spotify-pytorch-project
````

## ⚙️ Installation & Usage

2. **Install dependencies**
```bash
pip install -r requirements.txt
````
## ▶️ Run Jupyter Notebook

```bash
jupyter notebook notebooks/02_model_pytorch.ipynb
````

## 📝 Notes

- Popularity is influenced by many **non-audio factors** such as artist fame, release timing, and playlist exposure.
- This project predicts popularity **only from audio features**.
- Model performance can be further improved with additional metadata, such as:
  - Artist popularity
  - Release year
  - Playlist counts

---

## 👥 Project Team  

This project was developed by:  
- **Ashok**  
- **Priti**  
- **Jeel**  

