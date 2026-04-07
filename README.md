# 🎬 CineMatch: AI Movie Recommendation System

This project is a part of the **ITCS227: Introduction To Data Science** course. 
CineMatch is a full-featured, visually stunning web application built with Streamlit, leveraging modern Data Science techniques and Natural Language Processing to bring accurate, personalized movie recommendations right to your screen.

## ✨ Features
- **AI-Powered Recommendations:** Utilizes `sentence-transformers` (`all-MiniLM-L6-v2`) to compute semantic similarities across movies by analyzing synopses, genres, keywords, cast, and directors simultaneously.
- **Dynamic Dashboard:** Includes a beautiful data exploration dashboard powered by `Plotly` to visualize IMDB distributions, genre trends, release years, and revenue stats.
- **Advanced Filtering and Preferences:** Generate recommendations without a specific seed movie—just select your favorite genres, preferred minimum IMDB rating, and release decade to instantly uncover hidden gems.
- **Beautiful Cinematic UI:** Built with custom CSS to provide a dark cinematic theme, dynamic hover effects, stat cards, and smooth animations mimicking modern premium streaming platforms.

## 📦 Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Movie-Recommendation
   ```
2. **Install dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Dataset Setup:**
   Download the TMDB-IMDB Merged Movies Dataset from the reference link below and extract `TMDB_IMDB_Movies_Dataset.csv` into the `archive/` folder.
   ```text
   Movie-Recommendation/
   ├── archive/
   │   └── TMDB_IMDB_Movies_Dataset.csv
   ```
4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📚 Reference
- Dataset: [TMDB-IMDB Merged Movies Dataset - Kaggle](https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset)
