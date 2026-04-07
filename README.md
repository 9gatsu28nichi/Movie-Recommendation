# 🎬 CineMatch 2.0: AI-Powered Movie Discovery Platform

> **ITCS227: Introduction to Data Science & Artificial Intelligence**
> A production-grade streaming-style application delivering weighted statistical rankings, NLP semantic "vibe" search, and interactive 3D cinematic relationship mapping.

CineMatch 2.0 is an advanced content-based recommendation engine built with **Streamlit** and **Python**. It features a state-of-the-art AI core that transcends simple keyword matching to understand the "vibe" of cinema through high-dimensional embeddings and interactive network analysis.

---

## 🚀 Key Features (CineMatch 2.0)

### 🧠 AI Semantic "Vibe" Search
- Leverages **Sentence Transformers** (`all-MiniLM-L6-v2`) to perform natural language interpretation of movie plots and synopses.
- Search for "vibes" rather than just titles (e.g., *"lonely space travel"* or *"gritty urban survival"*).

### 🕸️ 3D Cinematic Discovery Explorer
- Interactive **Plotly 3D Network Graph** visualizing the relationships between 10,000+ movies.
- Explore movie clusters and "bridges" between genres in a dynamic 3D space.

### ⚖️ Weighted Statistical Ranking
- Implements the **IMDB Weighted Rating formula** to ensure global quality standards.
- Balances vote volume against average ratings to highlight objective "Viral Hits" and "All-Time Classics."

### 📂 Hybrid Watchlist Discovery
- Personalized recommendation engine that aggregates the "Cinematic DNA" of your entire watchlist.
- Discovers the perfect "next watch" by cross-referencing multiple movie affinities simultaneously.

### 🎨 Premium Cinematic UI
- **Glassmorphic Design**: Modern dark-mode interface with translucent cards and smooth transitions.
- **Dynamic Detail Panels**: Auto-navigating movie details that rise to the top for a seamless browsing experience.
- **Auto-Poster Discovery**: Multi-stage fallback system ensuring every movie has a visual poster, even for obscure titles.

---

## 🔬 Technical Research Record

The project includes a comprehensive **[MovieRecommendation.ipynb](MovieRecommendation.ipynb)** notebook. This is not just a draft, but a **Professional Technical Report** structured into 6 modules:
1. **Executive Summary**: Project abstract and problem statement.
2. **Data Engineering**: "The Soup" generation and feature engineering.
3. **Exploratory Data Analysis (EDA)**: Cinematic trend visualizations and correlation matrices.
4. **CineMatch 2.0 AI Core**: Full mathematical documentation of the algorithms.
5. **Model Validation**: Technical metrics, confusion matrices, and logic assessment.
6. **Final Conclusion**: Research findings and future roadmap.

---

## 📦 Installation & Quick Start

Follow these steps to set up the environment and launch the CineMatch platform:

### 1. Clone the repository
```bash
git clone <repo-url>
cd Movie-Recommendation
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Initialize the Dataset
The project utilizes the **TMDB-IMDB Merged Dataset**. We provide an automated installation script:
```bash
python install_data.py
```
> [!TIP]
> This script uses `kagglehub` to securely download and verify the 100MB+ dataset into the `archive/` folder.

### 5. Launch the Platform
```bash
# This will open CineMatch in your default web browser
streamlit run app.py
```

---

## 🛠️ Tech Stack
- **UI/UX**: Streamlit, Vanilla CSS, Plotly.
- **Core AI**: Sentence Transformers, Scikit-Learn.
- **Data**: Pandas, Numpy, NLTK.
- **Visualization**: Seaborn, Matplotlib.

## 📚 References & Dataset
- **Dataset**: [TMDB-IMDB Merged Movies Dataset - Kaggle](https://www.kaggle.com/datasets/ggtejas/tmdb-imdb-merged-movies-dataset)
- **Course**: ITCS227: Introduction to Data Science, Faculty of ICT, Mahidol University.

---
*Created by Jirathiwat Sun for ITCS227.*
