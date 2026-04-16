# 🎬 CineMatch 2.0: AI-Powered Movie Discovery Platform

> **ITCS227: Introduction to Data Science & Artificial Intelligence**
> A production-grade streaming-style application delivering weighted statistical rankings, NLP semantic "vibe" search, and interactive 3D cinematic relationship mapping.

CineMatch 2.0 is an advanced content-based recommendation engine built with **Streamlit** and **Python**. It features a state-of-the-art AI core that transcends simple keyword matching to understand the "vibe" of cinema through high-dimensional embeddings and interactive network analysis.

---

## 🚀 Key Features (CineMatch Oracle Edition)

### 🧠 CineMatch Oracle 🔮
- **Intent-Based Discovery**: The engine now understands the "Soul" of your search. It detects if you want **Visual Style** (Neon, Noir), **Emotional Impact** (Heartbreaking, Happy), or **Narrative Complexity** (Twists, Plot).
- **Synonym Intelligence**: Integrated **NLTK WordNet** to expand queries. Searching for "Astronaut" automatically discovers "Space," "Galaxy," and "NASA" connections.

### ⚡ Mood Fusion Engine
- **Thematic Overlaps**: Finds movies that sit at the intersection of multiple vibes (e.g., *"Beautiful [Visual] but Sad [Emotional]"*).
- **Fusion Scoring**: Prioritizes "Resonance" matches where every part of a complex query is satisfied.

### 💎 Hidden Gem Radar
- **Critic's Darlings**: Automatically surfaces highly-rated independent films (>8.0 IMDB) with lower vote counts that might otherwise be buried by blockbusters.
- **Fair Discovery**: Implements **Short-Overview Hydration** to ensure indie films with brief descriptions have a strong "Vibe Signal."

### 🏗️ Global Quality Priority
- **Strict IMDB Ranking**: All semantic matches are globally sorted from highest to lowest quality. Page 1 always contains the "Best of the Best."
- **Unlimited Scanning**: The search engine now scans the entire **436k movie dataset** without retrieval caps.

### 🕸️ 3D Cinematic Discovery Explorer
- Interactive **Plotly 3D Network Graph** visualizing relationships between 436,000+ movies.

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

### 4. Initialize the Dataset & AI Models
CineMatch requires both the movie database and the high-dimensional AI model files. We provide an automated installation and health-check script:
```bash
python install_data.py
```
> [!TIP]
> This script does two critical tasks:
> 1. Uses `kagglehub` to securely download the 270MB+ movie dataset into `archive/`.
> 2. **Health Check**: Verifies that your AI model files (`models/`) are correctly downloaded via Git LFS. If it detects "pointer" files, it will attempt to hydrate them automatically.

### 5. Launch the Platform
```bash
# This will open CineMatch in your default web browser
streamlit run app.py
# This will open CineMatch in .env your default web browser
.venv\Scripts\streamlit.exe run app.py
```

### 🔧 Troubleshooting

#### 1. AI Engine Initializing on CPU
The CineMatch AI engine is designed to **use your GPU (CUDA) automatically** if available. If the application states it is loading the engine on the CPU, PyTorch could not detect a compatible GPU setup.

To fix this and force hardware acceleration (NVIDIA GPUs only):
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 2. Redundant AI Training / Missing Models
If the application spends a long time "Encoding movies" even though the `models/` folder exists, you likely have **Git LFS pointers** instead of actual model binaries.

**Solution:**
1. Install [Git LFS](https://git-lfs.github.com/).
2. Run `git lfs pull` in the project root.
3. Run `python install_data.py` to verify the fix.

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
