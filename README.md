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
# This will open CineMatch in .env your default web browser
.venv\Scripts\streamlit.exe run app.py
```

### 🔧 Troubleshooting: AI Engine Initializing on CPU
The CineMatch AI engine is designed to **use your GPU (CUDA) automatically** if available. If the application states it is loading the engine on the CPU, PyTorch could not detect a compatible GPU setup. This typically happens if the standard CPU version of PyTorch was installed, or if your Python version lacks pre-compiled GPU binaries.

To fix this and force hardware acceleration (NVIDIA GPUs only), run the following commands in your virtual environment:
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
*(This command forces pip to download the CUDA 12.1 enabled binaries instead of defaulting to the CPU versions from PyPI).*

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
