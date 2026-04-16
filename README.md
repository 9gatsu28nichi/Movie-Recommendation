# 🎬 CineMatch 2.0: The Cinematic Oracle


> **ITCS227: Introduction to Data Science & Artificial Intelligence**
> A production-grade movie discovery platform featuring GPU-accelerated semantic search, 3D cinematic mapping, and complex hybrid recommendation logic.

![CineMatch Platform Showcase](https://img.shields.io/badge/Status-Production--Ready-success?style=for-the-badge&logo=powerbi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA--Accelerated-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit--Glassmorphism-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

CineMatch 2.0 is an advanced discovery engine that transcends simple keyword matching. By utilizing high-dimensional embeddings and a custom **Hybrid Scoring Engine**, it understands the "Soul" and "Vibe" of over **436,000 movies** to deliver near-perfect recommendations.

---

## 📂 Project Architecture

```mermaid
graph TD
    subgraph Data_Layer [Data Engineering]
        K[Kaggle Dataset] -->|install_data.py| R[Raw Data Folder]
        R -->|Preprocessing| S[Metadata Soup]
    end

    subgraph AI_Core [CineMatch Oracle Engine]
        S -->|all-MiniLM-L6-v2| E[GPU-Accelerated Vectorization]
        E -->|Persistence| M[Saved Models/Embeddings]
    end

    subgraph Web_UI [Platform Interface]
        M -->|app.py| U[Streamlit Visual Layout]
        U -->|Query| Q[Hybrid Search Engine]
        Q -->|Hybrid Ranking| D[3D Network Graph & List Results]
    end
```

---

## 🧠 The AI Core: How it Works

CineMatch 2.0 uses a specialized **Metadata Soup** (Title + Overview + Genre + Keywords + Era Tags) to create a unique "fingerprint" for every film.

### ⚡ Hybrid Scoring Engine
Unlike standard search engines, CineMatch uses a **50/50 Hybrid Score**:
1.  **Semantic Similarity (50%)**: Measured using Cosine Similarity of query embeddings against the 436k movie vectors.
2.  **Weighted Statistical Quality (50%)**: Calculated using the IMDB weighted rating formula:
    $$\text{Score} = \frac{v}{v+m}R + \frac{m}{v+m}C$$
    *(Where $v$ is votes, $m$ is minimum votes, $R$ is average rating, and $C$ is the mean across all movies).*

### 🔮 Oracle Features
- **Mood Fusion Intent Parser**: Automatically detects if your query matches **Visual Styles** (Neon, Noir), **Emotional Impacts** (Heartbreaking), or **Narrative Complexity** (Twists) to adjust weights on the fly.
- **Hidden Gem Radar**: Specifically surfaces independent films with high ratings but lower vote counts to prevent "Blockbuster Bias."
- **3D discovery explorer**: An interactive Plotly map visualizing relationships between films in a 3D orbital space.

---

## 📦 Professional Setup & Launch

The project has been organized into a standardized directory structure for production reliability.

### 1. Initialize Environment
```bash
# Clone and enter
git clone https://github.com/9gatsu28nichi/Movie-Recommendation.git
cd Movie-Recommendation

# Setup virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# MacOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. High-Signal Data Installation
Our automated script hydrates the dataset directly from Kaggle into the project's logic core.
```bash
python scripts/install_data.py
```

### 3. Launch the Platform
```bash
streamlit run src/app.py
```

---

## 🚀 GPU Acceleration Setup

If the application displays **🖥️ CPU Mode** in the sidebar, or runs slowly, you need to install the CUDA-enabled version of PyTorch.

### 🛡️ Recommended Fix (Forced Index Override)
Standard `pip install` commands can sometimes default to CPU versions even if `--extra-index-url` is provided. To force hardware acceleration:

1. **Activate Environment**:
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Nuclear Reset & Force CUDA**:
   ```powershell
   pip uninstall torch torchvision torchaudio -y
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

> [!IMPORTANT]
> **Python 3.14 Compatibility**: If you are using Python 3.14 (amd64) on Windows, pre-compiled CUDA wheels may not be available for the current Torch version. **It is highly recommended to use Python 3.11 or 3.12** for stable GPU acceleration.

> [!TIP]
> Use `python scripts/check_cuda.py` to identify exactly why your engine is defaulting to CPU.

---

## 📂 Folder Structure

| Directory | Purpose |
| :--- | :--- |
| **`data/`** | Raw and processed movie datasets (.csv) |
| **`models/`** | Pre-trained embeddings and persistent AI states |
| **`src/`** | The platform source code and custom CSS styling |
| **`scripts/`** | Automation scripts for data fetching and maintenance |
| **`notebooks/`**| Technical research reports and logic validation |

---

## 🛠️ Tech Stack
- **AI/ML**: Sentence-Transformers (BERT), Scikit-Learn, PyTorch.
- **Frontend**: Streamlit, Custom Glassmorphism CSS, Plotly.
- **Analytics**: Pandas, NumPy, NLTK (WordNet Expansion).

---
*Developed by Jirathiwat Sun for ITCS227: Introduction to Data Science.*
