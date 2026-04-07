import os
import shutil
import glob
import subprocess

def install_dataset():
    print("🚀 CineMatch 2.0: Initializing Dataset...")
    try:
        import kagglehub
    except ImportError:
        print("📦 Installing kagglehub...")
        subprocess.check_call(["pip", "install", "kagglehub"])
        import kagglehub

    # 1. Download from Kaggle
    print("⬇️ Downloading TMDB-IMDB Merged Dataset...")
    path = kagglehub.dataset_download('ggtejas/tmdb-imdb-merged-movies-dataset')
    
    # 2. Setup archive directory
    os.makedirs('archive', exist_ok=True)
    
    # 3. Locate and copy CSV
    csv_files = glob.glob(os.path.join(path, '*.csv'))
    if not csv_files:
        print("❌ Error: No CSV file found in the download.")
        return
        
    target_path = os.path.join('archive', 'TMDB_IMDB_Movies_Dataset.csv')
    shutil.copy(csv_files[0], target_path)
    
    print(f"✅ Success! Dataset installed at: {target_path}")
    print("🎥 You can now run 'streamlit run app.py' to launch CineMatch 2.0.")

if __name__ == "__main__":
    install_dataset()
