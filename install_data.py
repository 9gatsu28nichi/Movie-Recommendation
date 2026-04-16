import os
import shutil
import kagglehub
import sys

# Force UTF-8 encoding for console output to handle emojis on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        # Fallback for older python versions if necessary
        pass

def install_data():
    print("🎬 CineMatch 2.0: Data Installation Oracle")
    print("------------------------------------------")
    
    dataset_handle = "ggtejas/tmdb-imdb-merged-movies-dataset"
    target_dir = "archive"
    target_filename = "TMDB_IMDB_Movies_Dataset.csv"
    target_path = os.path.join(target_dir, target_filename)

    # Ensure target directory exists
    if not os.path.exists(target_dir):
        print(f"📁 Creating {target_dir}/ directory...")
        os.makedirs(target_dir, exist_ok=True)

    print(f"📡 Requesting high-signal dataset from Kaggle Hub...")
    try:
        # Download latest version
        # This will show progress bars in the console automatically
        download_path = kagglehub.dataset_download(dataset_handle)
        
        print(f"✅ Download complete. Verifying payload...")
        
        # Search for any CSV file in the downloaded path (handles different naming versions)
        csv_files = [f for f in os.listdir(download_path) if f.endswith('.csv')]
        
        if not csv_files:
            print("❌ Error: No CSV files found in the downloaded dataset.")
            sys.exit(1)
            
        source_file = csv_files[0]
        source_path = os.path.join(download_path, source_file)
        
        print(f"🧪 Hydrating {target_filename} into project archive...")
        
        # Copy to target location with the expected name
        shutil.copy2(source_path, target_path)
        
        print("------------------------------------------")
        print("🎯 MISSION SUCCESS SUCCESS")
        print(f"📍 Dataset Location: {target_path}")
        print(f"📊 Size: {os.path.getsize(target_path) / (1024*1024):.2f} MB")
        print("🚀 You are now ready to launch CineMatch with 'streamlit run app.py'")
        print("------------------------------------------")

    except Exception as e:
        print(f"❌ Critical Error during installation: {e}")
        print("💡 Tip: Ensure you have an active internet connection and 'kagglehub' installed.")
        sys.exit(1)

def check_lfs():
    """Checks if the model files are Git LFS pointers and attempts to pull them."""
    print("\n🔍 Checking AI Model integrity (Git LFS)...")
    models_dir = "models"
    files_to_check = ["embeddings.pt", "processed_df.pkl"]
    
    needs_pull = False
    for f in files_to_check:
        path = os.path.join(models_dir, f)
        if os.path.exists(path):
            if os.path.getsize(path) < 1024:
                with open(path, 'r') as file:
                    if "version https://git-lfs.github.com" in file.read(100):
                        print(f"⚠️  Detected LFS pointer for {f}")
                        needs_pull = True
    
    if needs_pull:
        print("📡 Large files detected as pointers. Attempting automated Git LFS pull...")
        try:
            import subprocess
            result = subprocess.run(["git", "lfs", "pull"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Git LFS pull successful! Models are now hydrated.")
            else:
                print("❌ Git LFS pull failed.")
                print("💡 Please install Git LFS (https://git-lfs.github.com) and run 'git lfs pull' manually.")
        except FileNotFoundError:
            print("❌ 'git' command not found.")
            print("💡 Please install Git LFS and run 'git lfs pull' manually to download the 1GB+ model files.")
    else:
        print("✅ AI Models verified (Full binaries detected).")

if __name__ == "__main__":
    install_data()
    check_lfs()
