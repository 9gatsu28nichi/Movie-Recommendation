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

if __name__ == "__main__":
    install_data()
