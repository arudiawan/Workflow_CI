import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- PENYELARASAN PINTAR: SINKRONISASI MLFLOW RUN vs PYTHON DIRECT ---
is_github = os.environ.get('GITHUB_ACTIONS') == 'true'

if is_github:
    print("🤖 Terdeteksi lingkungan GitHub Actions: Mengandalkan sesi otomatis dari 'mlflow run'.")
else:
    print("💻 Terdeteksi lingkungan laptop lokal: Mengaktifkan server tracking port 5000.")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Hanya atur eksperimen secara manual jika dijalankan langsung (bukan via mlflow run)
    mlflow.set_experiment("Eksperimen Diabetes Real Data")

# Mengaktifkan Autolog otomatis (Wajib untuk merekam parameter & metrik)
mlflow.sklearn.autolog()

def train_diabetes_model():
    print("⏳ Memuat dataset 'diabetes_preprocessing.csv'...")
    csv_path = "MLProject/diabetes_preprocessing.csv" if os.path.exists("MLProject/diabetes_preprocessing.csv") else "diabetes_preprocessing.csv"
    data = pd.read_csv(csv_path)
    
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi model
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    # Logika eksekusi pelatihan berdasarkan cara pemanggilan skrip
    if is_github or mlflow.active_run():
        # Jika dipanggil lewat 'mlflow run', sesi sudah aktif. Langsung lakukan fit().
        print("🚀 Menjalankan pelatihan di dalam sesi aktif...")
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"📊 Akurasi Model pada Data Uji: {accuracy:.4f}")
    else:
        # Jika dijalankan manual via 'python modelling.py' di laptop, buat blok run manual.
        with mlflow.start_run(run_name="Diabetes_RandomForest_Basic"):
            print("🚀 Menjalankan pelatihan via blok start_run manual...")
            model.fit(X_train, y_train)
            accuracy = model.score(X_test, y_test)
            print(f"📊 Akurasi Model pada Data Uji: {accuracy:.4f}")

if __name__ == "__main__":
    train_diabetes_model()
