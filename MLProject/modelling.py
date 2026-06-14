import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

if os.environ.get('GITHUB_ACTIONS') == 'true':
    print("🤖 Berjalan di GitHub Actions: Menggunakan Local File Store.")
else:
    print("💻 Berjalan di Laptop Lokal: Menghubungkan ke http://127.0.0.1:5000")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("Eksperimen Diabetes Real Data")
mlflow.sklearn.autolog()

def train_diabetes_model():
    csv_path = "MLProject/diabetes_preprocessing.csv" if os.path.exists("MLProject/diabetes_preprocessing.csv") else "diabetes_preprocessing.csv"
    data = pd.read_csv(csv_path)
    
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Diabetes_RandomForest_Basic"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)

if __name__ == "__main__":
    train_diabetes_model()
