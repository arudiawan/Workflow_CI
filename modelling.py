import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Set MLflow tracking local server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Eksperimen Diabetes Risk") # Changed Experiment Name

# 2. Enable Autolog
mlflow.sklearn.autolog()

def train_diabetes_model():
    print("⏳ Loading 'diabetes_preprocessing.csv'...")
    data = pd.read_csv("diabetes_preprocessing.csv") # Updated file name
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="Diabetes_RandomForest_Basic"):
        print("🚀 Training Random Forest Model...")
        model = RandomForestClassifier(n_estimators=100, max_depth=37, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        print(f"📊 Training Done! Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    train_diabetes_model()
