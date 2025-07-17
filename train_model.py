import os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model(X_train, y_train, X_test, y_test, model_path='models/model.pkl'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("⚙️ Training RandomForestClassifier...")
    model = RandomForestClassifier(n_jobs=-1, n_estimators=100)  # Use all cores
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Accuracy: {acc:.4f}")

    joblib.dump(model, model_path)
    print(f"💾 Model saved to: {model_path}")