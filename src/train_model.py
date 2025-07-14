import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_save_model(X_train, y_train, X_test=None, y_test=None, model_path="models/model.pkl"):
    # ✅ Ensure the directory exists
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
    else:
        print(f"Directory already exists: {model_dir}")

    # ✅ Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("✅ Model training completed.")

    # ✅ Evaluate model if test data is provided
    if X_test is not None and y_test is not None:
        accuracy = accuracy_score(y_test, model.predict(X_test))
        print(f"✅ Test Accuracy: {accuracy:.4f}")

    # ✅ Save the trained model
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")

    return model