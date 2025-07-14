from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model(X_train, y_train, X_test, y_test, model_path='models/model.pkl'):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Model accuracy: {accuracy:.4f}')
    joblib.dump(model, model_path)
    return model