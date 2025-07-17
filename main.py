from src.data_preprocessing import preprocess_data
from src.train_test_split import split_data
from train_model import train_and_save_model
from sklearn.feature_extraction.text import CountVectorizer

print("🚀 Step 1: Loading and preprocessing data...")
df, encoder = preprocess_data("data/gene_families.csv", k=3)
print(f"✅ Loaded {len(df)} sequences")

X_text, y = df["encoded_sequence"], df["label"]

print("🔤 Step 2: Vectorizing sequences...")
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_text)
print(f"✅ Vectorized shape: {X.shape}")

print("✂️ Step 3: Splitting data...")
X_train, X_test, y_train, y_test = split_data(X, y)
print(f"✅ Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

print("🧠 Step 4: Training model...")
train_and_save_model(X_train, y_train, X_test, y_test)
print("🎉 All done!")